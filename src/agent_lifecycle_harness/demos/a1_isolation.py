"""A1 — Concurrent-user isolation demo.

Demonstrates LangGraph's thread-scoped persistence, the accidental-reuse
bug when two users share a `thread_id`, and the namespaced-id fix.

SPEC assertions:
  (a) each thread resumes only its own history
  (b) no thread's state contains a key seeded only by another thread
  (c) per-thread checkpoint counts independent
  (d) two concurrent writers to the SAME thread serialize (no torn writes)
  (e) accidental-reuse bug reproduced then fixed (id namespacing)
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from agent_lifecycle_harness.agent import LifecycleHarness
from agent_lifecycle_harness.llm import LLMClient, MockLLMClient, RealLLMClient


@dataclass
class AssertionResult:
    name: str
    passed: bool
    evidence: str


@dataclass
class DemoResult:
    name: str
    passed: bool
    assertions: list[AssertionResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_harness(
    llm: LLMClient | None = None,
    judge: LLMClient | None = None,
) -> tuple[LifecycleHarness, str]:
    """Create a harness backed by a temporary SQLite file. Returns (harness, path)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    if llm is None:
        llm = MockLLMClient(prefix="a1-sut")
    if judge is None:
        judge = MockLLMClient(prefix="a1-judge")
    return LifecycleHarness(db_path=db_path, llm=llm, judge=judge), db_path


def _user_msg(thread_id: str, seed: str) -> str:
    return f"[{thread_id}] seed={seed} unique-key={thread_id}:{seed}"


def _messages_from_state(state: dict[str, Any]) -> list[dict[str, str]]:
    msgs = state.get("messages", [])
    out: list[dict[str, str]] = []
    for m in msgs:
        if hasattr(m, "content"):
            out.append({"role": getattr(m, "type", "user"), "content": m.content})
        elif isinstance(m, dict):
            out.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return out


def _collect_texts(state: dict[str, Any]) -> list[str]:
    return [m.get("content", "") for m in _messages_from_state(state)]


def _texts_contain(texts: list[str], needle: str) -> bool:
    return any(needle in t for t in texts)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_resume_own_history_only(
    h: LifecycleHarness,
    thread_ids: Sequence[str],
    seed_by_thread: dict[str, str],
) -> AssertionResult:
    """(a) Each thread resumes only its own history."""
    for tid in thread_ids:
        state = h.get_state(tid)
        assert state is not None, f"state missing for {tid}"
        texts = _collect_texts(state)
        seed = seed_by_thread[tid]
        expected_seed = _user_msg(tid, seed)
        if not _texts_contain(texts, expected_seed):
            return AssertionResult(
                name="resume_own_history_only",
                passed=False,
                evidence=f"thread {tid} missing its own seed message",
            )
        # Ensure no other thread's seed is present.
        for other in thread_ids:
            if other == tid:
                continue
            other_seed = _user_msg(other, seed_by_thread[other])
            if _texts_contain(texts, other_seed):
                return AssertionResult(
                    name="resume_own_history_only",
                    passed=False,
                    evidence=f"thread {tid} leaked seed from {other}",
                )
    return AssertionResult(
        name="resume_own_history_only",
        passed=True,
        evidence=f"All {len(thread_ids)} threads contain only their own history.",
    )


def assert_no_cross_thread_key_leak(
    h: LifecycleHarness,
    thread_ids: Sequence[str],
    seed_by_thread: dict[str, str],
) -> AssertionResult:
    """(b) No thread's state contains a key seeded only by another thread."""
    for tid in thread_ids:
        state = h.get_state(tid)
        assert state is not None
        texts = _collect_texts(state)
        for other in thread_ids:
            if other == tid:
                continue
            unique_key = f"{other}:{seed_by_thread[other]}"
            if _texts_contain(texts, unique_key):
                return AssertionResult(
                    name="no_cross_thread_key_leak",
                    passed=False,
                    evidence=f"thread {tid} contains unique key from {other}",
                )
    return AssertionResult(
        name="no_cross_thread_key_leak",
        passed=True,
        evidence="No cross-thread key leakage detected.",
    )


def assert_independent_checkpoint_counts(
    h: LifecycleHarness,
    thread_ids: Sequence[str],
) -> AssertionResult:
    """(c) Per-thread checkpoint counts are independent.
    
    NOTE: Same operation count on different threads naturally produces the
    same checkpoint count. Independence here means each thread's checkpoints
    reference only that thread's data, not that counts must differ.
    """
    counts: dict[str, int] = {}
    for tid in thread_ids:
        counts[tid] = len(h.list_checkpoints(tid))
    # Sanity: each thread has at least one checkpoint.
    if any(c == 0 for c in counts.values()):
        return AssertionResult(
            name="independent_checkpoint_counts",
            passed=False,
            evidence=f"Some thread has zero checkpoints: {counts}",
        )
    # Verify checkpoint contents are thread-scoped by inspecting stored state.
    for tid in thread_ids:
        state = h.get_state(tid)
        assert state is not None
        meta = state.get("_harness_meta", {})
        if meta.get("thread_id") != tid:
            return AssertionResult(
                name="independent_checkpoint_counts",
                passed=False,
                evidence=f"Checkpoint for {tid} has mismatched thread_id in meta.",
            )
    return AssertionResult(
        name="independent_checkpoint_counts",
        passed=True,
        evidence=f"Checkpoint counts are thread-scoped: {counts}",
    )


def assert_same_thread_writers_serialize(
    h: LifecycleHarness,
    thread_id: str,
) -> AssertionResult:
    """(d) Two concurrent writers to the SAME thread serialize without torn writes.

    Timeouts are sized for real-LLM latency (each writer makes one live model
    call; under a remote provider a single invoke can take tens of seconds).
    The 15s budget that passes in mock mode is too tight for real mode and
    produces a false "did not complete" failure that has nothing to do with
    serialization correctness.
    """
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    results: list[dict[str, Any]] = [None, None]  # type: ignore[list-item]

    def _writer(idx: int, msg: str) -> None:
        try:
            barrier.wait(timeout=30)
            results[idx] = h.invoke(thread_id, msg)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=_writer, args=(0, "writer-a: hello"))
    t2 = threading.Thread(target=_writer, args=(1, "writer-b: world"))
    t1.start()
    t2.start()
    t1.join(timeout=120)
    t2.join(timeout=120)

    if errors:
        return AssertionResult(
            name="same_thread_writers_serialize",
            passed=False,
            evidence=f"Concurrent writers raised: {errors}",
        )
    if any(r is None for r in results):
        return AssertionResult(
            name="same_thread_writers_serialize",
            passed=False,
            evidence="A concurrent writer did not complete.",
        )
    # Both writes completed; state should contain both messages.
    state = h.get_state(thread_id)
    assert state is not None
    texts = _collect_texts(state)
    if not _texts_contain(texts, "writer-a: hello") or not _texts_contain(texts, "writer-b: world"):
        return AssertionResult(
            name="same_thread_writers_serialize",
            passed=False,
            evidence=f"Missing messages after concurrent writes. Got: {texts}",
        )
    return AssertionResult(
        name="same_thread_writers_serialize",
        passed=True,
        evidence="Concurrent writers to same thread completed without torn writes.",
    )


def assert_accidental_reuse_then_fix(h: LifecycleHarness) -> AssertionResult:
    """(e) Reproduce the accidental-reuse bug, then show namespaced IDs fix it."""
    # Bug: two users share the bare id "shared-session".
    h.invoke("shared-session", "user-a: secret-alpha")
    h.invoke("shared-session", "user-b: secret-beta")

    state_shared = h.get_state("shared-session")
    assert state_shared is not None
    texts = _collect_texts(state_shared)
    leaked = _texts_contain(texts, "secret-alpha") and _texts_contain(texts, "secret-beta")
    if not leaked:
        return AssertionResult(
            name="accidental_reuse_then_fix",
            passed=False,
            evidence="Bug did not reproduce: shared-session should have mixed both users' secrets.",
        )

    # Fix: namespaced ids keep users isolated.
    h2, _ = _make_harness(llm=h.llm, judge=h.judge)
    h2.invoke("user:u1:sess:s1", "user-a: secret-alpha")
    h2.invoke("user:u2:sess:s2", "user-b: secret-beta")
    s1 = h2.get_state("user:u1:sess:s1")
    s2 = h2.get_state("user:u2:sess:s2")
    assert s1 is not None and s2 is not None
    t1 = _collect_texts(s1)
    t2 = _collect_texts(s2)
    fixed = _texts_contain(t1, "secret-alpha") and not _texts_contain(t1, "secret-beta") and \
            _texts_contain(t2, "secret-beta") and not _texts_contain(t2, "secret-alpha")
    if not fixed:
        return AssertionResult(
            name="accidental_reuse_then_fix",
            passed=False,
            evidence="Namespaced ids still leaked.",
        )
    return AssertionResult(
        name="accidental_reuse_then_fix",
        passed=True,
        evidence="Bug reproduced on bare id; namespaced ids isolate users.",
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------

def demo_A1_isolation(
    harness: LifecycleHarness | None = None,
    n_threads: int = 3,
    n_seeds: int = 2,
) -> DemoResult:
    """Run the A1 concurrent-user isolation scenario."""
    if harness is None:
        harness, path = _make_harness()
    else:
        path = str(harness.db_path)

    thread_ids = [f"thread-{i}" for i in range(n_threads)]
    seed_by_thread: dict[str, str] = {}
    assertions: list[AssertionResult] = []

    # Seed each thread with unique messages.
    for tid in thread_ids:
        for s in range(n_seeds):
            seed = f"seed-{s}"
            seed_by_thread[tid] = seed
            harness.invoke(tid, _user_msg(tid, seed))

    # (a) resume own history only
    assertions.append(assert_resume_own_history_only(harness, thread_ids, seed_by_thread))

    # (b) no cross-thread key leak
    assertions.append(assert_no_cross_thread_key_leak(harness, thread_ids, seed_by_thread))

    # (c) independent checkpoint counts
    assertions.append(assert_independent_checkpoint_counts(harness, thread_ids))

    # (d) same-thread writers serialize
    assertions.append(assert_same_thread_writers_serialize(harness, thread_ids[0]))

    # (e) accidental reuse + fix
    assertions.append(assert_accidental_reuse_then_fix(harness))

    passed = all(a.passed for a in assertions)
    metrics = {
        "threads": n_threads,
        "seeds_per_thread": n_seeds,
        "db_path": path,
        "framework": harness.framework,
    }
    return DemoResult(name="A1_isolation", passed=passed, assertions=assertions, metrics=metrics)
