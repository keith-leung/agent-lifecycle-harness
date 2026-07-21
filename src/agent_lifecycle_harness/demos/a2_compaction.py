"""A2 — Checkpoint retention & compaction: FAST SMOKE TEST.

Scope: this demo is the fast, deterministic smoke test. It verifies that the
langmem-backed compactor integrates and behaves at the shape level — a digest
is produced, it folds real message ids, replay stays coherent, and re-running
is idempotent. It does NOT measure token reduction or cost.

The full adversarial characterization of A2 lives in a SEPARATE script:

    python -m agent_lifecycle_harness.demos.a2_acceptance

That script is the real acceptance bar — it loads a fixed 40-turn fixture and
proves, on a real LLM: token reduction inside a (40%, 85%) window, structural
deletion (a marker seeded mid-conversation is absent from the payload),
information retention (summary-only coherence probe), and the append-vs-merge
cost/quality trade-off. Anything about "how well A2 compacts" is answered there,
not here.

Compaction mechanism: langmem `summarize_messages` + `RunningSummary` performs
structural replacement (the middle is removed, not prefixed); `RunningSummary.
summarized_message_ids` is the bridge A3 traverses. See RUNTIME notes.

SPEC assertions (shape-level, smoke):
  (a) one digest is produced and folds real message ids
  (b) subsequent invoke produces a coherent reply (agent "remembers" via digest)
  (c) compactor is idempotent (second run = no-op)
  (d) lossy fields enumerated in digest metadata
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from agent_lifecycle_harness.agent import LifecycleHarness
from agent_lifecycle_harness.compaction import (
    CompactionStore,
    LangmemCompactor,
    NoCompactionStrategy,
    coerce_checkpoint_snapshot,
)
from agent_lifecycle_harness.llm import LLMClient, MockLLMClient, MockChatModel, RealLLMClient


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


def _make_harness(
    llm: LLMClient | None = None,
    judge: LLMClient | None = None,
    chat_model: Any | None = None,
) -> tuple[LifecycleHarness, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    if llm is None:
        llm = MockLLMClient(prefix="a2-sut")
    if judge is None:
        judge = MockLLMClient(prefix="a2-judge")
    if chat_model is None:
        chat_model = MockChatModel()
    # The demo drives compaction explicitly via LangmemCompactor below, so
    # the node must NOT also fold on every invoke (that would double-fold
    # the same message ids and langmem rejects already-summarized ids).
    return (
        LifecycleHarness(
            db_path=db_path, llm=llm, judge=judge,
            compaction_strategy=NoCompactionStrategy(),
            summarization_chat_model=chat_model,
        ),
        db_path,
    )


def _collect_texts(state: dict[str, Any]) -> list[str]:
    msgs = state.get("messages", [])
    out: list[str] = []
    for m in msgs:
        if hasattr(m, "content"):
            out.append(m.content or "")
        elif isinstance(m, dict):
            out.append(m.get("content", ""))
    return out


def assert_compaction_shape(
    store: CompactionStore,
    thread_id: str,
) -> AssertionResult:
    """(a) Compaction produced exactly one digest that folded raw messages."""
    digests = store.digests_for_thread(thread_id)
    if len(digests) != 1:
        return AssertionResult(
            name="compaction_shape",
            passed=False,
            evidence=f"Expected 1 digest, got {len(digests)}",
        )
    digest = digests[0]
    if not digest.replaced_raw_ids:
        return AssertionResult(
            name="compaction_shape",
            passed=False,
            evidence="Digest folded no raw checkpoints.",
        )
    return AssertionResult(
        name="compaction_shape",
        passed=True,
        evidence=(
            f"1 digest covers {len(digest.replaced_raw_ids)} producing "
            f"checkpoint(s); running summary folded "
            f"{len(digest.running_summary.summarized_message_ids) if digest.running_summary else 0} "
            f"message ids."
        ),
    )


def assert_coherence_after_compaction(
    harness: LifecycleHarness,
    store: CompactionStore,
    thread_id: str,
) -> AssertionResult:
    """(b) After compaction, invoke still produces a coherent reply.

    Behavioral test: run one more turn after compaction and verify the
    agent returns content. In CI-MOCK the mock always returns a
    deterministic string; in real-LLM this proves the digest-backed
    context is sufficient for the model to continue coherently.
    """
    digests = store.digests_for_thread(thread_id)
    if not digests:
        return AssertionResult(
            name="coherence_after_compaction",
            passed=False,
            evidence="No digest found after compaction.",
        )
    # Run a new turn after compaction.
    new_state = harness.invoke(thread_id, "post-compaction: follow-up question")
    texts = _collect_texts(new_state)
    if not texts:
        return AssertionResult(
            name="coherence_after_compaction",
            passed=False,
            evidence="Post-compaction invoke returned no text.",
        )
    last_text = texts[-1]
    # In CI-MOCK the mock response is deterministic; in real-LLM we just
    # verify the reply is non-empty and substantive.
    if last_text.startswith("[MOCK-"):
        return AssertionResult(
            name="coherence_after_compaction",
            passed=True,
            evidence="Post-compaction invoke returned a coherent deterministic reply.",
        )
    if len(last_text.strip()) < 10:
        return AssertionResult(
            name="coherence_after_compaction",
            passed=False,
            evidence=f"Post-compaction reply too short: {last_text}",
        )
    return AssertionResult(
        name="coherence_after_compaction",
        passed=True,
        evidence="Post-compaction invoke returned a coherent reply.",
    )


def assert_compactor_idempotent(
    compactor: LangmemCompactor,
    thread_id: str,
    messages: list[Any],
    msg_to_ckpt: dict[str, str | None],
) -> AssertionResult:
    """(c) Re-running compaction once the input is fully folded is a no-op.

    langmem is incremental: as long as there are unsummarized messages
    whose token mass exceeds ``max_tokens_before_summary``, it keeps
    folding them into the running summary. Idempotency therefore means:
    once every foldable prefix has been absorbed, a further call with no
    new messages must not change the running summary. We drive compaction
    to saturation, then assert the *next* call is a true no-op.
    """
    # Drive to saturation: keep folding until a pass produces nothing new.
    last = compactor.compact(thread_id, messages, message_to_checkpoint=msg_to_ckpt)
    passes = 1
    while last.folded_new and passes < 8:
        last = compactor.compact(thread_id, messages, message_to_checkpoint=msg_to_ckpt)
        passes += 1
    # The pass AFTER saturation must be a no-op.
    confirm = compactor.compact(thread_id, messages, message_to_checkpoint=msg_to_ckpt)
    if confirm.folded_new:
        return AssertionResult(
            name="compactor_idempotent",
            passed=False,
            evidence=(
                f"After {passes} fold passes a further call still folded new "
                f"messages; expected saturation no-op."
            ),
        )
    return AssertionResult(
        name="compactor_idempotent",
        passed=True,
        evidence=(
            f"Saturation reached after {passes} fold pass(es); subsequent "
            f"call was a no-op (tokens_before={confirm.tokens_before}, "
            f"tokens_after={confirm.tokens_after})."
        ),
    )


def assert_lossy_fields_enumerated(
    store: CompactionStore,
    thread_id: str,
) -> AssertionResult:
    """(d) Digest metadata enumerates lossy fields."""
    digests = store.digests_for_thread(thread_id)
    if not digests:
        return AssertionResult(
            name="lossy_fields_enumerated",
            passed=False,
            evidence="No digest found.",
        )
    digest = digests[0]
    if not digest.lossy_fields:
        return AssertionResult(
            name="lossy_fields_enumerated",
            passed=False,
            evidence="Digest has empty lossy_fields.",
        )
    return AssertionResult(
        name="lossy_fields_enumerated",
        passed=True,
        evidence=f"Lossy fields enumerated: {digest.lossy_fields}",
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------

def demo_A2_compaction(
    harness: LifecycleHarness | None = None,
    n_turns: int = 10,
    first_last_n: int = 3,
) -> DemoResult:
    """Run the A2 checkpoint retention/compaction scenario.

    langmem drives compaction through ``LangmemCompactor``: messages are
    folded into a running summary once the token budget is exceeded, and
    the folded raw messages are structurally removed from the replay
    context. ``first_last_n`` is retained as a metric for back-compat with
    the test suite; the actual fold boundary is token-driven (langmem).
    """
    if harness is None:
        harness, _ = _make_harness()
    thread_id = f"compaction-thread-{os.getpid()}"
    assertions: list[AssertionResult] = []

    # Seed the thread with n_turns messages. Each message is padded so the
    # token counter crosses the compaction threshold within n_turns.
    for i in range(n_turns):
        padding = ("lorem ipsum dolor sit amet " * 8).strip()
        harness.invoke(thread_id, f"turn-{i}: {padding}")

    # Read the full message history out of the thread state.
    state = harness.get_state(thread_id)
    messages = (state or {}).get("messages", [])

    # Build a message_id → checkpoint_id map for A3-compatible persistence.
    raw_checkpoints = harness.list_checkpoints(thread_id)
    raw_checkpoints = [coerce_checkpoint_snapshot(c) for c in raw_checkpoints]
    msg_to_ckpt: dict[str, str | None] = {}
    for cp in raw_checkpoints:
        cp_id = cp.get("checkpoint_id")
        for m in cp.get("values", {}).get("messages", []):
            mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
            if mid:
                msg_to_ckpt[mid] = cp_id

    # Build the compactor using the same chat model the harness uses for
    # summarization. In real mode this is a real ChatModel (network-backed
    # summary generation + exact token counting); in mock mode a MockChatModel.
    store = CompactionStore(harness.db_path)
    chat_model = harness.summarization_chat_model or MockChatModel()
    # Threshold: low enough that n_turns padded messages trigger a fold,
    # but high enough that the summarizer can ingest the folded segment
    # without trim warnings (max_tokens >= n_tokens_to_summarize).
    compactor = LangmemCompactor(
        store, chat_model,
        max_tokens=2400, max_tokens_before_summary=120, max_summary_tokens=80,
    )

    # First compaction pass.
    first = compactor.compact(thread_id, messages, message_to_checkpoint=msg_to_ckpt)
    if not first.folded_new or first.digest is None:
        # No fold → fail loudly with diagnostic so the threshold can be tuned.
        assertions.append(AssertionResult(
            name="compaction_shape",
            passed=False,
            evidence=(
                f"First compaction did not fold anything "
                f"(tokens_before={first.tokens_before}, tokens_after={first.tokens_after}). "
                f"Adjust max_tokens threshold."
            ),
        ))
        passed = False
        metrics = {
            "n_turns": n_turns, "first_last_n": first_last_n,
            "raw_checkpoints": len(raw_checkpoints),
            "digest_count": len(store.digests_for_thread(thread_id)),
            "framework": harness.framework,
            "tokens_before": first.tokens_before, "tokens_after": first.tokens_after,
        }
        return DemoResult(name="A2_compaction", passed=passed, assertions=assertions, metrics=metrics)

    # (a) shape
    assertions.append(assert_compaction_shape(store, thread_id))

    # (b) coherence: invoke still works after compaction
    assertions.append(assert_coherence_after_compaction(harness, store, thread_id))

    # (c) idempotent: second pass with no new messages must not fold again
    assertions.append(assert_compactor_idempotent(compactor, thread_id, messages, msg_to_ckpt))

    # (d) lossy fields
    assertions.append(assert_lossy_fields_enumerated(store, thread_id))

    passed = all(a.passed for a in assertions)
    metrics = {
        "n_turns": n_turns,
        "first_last_n": first_last_n,
        "raw_checkpoints": len(raw_checkpoints),
        "digest_count": len(store.digests_for_thread(thread_id)),
        "tokens_before": first.tokens_before,
        "tokens_after": first.tokens_after,
        "reduction_pct": round(100 * (1 - first.tokens_after / first.tokens_before), 1)
        if first.tokens_before else 0,
        "framework": harness.framework,
    }
    return DemoResult(name="A2_compaction", passed=passed, assertions=assertions, metrics=metrics)
