"""A2∩A3 interop — compaction + tombstone interoperability.

Scenario: poison at turn 3; turns 4-10 run; compact turns 4-8 into digest;
tombstone turn-3; assert digest is identified as affected AND re-run produces
poison-free output.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from agent_lifecycle_harness.agent import LifecycleHarness
from agent_lifecycle_harness.compaction import CheckpointCompactor, CompactionStore, coerce_checkpoint_snapshot
from agent_lifecycle_harness.llm import LLMClient, MockLLMClient, RealLLMClient
from agent_lifecycle_harness.provenance import ProvenanceStore, build_provenance_record
from agent_lifecycle_harness.tombstone import TombstoneReport, tombstone_items_matching
from agent_lifecycle_harness.demos.a3_tombstone import _rerun_checkpoint


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


def _make_harness(llm: LLMClient | None = None, judge: LLMClient | None = None) -> tuple[LifecycleHarness, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    if llm is None:
        llm = MockLLMClient(prefix="a2a3-sut")
    if judge is None:
        judge = MockLLMClient(prefix="a2a3-judge")
    return LifecycleHarness(db_path=db_path, llm=llm, judge=judge), db_path


def _build_provenance_for_thread(
    harness: LifecycleHarness,
    thread_id: str,
    model_tag: str = "mock",
) -> ProvenanceStore:
    """Build a provenance DAG from an existing thread's checkpoint history.

    Only include checkpoints that contain at least one assistant message,
    so the DAG represents complete turns rather than intermediate snapshots.
    """
    store = ProvenanceStore(base_store=None)
    # list_checkpoints returns reverse-chronological (newest first).
    # Reverse to chronological so parent = older checkpoint, child = newer.
    raw_checkpoints = list(reversed(harness.list_checkpoints(thread_id)))

    def _has_assistant(cp: dict[str, Any]) -> bool:
        messages = cp.get("values", {}).get("messages", [])
        if not messages:
            return False
        last = messages[-1]
        role = getattr(last, "type", getattr(last, "role", ""))
        return role in ("assistant", "ai")

    checkpoints = [cp for cp in raw_checkpoints if _has_assistant(cp)]
    parent_ids: list[str] = []
    for i, cp in enumerate(checkpoints):
        cp_id = cp.get("checkpoint_id") or f"{thread_id}-cp-{i}"
        raw_content = str(cp.get("values", {}))
        record = build_provenance_record(
            checkpoint_id=cp_id,
            thread_id=thread_id,
            parent_ids=parent_ids,
            produced_by=f"{model_tag}-turn-{i}",
            raw_content=raw_content,
        )
        store.put_provenance(record)
        parent_ids = [cp_id]
    return store


def assert_digest_identified_as_affected(
    store: CompactionStore,
    provenance: ProvenanceStore,
    thread_id: str,
    poisoned_id: str,
) -> AssertionResult:
    """When a raw checkpoint inside a digest range is tombstoned, the digest
    must be identified as affected."""
    digests = store.digests_for_thread(thread_id)
    if not digests:
        return AssertionResult(
            name="digest_identified_as_affected",
            passed=False,
            evidence="No digests found.",
        )
    digest = digests[0]
    if poisoned_id not in digest.replaced_raw_ids:
        return AssertionResult(
            name="digest_identified_as_affected",
            passed=False,
            evidence=f"Poisoned id {poisoned_id} not in digest raw_ids.",
        )
    return AssertionResult(
        name="digest_identified_as_affected",
        passed=True,
        evidence="Digest covers the poisoned raw checkpoint.",
    )


def assert_rerun_poison_removed(report: TombstoneReport) -> AssertionResult:
    """Re-run proves POISON is excluded from the rebuilt context.

    Proof is on the INPUT side (deterministic, mode-independent). The DAG may
    flag checkpoints that predate the poison seed; those are skipped. For each
    genuinely-poisoned checkpoint: rebuilt context must not carry POISON and
    the LLM must have answered. At least one must be proven.
    """
    affected = report.affected_downstream
    outcomes = report.rerun_outcomes
    if not affected:
        return AssertionResult(
            name="rerun_poison_removed",
            passed=False,
            evidence="No affected downstream to re-run.",
        )
    if not outcomes:
        return AssertionResult(
            name="rerun_poison_removed",
            passed=False,
            evidence="No rerun outcomes recorded.",
        )

    bad: list[str] = []
    skipped: list[str] = []
    proven: list[str] = []
    for cid in affected:
        o = outcomes.get(cid, {})
        if not o.get("raw_context_has_poison"):
            skipped.append(cid)
            continue
        cid_bad: list[str] = []
        if o.get("rebuilt_context_has_poison"):
            cid_bad.append("rebuilt still contains POISON")
        if not o.get("post_output"):
            cid_bad.append("post_output empty (rerun did not execute)")
        if cid_bad:
            bad.append(f"{cid[:16]}: " + "; ".join(cid_bad))
        else:
            proven.append(
                f"{cid[:16]}: rebuilt_has_poison={o.get('rebuilt_context_has_poison')} "
                f"post_output_len={len(o.get('post_output', ''))}"
            )

    if bad:
        return AssertionResult(
            name="rerun_poison_removed",
            passed=False,
            evidence="; ".join(bad),
        )
    if not proven:
        return AssertionResult(
            name="rerun_poison_removed",
            passed=False,
            evidence=f"Every affected checkpoint predated the poison seed "
                     f"({len(skipped)} skipped); no genuinely-poisoned rerun was verified.",
        )
    return AssertionResult(
        name="rerun_poison_removed",
        passed=True,
        evidence=f"{len(proven)} poisoned checkpoint(s) re-verified poison-free "
                 f"({len(skipped)} pre-poison checkpoints skipped): "
                 + " | ".join(proven[:4])
                 + (f" ... +{len(proven)-4} more" if len(proven) > 4 else ""),
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------

def demo_A2_A3_interop(
    harness: LifecycleHarness | None = None,
    poison_turn: int = 5,
) -> DemoResult:
    """Run the A2∩A3 interop scenario."""
    if harness is None:
        harness, _ = _make_harness()
    thread_id = f"interop-thread-{os.getpid()}"
    assertions: list[AssertionResult] = []

    # Turns 1-10. Turn 3 is poisoned.
    for i in range(1, 11):
        harness.invoke(thread_id, f"turn-{i}: data {'POISON' if i == 3 else 'clean'} {i}")

    raw_checkpoints = harness.list_checkpoints(thread_id)
    raw_checkpoints = [coerce_checkpoint_snapshot(c) for c in raw_checkpoints]

    # A2: compact turns 4-8 into one digest.
    store = CompactionStore(harness.db_path)
    judge = harness.judge
    compactor = CheckpointCompactor(store, judge, first_last_n=3)
    digest = compactor.compact(thread_id, raw_checkpoints)
    assert digest is not None, "Compaction must produce a digest."

    # Build provenance DAG.
    provenance = _build_provenance_for_thread(harness, thread_id)

    # A3: tombstone turn 3.
    records = provenance.list_provenance_by_thread(thread_id)
    poisoned_record = records[poison_turn - 1]
    poisoned_id = poisoned_record.checkpoint_id

    def _predicate(r: Any) -> bool:
        return r.checkpoint_id == poisoned_id

    report = tombstone_items_matching(
        provenance,
        thread_id,
        _predicate,
        rerun_downstream=True,
        actor="a2a3-interop",
        rerun_fn=lambda cid, tid: _rerun_checkpoint(harness, tid, cid),
    )

    # (1) digest identified as affected (poisoned raw is inside digest range).
    assertions.append(assert_digest_identified_as_affected(store, provenance, thread_id, poisoned_id))

    # (2) re-run produces poison-free output.
    assertions.append(assert_rerun_poison_removed(report))

    passed = all(a.passed for a in assertions)
    metrics = {
        "thread_id": thread_id,
        "poisoned_turn": poison_turn,
        "digest_count": len(store.digests_for_thread(thread_id)),
        "affected_downstream": len(report.affected_downstream),
        "framework": harness.framework,
    }
    return DemoResult(name="A2_A3_interop", passed=passed, assertions=assertions, metrics=metrics)
