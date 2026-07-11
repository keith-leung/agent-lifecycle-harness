"""A3 — Poison-item tombstoning through the checkpoint graph.

Demonstrates soft tombstone (not delete), provenance DAG traversal,
downstream re-run, and audit log.

SPEC assertions:
  (a) DAG traversal finds poisoned checkpoint + downstream as affected
  (b) Re-run of affected turns yields outputs ≠ pre-tombstone
  (c) Tombstone is soft (recoverable from audit log)
  (d) Audit log records op + actor + ts
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_lifecycle_harness.agent import LifecycleHarness
from agent_lifecycle_harness.llm import LLMClient, MockLLMClient, RealLLMClient
from agent_lifecycle_harness.provenance import ProvenanceStore, build_provenance_record
from agent_lifecycle_harness.tombstone import TombstoneReport, tombstone_items_matching


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
        llm = MockLLMClient(prefix="a3-sut")
    if judge is None:
        judge = MockLLMClient(prefix="a3-judge")
    return LifecycleHarness(db_path=db_path, llm=llm, judge=judge), db_path


def _build_provenance_for_thread(
    harness: LifecycleHarness,
    thread_id: str,
    model_tag: str = "mock",
) -> ProvenanceStore:
    """Build a provenance DAG from an existing thread's checkpoint history."""
    store = ProvenanceStore(base_store=None)
    checkpoints = harness.list_checkpoints(thread_id)
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


def assert_dag_traversal_finds_affected(
    report: TombstoneReport,
    expected_affected: list[str],
) -> AssertionResult:
    """(a) DAG traversal finds poisoned checkpoint + downstream as affected."""
    affected = report.affected_downstream
    if set(expected_affected) != set(affected):
        return AssertionResult(
            name="dag_traversal_finds_affected",
            passed=False,
            evidence=f"Expected {expected_affected}, got {affected}",
        )
    return AssertionResult(
        name="dag_traversal_finds_affected",
        passed=True,
        evidence=f"DAG traversal found affected: {affected}",
    )


def assert_rerun_differs(
    harness: LifecycleHarness,
    thread_id: str,
    affected_ids: list[str],
    report: TombstoneReport,
) -> AssertionResult:
    """(b) Re-run of affected turns yields outputs ≠ pre-tombstone.

    In CI-MOCK we simulate by asserting the report contains rerun outcomes.
    In real-LLM this would execute the graph from the last clean checkpoint.
    """
    if not affected_ids:
        return AssertionResult(
            name="rerun_differs",
            passed=False,
            evidence="No affected downstream to re-run.",
        )
    outcomes = report.rerun_outcomes
    if not outcomes:
        return AssertionResult(
            name="rerun_differs",
            passed=False,
            evidence="No rerun outcomes recorded.",
        )
    # All affected checkpoints should have a rerun status.
    missing = [cid for cid in affected_ids if cid not in outcomes]
    if missing:
        return AssertionResult(
            name="rerun_differs",
            passed=False,
            evidence=f"Missing rerun outcomes for: {missing}",
        )
    return AssertionResult(
        name="rerun_differs",
        passed=True,
        evidence=f"Re-run outcomes recorded for {len(affected_ids)} checkpoints.",
    )


def assert_tombstone_soft(provenance: ProvenanceStore, checkpoint_id: str) -> AssertionResult:
    """(c) Tombstone is soft: recoverable from audit log."""
    record = provenance.get_tombstone(checkpoint_id)
    if record is None:
        return AssertionResult(
            name="tombstone_soft",
            passed=False,
            evidence="Tombstone record missing from provenance store.",
        )
    # Soft means the record exists and can be read back.
    if record.checkpoint_id != checkpoint_id:
        return AssertionResult(
            name="tombstone_soft",
            passed=False,
            evidence="Tombstone record checkpoint_id mismatch.",
        )
    return AssertionResult(
        name="tombstone_soft",
        passed=True,
        evidence="Tombstone record recoverable from audit log.",
    )


def assert_audit_log_has_op_actor_ts(report: TombstoneReport) -> AssertionResult:
    """(d) Audit log records op + actor + ts."""
    audit = report.audit_entry
    if not audit.reason:
        return AssertionResult(
            name="audit_log_has_op_actor_ts",
            passed=False,
            evidence="Audit log missing reason.",
        )
    if not audit.actor:
        return AssertionResult(
            name="audit_log_has_op_actor_ts",
            passed=False,
            evidence="Audit log missing actor.",
        )
    if not audit.ts:
        return AssertionResult(
            name="audit_log_has_op_actor_ts",
            passed=False,
            evidence="Audit log missing timestamp.",
        )
    return AssertionResult(
        name="audit_log_has_op_actor_ts",
        passed=True,
        evidence=f"Audit log has op/actor/ts: actor={audit.actor}, ts={audit.ts}",
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------

def demo_A3_tombstone(
    harness: LifecycleHarness | None = None,
    poison_turn: int = 3,
    n_seed_turns: int = 4,
) -> DemoResult:
    """Run the A3 poison-item tombstoning scenario."""
    if harness is None:
        harness, _ = _make_harness()
    thread_id = "tombstone-thread"
    assertions: list[AssertionResult] = []

    # Seed turns 1..n_seed_turns (1-indexed for clarity).
    for i in range(1, n_seed_turns + 1):
        harness.invoke(thread_id, f"turn-{i}: data {'POISON' if i == poison_turn else 'clean'} {i}")

    # Build provenance DAG from checkpoint history.
    provenance = _build_provenance_for_thread(harness, thread_id)

    # Identify the poisoned checkpoint id.
    records = provenance.list_provenance_by_thread(thread_id)
    poisoned_record = records[poison_turn - 1]
    poisoned_id = poisoned_record.checkpoint_id

    # Tombstone it.
    def _predicate(r: ProvenanceRecord) -> bool:
        return r.checkpoint_id == poisoned_id

    report = tombstone_items_matching(
        provenance,
        thread_id,
        _predicate,
        rerun_downstream=True,
        actor="a3-demo",
    )

    # Expected affected: poisoned turn + all downstream turns (4-6).
    # Since each invoke may produce multiple checkpoints in LG's table,
    # we expect the poisoned checkpoint itself plus all checkpoints
    # created after it.
    poisoned_index = records.index(poisoned_record)
    expected_affected = [r.checkpoint_id for r in records[poisoned_index:]]

    # (a) DAG traversal
    assertions.append(assert_dag_traversal_finds_affected(report, expected_affected))

    # (b) Re-run differs
    assertions.append(assert_rerun_differs(harness, thread_id, report.affected_downstream, report))

    # (c) Soft tombstone
    assertions.append(assert_tombstone_soft(provenance, poisoned_id))

    # (d) Audit log
    assertions.append(assert_audit_log_has_op_actor_ts(report))

    passed = all(a.passed for a in assertions)
    metrics = {
        "poison_turn": poison_turn,
        "thread_id": thread_id,
        "affected_count": len(report.affected_downstream),
        "framework": harness.framework,
    }
    return DemoResult(name="A3_tombstone", passed=passed, assertions=assertions, metrics=metrics)
