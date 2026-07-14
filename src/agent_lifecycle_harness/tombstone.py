"""Tombstone mechanism + provenance DAG traversal.

Soft-delete: a poisoned checkpoint is marked in the provenance store,
not deleted. Downstream checkpoints whose parent-chain transitively
includes the poisoned one are flagged for re-run.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agent_lifecycle_harness.provenance import (
    ProvenanceRecord,
    ProvenanceStore,
    TombstoneRecord,
    build_provenance_record,
)


@dataclass
class TombstoneReport:
    tombstoned_id: str
    affected_downstream: list[str]
    rerun_outcomes: dict[str, Any]
    audit_entry: TombstoneRecord


def _find_affected_downstream(
    provenance: ProvenanceStore,
    poisoned_id: str,
    thread_id: str,
) -> list[str]:
    """BFS/DFS through provenance DAG to find all downstream checkpoints."""
    records = provenance.list_provenance_by_thread(thread_id)
    by_id = {r.checkpoint_id: r for r in records}
    affected: list[str] = []

    # Build parent -> children adjacency.
    children: dict[str, list[str]] = {}
    for r in records:
        for pid in r.parent_ids:
            children.setdefault(pid, []).append(r.checkpoint_id)

    # BFS from poisoned_id through children.
    affected = [poisoned_id]
    queue = deque(children.get(poisoned_id, []))
    visited = {poisoned_id}
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        affected.append(current)
        queue.extend(children.get(current, []))
    return affected


def tombstone_items_matching(
    provenance: ProvenanceStore,
    thread_id: str,
    predicate: Any,
    *,
    rerun_downstream: bool = True,
    actor: str = "test",
    rerun_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> TombstoneReport:
    """Find matching poisoned entries, mark them, and optionally re-run downstream.

    This is the hook D can call. A implements this; A does NOT implement D's registry.

    If `rerun_fn` is provided, it is called for each affected checkpoint id and
    must return a dict describing the rerun outcome. Otherwise a placeholder
    ``{"status": "needs_rerun"}`` is recorded.
    """
    records = provenance.list_provenance_by_thread(thread_id)
    matches = [r for r in records if predicate(r)]
    if not matches:
        raise ValueError("No provenance entries match the predicate.")

    poisoned = matches[0]
    poisoned_id = poisoned.checkpoint_id

    affected = _find_affected_downstream(provenance, poisoned_id, thread_id)

    audit = TombstoneRecord(
        checkpoint_id=poisoned_id,
        thread_id=thread_id,
        reason="matched predicate",
        actor=actor,
        ts="2026-07-06T00:00:00Z",
        affected_downstream=affected,
        rerun_outcomes={},
    )
    provenance.put_tombstone(audit)

    rerun_outcomes: dict[str, Any] = {}
    if rerun_downstream:
        for cid in affected:
            if rerun_fn is not None:
                rerun_outcomes[cid] = rerun_fn(cid, thread_id)
            else:
                rerun_outcomes[cid] = {"status": "needs_rerun", "poison_ancestor": poisoned_id}

    audit.rerun_outcomes = rerun_outcomes
    return TombstoneReport(
        tombstoned_id=poisoned_id,
        affected_downstream=affected,
        rerun_outcomes=rerun_outcomes,
        audit_entry=audit,
    )
