"""Provenance + tombstone tracking using LangGraph BaseStore.

App-owned indices because LangGraph treats checkpoint entries as opaque
blobs. The provenance DAG lives here, not in LG's checkpoint table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence


@dataclass
class ProvenanceRecord:
    checkpoint_id: str
    thread_id: str
    parent_ids: list[str]
    produced_by: str
    produced_at: str
    sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TombstoneRecord:
    checkpoint_id: str
    thread_id: str
    reason: str
    actor: str
    ts: str
    affected_downstream: list[str] = field(default_factory=list)
    rerun_outcomes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProvenanceStore:
    """App-owned provenance + tombstone store backed by BaseStore.

    In CI-MOCK mode we fall back to an in-memory dict so tests run without
    a running LangGraph graph instance.
    """

    def __init__(self, base_store: Any | None = None) -> None:
        self._store = base_store
        self._mock_mode = base_store is None
        self._memory_provenance: dict[str, ProvenanceRecord] = {}
        self._memory_tombstones: dict[str, TombstoneRecord] = {}

    def _namespace(self, thread_id: str) -> str:
        return f"provenance/{thread_id}"

    def put_provenance(self, record: ProvenanceRecord) -> None:
        if self._mock_mode:
            self._memory_provenance[record.checkpoint_id] = record
            return
        ns = self._namespace(record.thread_id)
        self._store.put(
            ns,
            record.checkpoint_id,
            {
                "type": "provenance",
                "checkpoint_id": record.checkpoint_id,
                "thread_id": record.thread_id,
                "parent_ids": json.dumps(record.parent_ids),
                "produced_by": record.produced_by,
                "produced_at": record.produced_at,
                "sha256": record.sha256,
                "metadata": json.dumps(record.metadata),
            },
        )

    def get_provenance(self, checkpoint_id: str) -> ProvenanceRecord | None:
        if self._mock_mode:
            return self._memory_provenance.get(checkpoint_id)
        # Search across all provenance namespaces for this checkpoint_id.
        # BaseStore.search is namespace-scoped; without a thread_id we cannot
        # target one namespace, so this requires a linear scan of known threads.
        # In practice callers always have thread_id and use list_provenance_by_thread.
        # Fall back to scanning the single namespace if only one thread is active.
        raise NotImplementedError(
            "get_provenance without thread_id is not supported on BaseStore; "
            "use list_provenance_by_thread(thread_id) and filter."
        )

    def list_provenance_by_thread(self, thread_id: str) -> list[ProvenanceRecord]:
        if self._mock_mode:
            return [r for r in self._memory_provenance.values() if r.thread_id == thread_id]
        ns = self._namespace(thread_id)
        items = self._store.search(ns)
        out: list[ProvenanceRecord] = []
        for item in items:
            v = item.value
            out.append(ProvenanceRecord(
                checkpoint_id=v.get("checkpoint_id", item.key),
                thread_id=v.get("thread_id", thread_id),
                parent_ids=json.loads(v.get("parent_ids", "[]")),
                produced_by=v.get("produced_by", ""),
                produced_at=v.get("produced_at", ""),
                sha256=v.get("sha256", ""),
                metadata=json.loads(v.get("metadata", "{}")) if v.get("metadata") else {},
            ))
        return out

    def put_tombstone(self, record: TombstoneRecord) -> None:
        if self._mock_mode:
            self._memory_tombstones[record.checkpoint_id] = record
            return
        ns = self._namespace(record.thread_id)
        self._store.put(
            ns,
            record.checkpoint_id,
            {
                "type": "tombstone",
                "checkpoint_id": record.checkpoint_id,
                "thread_id": record.thread_id,
                "reason": record.reason,
                "actor": record.actor,
                "ts": record.ts,
                "affected_downstream": json.dumps(record.affected_downstream),
                "rerun_outcomes": json.dumps(record.rerun_outcomes),
                "metadata": json.dumps(record.metadata),
            },
        )

    def get_tombstone(self, checkpoint_id: str) -> TombstoneRecord | None:
        if self._mock_mode:
            return self._memory_tombstones.get(checkpoint_id)
        # Tombstones are namespaced by thread; without thread_id we scan.
        raise NotImplementedError(
            "get_tombstone without thread_id is not supported on BaseStore; "
            "use list_tombstones_by_thread(thread_id) and filter."
        )

    def list_tombstones_by_thread(self, thread_id: str) -> list[TombstoneRecord]:
        if self._mock_mode:
            return [r for r in self._memory_tombstones.values() if r.thread_id == thread_id]
        ns = self._namespace(thread_id)
        items = self._store.search(ns)
        out: list[TombstoneRecord] = []
        for item in items:
            v = item.value
            if v.get("type") != "tombstone":
                continue
            out.append(TombstoneRecord(
                checkpoint_id=v.get("checkpoint_id", item.key),
                thread_id=v.get("thread_id", thread_id),
                reason=v.get("reason", ""),
                actor=v.get("actor", ""),
                ts=v.get("ts", ""),
                affected_downstream=json.loads(v.get("affected_downstream", "[]")),
                rerun_outcomes=json.loads(v.get("rerun_outcomes", "{}")) if v.get("rerun_outcomes") else {},
                metadata=json.loads(v.get("metadata", "{}")) if v.get("metadata") else {},
            ))
        return out

    def is_tombstoned(self, checkpoint_id: str) -> bool:
        return self.get_tombstone(checkpoint_id) is not None


def build_provenance_record(
    checkpoint_id: str,
    thread_id: str,
    parent_ids: Sequence[str],
    produced_by: str,
    raw_content: str,
) -> ProvenanceRecord:
    sha256 = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()[:16]
    return ProvenanceRecord(
        checkpoint_id=checkpoint_id,
        thread_id=thread_id,
        parent_ids=list(parent_ids),
        produced_by=produced_by,
        produced_at=datetime.now(timezone.utc).isoformat(),
        sha256=sha256,
    )
