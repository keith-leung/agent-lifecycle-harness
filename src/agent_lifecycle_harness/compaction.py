"""A2 — Checkpoint retention & compaction.

Policy: first-N + last-N raw checkpoints preserved; middle compressed into
ONE model-generated digest entry that embeds the replaced raw ids.

The digest is app-owned storage keyed by thread_id + checkpoint range.
Compactor is idempotent: re-running on an already-compacted segment is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agent_lifecycle_harness.llm import LLMClient, LLMResponse


@dataclass
class DigestEntry:
    digest_id: str
    thread_id: str
    summary: str
    replaced_raw_ids: list[str]
    lossy_fields: list[str]
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CompactionStore:
    """App-owned compaction store backed by a dedicated SQLite table.

    This is separate from LangGraph's checkpoint table because LG does not
    provide compaction primitives. The store maps checkpoint ranges to digest
    entries so A3's DAG traversal can descend through them.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS compaction_digests (
                digest_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                replaced_raw_ids TEXT NOT NULL,
                lossy_fields TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add_digest(self, entry: DigestEntry) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO compaction_digests VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry.digest_id,
                entry.thread_id,
                entry.summary,
                json.dumps(entry.replaced_raw_ids),
                json.dumps(entry.lossy_fields),
                entry.created_at,
                json.dumps(entry.metadata),
            ),
        )
        self._conn.commit()

    def digests_for_thread(self, thread_id: str) -> list[DigestEntry]:
        rows = self._conn.execute(
            "SELECT * FROM compaction_digests WHERE thread_id = ? ORDER BY created_at",
            (thread_id,),
        ).fetchall()
        out: list[DigestEntry] = []
        for row in rows:
            out.append(
                DigestEntry(
                    digest_id=row[0],
                    thread_id=row[1],
                    summary=row[2],
                    replaced_raw_ids=json.loads(row[3]),
                    lossy_fields=json.loads(row[4]),
                    created_at=row[5],
                    metadata=json.loads(row[6]),
                )
            )
        return out

    def is_range_compacted(self, thread_id: str, raw_ids: Sequence[str]) -> bool:
        """Return True if all given raw_ids are already covered by some digest."""
        digests = self.digests_for_thread(thread_id)
        covered: set[str] = set()
        for d in digests:
            covered.update(d.replaced_raw_ids)
        return all(rid in covered for rid in raw_ids)


class CheckpointCompactor:
    """App-owned compaction policy for LangGraph checkpoint histories.

    Locked policy: first-N + last-N raw checkpoints preserved; middle
    compressed into ONE model-generated digest entry.
    """

    def __init__(self, store: CompactionStore, judge: LLMClient, *, first_last_n: int = 3) -> None:
        self.store = store
        self.judge = judge
        self.first_last_n = first_last_n

    async def acompact(self, thread_id: str, raw_checkpoints: list[dict[str, Any]]) -> DigestEntry | None:
        """Compact the middle of raw_checkpoints into a single digest.

        Returns None if the range is already compacted (idempotent no-op).
        """
        if len(raw_checkpoints) <= 2 * self.first_last_n:
            return None

        # Candidate raw ids for the middle segment.
        middle = raw_checkpoints[self.first_last_n : -self.first_last_n]
        middle_ids = [c.get("checkpoint_id") for c in middle if c.get("checkpoint_id")]
        if not middle_ids:
            return None

        if self.store.is_range_compacted(thread_id, middle_ids):
            return None

        # Build a compact textual summary from the middle checkpoints.
        # In a real-LLM run this is model-generated; in CI-MOCK we still
        # produce a deterministic digest string.
        combined = "\n".join(
            f"[{c.get('checkpoint_id')}] values={_truncate_values_for_prompt(c.get('values', {}))}"
            for c in middle
        )
        prompt = (
            "Summarize the following checkpoint segment into a single concise digest.\n"
            "Preserve enough detail to keep replay coherent.\n"
            "Return ONLY the digest text.\n\n" + combined
        )
        response = await self.judge.ainvoke([{"role": "user", "content": prompt}])
        summary = response.content.strip()

        # Lossy fields: anything not present in the digest summary.
        # We conservatively list all keys from the middle values as potentially lossy.
        all_keys: set[str] = set()
        for c in middle:
            all_keys.update(c.get("values", {}).keys())
        lossy_fields = sorted(all_keys)

        digest_id = _make_digest_id(thread_id, middle_ids)
        entry = DigestEntry(
            digest_id=digest_id,
            thread_id=thread_id,
            summary=summary,
            replaced_raw_ids=middle_ids,
            lossy_fields=lossy_fields,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"first_last_n": self.first_last_n, "compacted_count": len(middle_ids)},
        )
        self.store.add_digest(entry)
        return entry

    def compact(self, thread_id: str, raw_checkpoints: list[dict[str, Any]]) -> DigestEntry | None:
        """Sync wrapper around acompact."""
        import asyncio
        return asyncio.run(self.acompact(thread_id, raw_checkpoints))


def _make_digest_id(thread_id: str, raw_ids: Sequence[str]) -> str:
    payload = thread_id + "|" + "|".join(raw_ids)
    return "digest-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _truncate_values_for_prompt(values: dict[str, Any], max_chars: int = 4000) -> dict[str, Any]:
    """Truncate checkpoint values to keep prompt size manageable.
    
    Keeps only the last 10 messages and truncates long string fields.
    """
    if not values:
        return values
    truncated = dict(values)
    messages = truncated.get("messages", [])
    if isinstance(messages, list) and len(messages) > 10:
        truncated["messages"] = messages[-10:]
    for key, val in truncated.items():
        if isinstance(val, str) and len(val) > max_chars:
            truncated[key] = val[:max_chars] + "...[truncated]"
    return truncated


def coerce_checkpoint_snapshot(snap: Any) -> dict[str, Any]:
    """Normalize a StateSnapshot or dict into a plain checkpoint dict."""
    if isinstance(snap, dict):
        return snap
    return {
        "checkpoint_id": getattr(snap, "config", {})
        .get("configurable", {})
        .get("checkpoint_id"),
        "next": getattr(snap, "next", []),
        "values": getattr(snap, "values", {}),
    }
