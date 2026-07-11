"""A6 — State schema migration.

App-owned schema registry + migration functions. Backward-compatible
migration means old state can be read by new code without data loss
of fields not covered by the migration.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SchemaVersion:
    version: str
    schema: dict[str, Any]
    migration_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    backward_compatible: bool = True


class SchemaRegistry:
    """App-owned schema registry.

    LangGraph checkpoint entries are opaque; the app layer owns schema
    evolution and migrates state before handing it to the graph.
    """

    def __init__(self) -> None:
        self._versions: dict[str, SchemaVersion] = {}
        self._ordered: list[str] = []

    def register(self, version: str, schema: dict[str, Any], *, backward_compatible: bool = True, migration_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> None:
        if version in self._versions:
            raise ValueError(f"Schema version {version} already registered.")
        self._versions[version] = SchemaVersion(
            version=version,
            schema=schema,
            backward_compatible=backward_compatible,
            migration_fn=migration_fn,
        )
        self._ordered.append(version)

    def get(self, version: str) -> SchemaVersion | None:
        return self._versions.get(version)

    def latest(self) -> SchemaVersion | None:
        if not self._ordered:
            return None
        return self._versions[self._ordered[-1]]

    def migrate(self, state: dict[str, Any], from_version: str, to_version: str) -> dict[str, Any]:
        """Migrate state from from_version to to_version."""
        if from_version == to_version:
            return state
        source = self._versions.get(from_version)
        target = self._versions.get(to_version)
        if source is None or target is None:
            raise ValueError(f"Unknown schema version: {from_version} or {to_version}")
        if target.migration_fn is None:
            return state
        return target.migration_fn(copy.deepcopy(state))


class TransactionalMigrator:
    """DB-backed transactional migrator with on_record hook and ledger.

    Wraps SchemaRegistry.migrate() over a SQLite table. Each record is
    migrated inside a single transaction; on_record is called after each
    successful record migration. A ledger table tracks which records have
    already been migrated so re-runs can skip them (resumable).
    """

    def __init__(self, registry: SchemaRegistry, db_path: str, *, on_record: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.registry = registry
        self.db_path = db_path
        self.on_record = on_record
        self._ensure_ledger()

    def _ensure_ledger(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS migration_ledger (record_id TEXT PRIMARY KEY, migrated_at TEXT)"
            )
            conn.commit()
        finally:
            conn.close()

    def _already_migrated(self, conn: sqlite3.Connection, record_id: str) -> bool:
        row = conn.execute("SELECT 1 FROM migration_ledger WHERE record_id = ?", (record_id,)).fetchone()
        return row is not None

    def migrate_table(
        self,
        table: str,
        from_version: str,
        to_version: str,
        *,
        crash_after: int | None = None,
    ) -> int:
        """Migrate all rows in `table` from from_version to to_version.

        Each row is committed individually so a crash leaves previously
        committed rows intact and recorded in the ledger.
        """
        conn = sqlite3.connect(self.db_path)
        migrated = 0
        try:
            rows = conn.execute(
                f"SELECT id, state FROM {table} WHERE version = ?", (from_version,)
            ).fetchall()
            for row_id, state_json in rows:
                if self._already_migrated(conn, row_id):
                    continue
                state = json.loads(state_json)
                new_state = self.registry.migrate(state, from_version, to_version)
                conn.execute("BEGIN")
                try:
                    conn.execute(
                        f"UPDATE {table} SET state = ?, version = ? WHERE id = ?",
                        (json.dumps(new_state), to_version, row_id),
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO migration_ledger (record_id, migrated_at) VALUES (?, ?)",
                        (row_id, __import__("datetime").datetime.now().isoformat()),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                migrated += 1
                if self.on_record is not None:
                    self.on_record(row_id, new_state)
                if crash_after is not None and migrated >= crash_after:
                    raise RuntimeError(f"Simulated crash after {migrated} records")
        finally:
            conn.close()
        return migrated

    def reset_ledger(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM migration_ledger")
            conn.commit()
        finally:
            conn.close()
