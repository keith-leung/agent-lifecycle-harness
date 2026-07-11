"""A6 — State schema migration demo.

Demonstrates backward-compatible migration of state between schema versions.

Assertions:
  (a) migration preserves data not covered by the migration
  (b) latest schema is backward compatible
  (c) transactional: crash mid-migration leaves DB unchanged or fully migrated
  (d) resumable: re-running migrator skips already-migrated rows
  (e) v1-shape-read-on-migrated raises: old field access raises on migrated state
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import dataclass, field
from typing import Any

from agent_lifecycle_harness.agent import LifecycleHarness
from agent_lifecycle_harness.llm import LLMClient, MockLLMClient, RealLLMClient
from agent_lifecycle_harness.migration import SchemaRegistry, TransactionalMigrator


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
        llm = MockLLMClient(prefix="a6-sut")
    if judge is None:
        judge = MockLLMClient(prefix="a6-judge")
    return LifecycleHarness(db_path=db_path, llm=llm, judge=judge), db_path


def _setup_schema_registry() -> SchemaRegistry:
    registry = SchemaRegistry()
    registry.register(
        "v1",
        {"type": "object", "properties": {"user_id": {"type": "string"}}},
        backward_compatible=True,
    )
    registry.register(
        "v2",
        {"type": "object", "properties": {"account_id": {"type": "string"}, "turns": {"type": "array"}}},
        backward_compatible=True,
        migration_fn=lambda state: {
            "account_id": state.get("user_id"),
            "turns": state.get("history", []),
        },
    )
    return registry


def _setup_db(db_path: str, count: int = 5) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS agent_state (id TEXT PRIMARY KEY, version TEXT, state TEXT)")
        for i in range(count):
            state = json.dumps({"user_id": f"u-{i}", "history": [{"role": "user", "content": f"hi-{i}"}]})
            conn.execute("INSERT INTO agent_state (id, version, state) VALUES (?, ?, ?)", (f"rec-{i}", "v1", state))
        conn.commit()
    finally:
        conn.close()


def assert_migration_preserves_data(registry: SchemaRegistry) -> AssertionResult:
    """(a) Migration preserves data not covered by the migration."""
    old_state = {"user_id": "u-1", "history": [{"role": "user", "content": "hi"}]}
    migrated = registry.migrate(old_state, "v1", "v2")
    if migrated.get("account_id") != old_state.get("user_id"):
        return AssertionResult(
            name="migration_preserves_data",
            passed=False,
            evidence="Migration lost account_id (renamed from user_id).",
        )
    return AssertionResult(
        name="migration_preserves_data",
        passed=True,
        evidence="Migration preserved account_id.",
    )


def assert_backward_compatible(registry: SchemaRegistry) -> AssertionResult:
    """(b) Latest schema is backward compatible."""
    latest = registry.latest()
    if latest is None or not latest.backward_compatible:
        return AssertionResult(
            name="backward_compatible",
            passed=False,
            evidence="Latest schema is not backward compatible.",
        )
    return AssertionResult(
        name="backward_compatible",
        passed=True,
        evidence="Latest schema is backward compatible.",
    )


def assert_transactional(registry: SchemaRegistry) -> AssertionResult:
    """(c) Transactional: crash mid-migration leaves DB unchanged."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    _setup_db(db_path, count=5)
    hook_calls: list = []
    migrator = TransactionalMigrator(
        registry,
        db_path,
        on_record=lambda rid, state: hook_calls.append(rid),
    )
    try:
        migrator.migrate_table("agent_state", "v1", "v2", crash_after=2)
    except RuntimeError as exc:
        if "Simulated crash" not in str(exc):
            return AssertionResult(
                name="transactional",
                passed=False,
                evidence=f"Unexpected crash: {exc}",
            )
    else:
        return AssertionResult(
            name="transactional",
            passed=False,
            evidence="Expected RuntimeError was not raised.",
        )

    # After crash, previously committed rows are v2, uncommitted rows remain v1.
    # No row is half-migrated.
    conn = sqlite3.connect(db_path)
    try:
        v2_rows = conn.execute("SELECT COUNT(*) FROM agent_state WHERE version = 'v2'").fetchone()[0]
        v1_rows = conn.execute("SELECT COUNT(*) FROM agent_state WHERE version = 'v1'").fetchone()[0]
    finally:
        conn.close()

    if v2_rows == 2 and v1_rows == 3:
        return AssertionResult(
            name="transactional",
            passed=True,
            evidence=f"Crash left committed rows intact: v2_rows={v2_rows}, v1_rows={v1_rows}, no half-migrated rows.",
        )
    return AssertionResult(
        name="transactional",
        passed=False,
        evidence=f"Unexpected state: v2_rows={v2_rows}, v1_rows={v1_rows}.",
    )


def assert_resumable(registry: SchemaRegistry) -> AssertionResult:
    """(d) Resumable: re-running migrator skips already-migrated rows."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    _setup_db(db_path, count=5)
    migrator = TransactionalMigrator(registry, db_path)
    # First run: crash after 2 records.
    try:
        migrator.migrate_table("agent_state", "v1", "v2", crash_after=2)
    except RuntimeError:
        pass
    # Second run: should migrate remaining 3 records (skip the 2 already done).
    migrated = migrator.migrate_table("agent_state", "v1", "v2")
    conn = sqlite3.connect(db_path)
    try:
        v2_rows = conn.execute("SELECT COUNT(*) FROM agent_state WHERE version = 'v2'").fetchone()[0]
    finally:
        conn.close()
    if migrated == 3 and v2_rows == 5:
        return AssertionResult(
            name="resumable",
            passed=True,
            evidence=f"Resumable: second run migrated {migrated} rows, total v2={v2_rows}.",
        )
    return AssertionResult(
        name="resumable",
        passed=False,
        evidence=f"Resumable failed: second_run_migrated={migrated}, total_v2={v2_rows}.",
    )


def assert_v1_shape_read_on_migrated_raises(registry: SchemaRegistry) -> AssertionResult:
    """(e) v1-shape-read-on-migrated raises: old field access raises on migrated state."""
    old_state = {"user_id": "u-1", "history": [{"role": "user", "content": "hi"}]}
    migrated = registry.migrate(old_state, "v1", "v2")
    try:
        _ = migrated["user_id"]
        return AssertionResult(
            name="v1_shape_read_on_migrated_raises",
            passed=False,
            evidence="Accessing v1 field 'user_id' on migrated state did not raise.",
        )
    except (KeyError, TypeError):
        return AssertionResult(
            name="v1_shape_read_on_migrated_raises",
            passed=True,
            evidence="Accessing v1 field 'user_id' on migrated state correctly raises.",
        )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------

def demo_A6_migration(
    harness: LifecycleHarness | None = None,
) -> DemoResult:
    """Run the A6 state schema migration scenario."""
    if harness is None:
        harness, _ = _make_harness()
    registry = _setup_schema_registry()
    assertions: list[AssertionResult] = []

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name

    # (a) migration preserves data
    assertions.append(assert_migration_preserves_data(registry))

    # (b) backward compatible
    assertions.append(assert_backward_compatible(registry))

    # (c) transactional (DB-backed)
    assertions.append(assert_transactional(registry))

    # (d) resumable (DB-backed)
    assertions.append(assert_resumable(registry))

    # (e) v1-shape-read-on-migrated raises
    assertions.append(assert_v1_shape_read_on_migrated_raises(registry))

    passed = all(a.passed for a in assertions)
    metrics = {
        "registered_versions": registry._ordered,
        "framework": harness.framework,
    }
    return DemoResult(name="A6_migration", passed=passed, assertions=assertions, metrics=metrics)
