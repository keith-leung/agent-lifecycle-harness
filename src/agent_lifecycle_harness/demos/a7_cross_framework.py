"""A7 — Cross-framework lifecycle matrix.

Maps A1-A6 lifecycle boundaries to OpenAI Agents SDK equivalents.
Each OAI-side cell either runs with assertion or is documented as
'framework-given, here's the API used' with a citation in code.

SPEC §5 A7: matrix complete + each OAI-side cell either runs (with assertion)
or is documented 'framework-given, here's the API used' with a citation in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_lifecycle_harness.openai_agents_harness import (
    BehaviorCell,
    DocCell,
    assert_oai_pop_item_removes_provenance,
    assert_oai_session_isolation,
    assert_oai_turn_span_exports,
    assert_oai_usage_accumulates,
    doc_oai_compaction_framework_given,
    doc_oai_run_config_hot_reload,
    doc_oai_schema_migration_app_owned,
)


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


# Mapping of LangGraph concepts to OpenAI Agents SDK equivalents.
# Each OAI-side cell is one of:
#   * behavior cell (assert_* function, runs against installed SDK, mutation-verified)
#   * doc cell (doc_* function, citation-only, NOT executed locally)
# The previous "import X; assert X is not None" middle ground has been removed.
CROSS_FRAMEWORK_MATRIX: dict[str, dict[str, Any]] = {
    "A1": {
        "langgraph_concept": "thread_id isolation + per-thread write lock",
        "oai_sdk_equivalent": "SQLiteSession (per-session row scoping via session_id)",
        "app_layer_boundary": "namespaced ids + write serialization",
        "framework_owned": "thread_id (LG) / Session (OAI)",
        "app_owned": "namespacing + locks",
        "oai_cell_kind": "behavior",
        "oai_cell": "assert_oai_session_isolation",
        "oai_citation": "site-packages/agents/memory/sqlite_session.py:17 (SQLiteSession.add_items, per-session row scoping)",
    },
    "A2": {
        "langgraph_concept": "SqliteSaver checkpoint history + app-owned CompactionStore",
        "oai_sdk_equivalent": "OpenAIResponsesCompactionSession (framework-given)",
        "app_layer_boundary": "first-N + last-N + middle-digest",
        "framework_owned": "checkpoint table (LG) / OpenAIResponsesCompactionSession (OAI)",
        "app_owned": "compaction policy + digest (LG only)",
        "oai_cell_kind": "documented",
        "oai_cell": "doc_oai_compaction_framework_given",
        "oai_citation": "site-packages/agents/memory/openai_responses_compaction_session.py:78 (run_compaction needs real client)",
    },
    "A3": {
        "langgraph_concept": "BaseStore provenance + soft tombstone + DAG traversal",
        "oai_sdk_equivalent": "Session.pop_item (DELETE...RETURNING, no tombstone)",
        "app_layer_boundary": "provenance DAG + audit log",
        "framework_owned": "opaque run records",
        "app_owned": "provenance + tombstone + rerun (both LG and OAI)",
        "oai_cell_kind": "behavior",
        "oai_cell": "assert_oai_pop_item_removes_provenance",
        "oai_citation": "site-packages/agents/memory/sqlite_session.py (SQLiteSession.pop_item, DELETE...RETURNING)",
    },
    "A4": {
        "langgraph_concept": "config-version tracker per thread_id",
        "oai_sdk_equivalent": "RunConfig (request-scoped config; no version/session binding)",
        "app_layer_boundary": "version tracker + session registration",
        "framework_owned": "thread_id / Session",
        "app_owned": "config version mapping (BOTH LG and OAI)",
        "oai_cell_kind": "documented",
        "oai_cell": "doc_oai_run_config_hot_reload",
        "oai_citation": "site-packages/agents/run_config.py:211 (RunConfig has no version field)",
    },
    "A5": {
        "langgraph_concept": "invoke instrumentation + threshold alerting",
        "oai_sdk_equivalent": "Usage.add + TurnSpanData.export (framework-given primitives)",
        "app_layer_boundary": "DegradationMonitor",
        "framework_owned": "execution trace primitives (Usage, TurnSpanData)",
        "app_owned": "sustained-delta detection + alerts (BOTH LG and OAI)",
        "oai_cell_kind": "behavior",
        "oai_cell": "assert_oai_usage_accumulates + assert_oai_turn_span_exports",
        "oai_citation": "site-packages/agents/usage.py:102 (Usage.add), agents/tracing/span_data.py:98 (TurnSpanData.export)",
    },
    "A6": {
        "langgraph_concept": "state schema registry + migration fn",
        "oai_sdk_equivalent": "none (app-owned migration)",
        "app_layer_boundary": "SchemaRegistry",
        "framework_owned": "RunConfig.trace_metadata slot (arbitrary user dict)",
        "app_owned": "schema registry + migration fn (BOTH LG and OAI)",
        "oai_cell_kind": "documented",
        "oai_cell": "doc_oai_schema_migration_app_owned",
        "oai_citation": "site-packages/agents/run_config.py:211 (trace_metadata is arbitrary dict, not a versioning API)",
    },
}


def assert_matrix_complete() -> AssertionResult:
    """Matrix covers all A1-A6 lifecycle boundaries."""
    expected_keys = {f"A{i}" for i in range(1, 7)}
    actual_keys = set(CROSS_FRAMEWORK_MATRIX.keys())
    if expected_keys != actual_keys:
        return AssertionResult(
            name="matrix_complete",
            passed=False,
            evidence=f"Missing or extra entries: {expected_keys ^ actual_keys}",
        )
    return AssertionResult(
        name="matrix_complete",
        passed=True,
        evidence="Matrix covers A1-A6.",
    )


def assert_oai_cells_run_or_cited() -> AssertionResult:
    """Each OAI-side cell is classified behavior or documented (no third kind).

    The previous "import-only" cell type is forbidden: every row must
    declare ``oai_cell_kind`` as either ``behavior`` (runs an observable
    assertion, mutation-verified) or ``documented`` (citation-only, not
    executed locally).
    """
    for key, row in CROSS_FRAMEWORK_MATRIX.items():
        kind = row.get("oai_cell_kind")
        citation = row.get("oai_citation")
        cell = row.get("oai_cell")
        if kind not in ("behavior", "documented"):
            return AssertionResult(
                name="oai_cells_run_or_cited",
                passed=False,
                evidence=f"Row {key} has invalid oai_cell_kind={kind!r}; must be behavior|documented.",
            )
        if not cell or not citation:
            return AssertionResult(
                name="oai_cells_run_or_cited",
                passed=False,
                evidence=f"Row {key} missing cell function or citation.",
            )
    return AssertionResult(
        name="oai_cells_run_or_cited",
        passed=True,
        evidence="All OAI cells classified as behavior or documented; none is import-only.",
    )


def assert_app_layer_boundary_documented() -> AssertionResult:
    """Each matrix row documents the app-layer boundary."""
    for key, row in CROSS_FRAMEWORK_MATRIX.items():
        if not row.get("app_layer_boundary"):
            return AssertionResult(
                name="app_layer_boundary_documented",
                passed=False,
                evidence=f"Row {key} missing app_layer_boundary.",
            )
    return AssertionResult(
        name="app_layer_boundary_documented",
        passed=True,
        evidence="All rows document app-layer boundary.",
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------

def demo_A7_cross_framework(harness: Any = None) -> DemoResult:
    """Run the A7 cross-framework lifecycle matrix scenario.

    Behavior cells run against the installed OAI SDK and must pass.
    Doc cells are citation-only (NOT executed) and do not count as passes.
    """
    assertions: list[AssertionResult] = []
    behavior_cells: list[BehaviorCell] = []
    doc_cells: list[DocCell] = []

    # Matrix structure checks
    assertions.append(assert_matrix_complete())
    assertions.append(assert_oai_cells_run_or_cited())
    assertions.append(assert_app_layer_boundary_documented())

    # Behavior cells — actually run, must pass.
    behavior_cells.append(assert_oai_session_isolation())
    behavior_cells.append(assert_oai_pop_item_removes_provenance())
    behavior_cells.append(assert_oai_usage_accumulates())
    behavior_cells.append(assert_oai_turn_span_exports())

    # Doc cells — citation-only. Not executed, do NOT count as passes.
    doc_cells.append(doc_oai_compaction_framework_given())
    doc_cells.append(doc_oai_run_config_hot_reload())
    doc_cells.append(doc_oai_schema_migration_app_owned())

    # Surface behavior cells as assertion rows (these drive overall PASS).
    for b in behavior_cells:
        assertions.append(AssertionResult(
            name=b.name, passed=b.passed, evidence=f"[behavior] {b.evidence}",
        ))
    # Surface doc cells as informational rows. These are NOT passing tests —
    # we set passed=True so the demo's overall PASS (driven by behavior cells
    # + structural checks) is not held hostage by documentation rows, but
    # the evidence explicitly marks them as not-executed citations.
    for d in doc_cells:
        assertions.append(AssertionResult(
            name=d.name, passed=True,
            evidence=f"[documented, not executed] {d.note} | citation: {d.citation}",
        ))

    # Overall PASS = behavior cells all passed + structural checks passed.
    # Doc cells are informational; they don't gate PASS because they're not tests.
    behavior_passed = all(b.passed for b in behavior_cells)
    structural_passed = all(a.passed for a in assertions[:3])  # matrix_complete / cells_run_or_cited / boundary
    passed = behavior_passed and structural_passed

    metrics = {
        "matrix_entries": len(CROSS_FRAMEWORK_MATRIX),
        "behavior_cells": len(behavior_cells),
        "behavior_cells_passed": sum(1 for b in behavior_cells if b.passed),
        "doc_cells": len(doc_cells),
        "doc_cells_executed": sum(1 for d in doc_cells if d.executed),
        "mutation_verified_cells": [
            "oai_session_isolation (shared_db mutation)",
            "oai_pop_item_removes_provenance (broken_pop mutation)",
            "oai_usage_accumulates (broken_add mutation)",
            "oai_turn_span_exports (broken_export mutation)",
        ],
    }
    return DemoResult(name="A7_cross_framework", passed=passed, assertions=assertions, metrics=metrics)
