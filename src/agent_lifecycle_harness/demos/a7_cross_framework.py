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
    assert_oai_compaction_framework_given,
    assert_oai_pop_item_removes_provenance,
    assert_oai_run_config_hot_reload,
    assert_oai_run_metrics,
    assert_oai_schema_migration_app_owned,
    assert_oai_session_isolation,
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
CROSS_FRAMEWORK_MATRIX: dict[str, dict[str, Any]] = {
    "A1": {
        "langgraph_concept": "thread_id isolation + per-thread write lock",
        "oai_sdk_equivalent": "Session + RunConfig (request-scoped reload)",
        "app_layer_boundary": "namespaced ids + write serialization",
        "framework_owned": "thread_id (LG) / Session (OAI)",
        "app_owned": "namespacing + locks",
        "oai_implementation": "assert_oai_session_isolation",
        "oai_citation": "openai.agents.Session (OAI SDK docs)",
    },
    "A2": {
        "langgraph_concept": "SqliteSaver checkpoint history + app-owned CompactionStore",
        "oai_sdk_equivalent": "OpenAIResponsesCompactionSession (framework-given)",
        "app_layer_boundary": "first-N + last-N + middle-digest",
        "framework_owned": "checkpoint table (LG) / OpenAIResponsesCompactionSession (OAI)",
        "app_owned": "compaction policy + digest (LG only)",
        "oai_implementation": "assert_oai_compaction_framework_given",
        "oai_citation": "openai.agents.OpenAIResponsesCompactionSession (OAI SDK docs)",
    },
    "A3": {
        "langgraph_concept": "BaseStore provenance + soft tombstone + DAG traversal",
        "oai_sdk_equivalent": "Session.pop_item (removes provenance, no tombstone)",
        "app_layer_boundary": "provenance DAG + audit log",
        "framework_owned": "opaque run records",
        "app_owned": "provenance + tombstone + rerun (both LG and OAI)",
        "oai_implementation": "assert_oai_pop_item_removes_provenance",
        "oai_citation": "openai.agents.Session.pop_item (OAI SDK docs)",
    },
    "A4": {
        "langgraph_concept": "config-version tracker per thread_id",
        "oai_sdk_equivalent": "RunConfig (request-scoped, framework-given)",
        "app_layer_boundary": "version tracker + session registration",
        "framework_owned": "thread_id / Session",
        "app_owned": "config version mapping (LG only)",
        "oai_implementation": "assert_oai_run_config_hot_reload",
        "oai_citation": "openai.agents.RunConfig (OAI SDK docs)",
    },
    "A5": {
        "langgraph_concept": "invoke instrumentation + threshold alerting",
        "oai_sdk_equivalent": "run metrics + alerting hooks (framework-given)",
        "app_layer_boundary": "DegradationMonitor",
        "framework_owned": "execution trace",
        "app_owned": "metrics + alerts (both LG and OAI)",
        "oai_implementation": "assert_oai_run_metrics",
        "oai_citation": "openai.agents.Run metrics (OAI SDK docs)",
    },
    "A6": {
        "langgraph_concept": "state schema registry + migration fn",
        "oai_sdk_equivalent": "none (app-owned migration; trace_metadata is arbitrary metadata, not a versioning API)",
        "app_layer_boundary": "SchemaRegistry",
        "framework_owned": "RunConfig.trace_metadata slot exists (arbitrary user dict)",
        "app_owned": "schema registry + migration fn (BOTH LG and OAI)",
        "oai_implementation": "assert_oai_schema_migration_app_owned",
        "oai_citation": "agents.RunConfig.trace_metadata (arbitrary metadata, NOT a versioning API)",
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
    """Each OAI-side cell either runs with assertion or is documented with citation."""
    for key, row in CROSS_FRAMEWORK_MATRIX.items():
        impl = row.get("oai_implementation")
        citation = row.get("oai_citation")
        if not impl and not citation:
            return AssertionResult(
                name="oai_cells_run_or_cited",
                passed=False,
                evidence=f"Row {key} missing both implementation and citation.",
            )
    return AssertionResult(
        name="oai_cells_run_or_cited",
        passed=True,
        evidence="All OAI cells have implementation or citation.",
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

    Runs OAI SDK implementations for each cell where available.
    """
    assertions: list[AssertionResult] = []

    # Matrix structure check
    assertions.append(assert_matrix_complete())
    assertions.append(assert_oai_cells_run_or_cited())
    assertions.append(assert_app_layer_boundary_documented())

    # Run OAI SDK implementations
    assertions.append(assert_oai_session_isolation())
    assertions.append(assert_oai_compaction_framework_given())
    assertions.append(assert_oai_pop_item_removes_provenance())
    assertions.append(assert_oai_run_config_hot_reload())
    assertions.append(assert_oai_run_metrics())
    assertions.append(assert_oai_schema_migration_app_owned())

    passed = all(a.passed for a in assertions)
    metrics = {
        "matrix_entries": len(CROSS_FRAMEWORK_MATRIX),
        "framework": "cross-framework",
    }
    return DemoResult(name="A7_cross_framework", passed=passed, assertions=assertions, metrics=metrics)
