"""A7 cross-framework matrix tests.

Runs in mock mode via config.ci.yaml. Real-LLM path is exercised by
`python -m agent_lifecycle_harness.run --demo A7` once `config.yaml` is filled.
"""

from __future__ import annotations

import pytest

from agent_lifecycle_harness.demos.a7_cross_framework import DemoResult, demo_A7_cross_framework


def test_a7_all_assertions_pass_mock():
    result: DemoResult = demo_A7_cross_framework()
    assert result.name == "A7_cross_framework"
    assert result.passed is True, _format_failures(result)
    names = {a.name for a in result.assertions}
    expected = {
        "matrix_complete",
        "oai_cells_run_or_cited",
        "app_layer_boundary_documented",
        # behavior cells (actually executed against the SDK)
        "oai_session_isolation",
        "oai_pop_item_removes_provenance",
        "oai_usage_accumulates",
        "oai_turn_span_exports",
        # doc cells (citation-only, not executed)
        "oai_compaction_framework_given",
        "oai_run_config_hot_reload",
        "oai_schema_migration_app_owned",
    }
    assert names == expected, f"Missing or extra assertions: {names ^ expected}"
    # Behavior cells + structural checks must pass.
    for a in result.assertions:
        if not a.evidence.startswith("[documented"):
            assert a.passed is True, a.evidence
    # Doc cells must carry a citation and be marked not-executed.
    doc_rows = [a for a in result.assertions if a.evidence.startswith("[documented")]
    assert len(doc_rows) == 3, f"expected 3 doc cells, got {len(doc_rows)}"
    for d in doc_rows:
        assert "site-packages" in d.evidence, f"doc cell missing verifiable citation: {d.evidence}"


def _format_failures(result: DemoResult) -> str:
    lines = ["A7 demo failed:"]
    for a in result.assertions:
        if not a.passed:
            lines.append(f"  - {a.name}: {a.evidence}")
    return "\n".join(lines)
