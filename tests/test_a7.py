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
        "oai_session_isolation",
        "oai_compaction_framework_given",
        "oai_pop_item_removes_provenance",
        "oai_run_config_hot_reload",
        "oai_run_metrics",
        "oai_schema_versioning",
    }
    assert names == expected, f"Missing or extra assertions: {names ^ expected}"
    for a in result.assertions:
        assert a.passed is True, a.evidence


def _format_failures(result: DemoResult) -> str:
    lines = ["A7 demo failed:"]
    for a in result.assertions:
        if not a.passed:
            lines.append(f"  - {a.name}: {a.evidence}")
    return "\n".join(lines)
