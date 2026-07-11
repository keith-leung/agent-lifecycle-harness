"""A6 migration tests.

Runs in mock mode via config.ci.yaml. Real-LLM path is exercised by
`python -m agent_lifecycle_harness.run --demo A6` once `config.yaml` is filled.
"""

from __future__ import annotations

import pytest

from agent_lifecycle_harness.demos.a6_migration import DemoResult, demo_A6_migration


def test_a6_all_assertions_pass_mock():
    result: DemoResult = demo_A6_migration()
    assert result.name == "A6_migration"
    assert result.passed is True, _format_failures(result)
    names = {a.name for a in result.assertions}
    expected = {
        "migration_preserves_data",
        "backward_compatible",
        "transactional",
        "resumable",
        "v1_shape_read_on_migrated_raises",
    }
    assert names == expected, f"Missing or extra assertions: {names ^ expected}"
    for a in result.assertions:
        assert a.passed is True, a.evidence


def test_a6_metrics():
    result = demo_A6_migration()
    assert result.metrics["registered_versions"] == ["v1", "v2"]
    assert result.metrics["framework"] == "langgraph"


def _format_failures(result: DemoResult) -> str:
    lines = ["A6 demo failed:"]
    for a in result.assertions:
        if not a.passed:
            lines.append(f"  - {a.name}: {a.evidence}")
    return "\n".join(lines)
