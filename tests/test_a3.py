"""A3 tombstone tests.

Runs in mock mode via config.ci.yaml. Real-LLM path is exercised by
`python -m agent_lifecycle_harness.run --demo A3` once `config.yaml` is filled.
"""

from __future__ import annotations

import pytest

from agent_lifecycle_harness.demos.a3_tombstone import DemoResult, demo_A3_tombstone


def test_a3_all_assertions_pass_mock():
    result: DemoResult = demo_A3_tombstone()
    assert result.name == "A3_tombstone"
    assert result.passed is True, _format_failures(result)
    names = {a.name for a in result.assertions}
    expected = {
        "dag_traversal_finds_affected",
        "rerun_poison_removed",
        "tombstone_soft",
        "audit_log_has_op_actor_ts",
    }
    assert names == expected, f"Missing or extra assertions: {names ^ expected}"
    for a in result.assertions:
        assert a.passed is True, a.evidence


def test_a3_metrics():
    result = demo_A3_tombstone()
    assert result.metrics["poison_turn"] == 3
    assert result.metrics["framework"] == "langgraph"


def _format_failures(result: DemoResult) -> str:
    lines = ["A3 demo failed:"]
    for a in result.assertions:
        if not a.passed:
            lines.append(f"  - {a.name}: {a.evidence}")
    return "\n".join(lines)
