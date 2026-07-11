"""A2∩A3 interop tests.

Runs in mock mode via config.ci.yaml. Real-LLM path is exercised by
`python -m agent_lifecycle_harness.run --demo A2_A3` once `config.yaml` is filled.
"""

from __future__ import annotations

import pytest

from agent_lifecycle_harness.demos.a2a3_interop import DemoResult, demo_A2_A3_interop


def test_a2a3_interop_all_assertions_pass_mock():
    result: DemoResult = demo_A2_A3_interop()
    assert result.name == "A2_A3_interop"
    assert result.passed is True, _format_failures(result)
    names = {a.name for a in result.assertions}
    expected = {
        "digest_identified_as_affected",
        "rerun_produces_poison_free_output",
    }
    assert names == expected, f"Missing or extra assertions: {names ^ expected}"
    for a in result.assertions:
        assert a.passed is True, a.evidence


def test_a2a3_interop_metrics():
    result = demo_A2_A3_interop()
    assert result.metrics["poisoned_turn"] == 5
    assert result.metrics["framework"] == "langgraph"


def _format_failures(result: DemoResult) -> str:
    lines = ["A2∩A3 interop demo failed:"]
    for a in result.assertions:
        if not a.passed:
            lines.append(f"  - {a.name}: {a.evidence}")
    return "\n".join(lines)
