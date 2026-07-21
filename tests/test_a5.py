"""A5 degradation tests (detector-correctness redesign).

Runs in mock mode via config.ci.yaml. Real-LLM path is exercised by
`python -m agent_lifecycle_harness.run --demo A5` once `config.yaml` is filled.
"""

from __future__ import annotations

import pytest

from agent_lifecycle_harness.demos.a5_degradation import DemoResult, demo_A5_degradation


def test_a5_all_assertions_pass_mock():
    result: DemoResult = demo_A5_degradation()
    assert result.name == "A5_degradation"
    assert result.passed is True, _format_failures(result)
    names = {a.name for a in result.assertions}
    expected = {
        "degradation_detected",
        "control_no_false_positive",
        "trend_based",
        "exactly_one_alert",
        "mitigation_hook_fires_once",
        "scores_from_judge",
    }
    assert names == expected, f"Missing or extra assertions: {names ^ expected}"
    for a in result.assertions:
        assert a.passed is True, a.evidence


def test_a5_judge_fed_and_edge_triggered():
    result = demo_A5_degradation()
    metrics = result.metrics
    assert metrics["delta_threshold"] == 0.05
    assert metrics["min_sustained"] == 3
    # Judge-fed: scores came from judge.score() calls.
    assert metrics["judge"] == "MockJudge"
    assert metrics["judge_score_calls"] > 0
    # Edge-triggered: one degradation event → one alert + one hook call.
    assert metrics["alerts_on_degrading_fixture"] == 1
    assert metrics["mitigation_hook_calls"] == 1
    # No fake framework label.
    assert "framework" not in metrics


def _format_failures(result: DemoResult) -> str:
    lines = ["A5 demo failed:"]
    for a in result.assertions:
        if not a.passed:
            lines.append(f"  - {a.name}: {a.evidence}")
    return "\n".join(lines)
