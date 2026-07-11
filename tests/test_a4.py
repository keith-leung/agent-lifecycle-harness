"""A4 hot-reload tests.

Runs in mock mode via config.ci.yaml. Real-LLM path is exercised by
`python -m agent_lifecycle_harness.run --demo A4` once `config.yaml` is filled.
"""

from __future__ import annotations

import pytest

from agent_lifecycle_harness.demos.a4_hotreload import DemoResult, demo_A4_hotreload


def test_a4_all_assertions_pass_mock():
    result: DemoResult = demo_A4_hotreload()
    assert result.name == "A4_hotreload"
    assert result.passed is True, _format_failures(result)
    names = {a.name for a in result.assertions}
    expected = {
        "ongoing_session_retains_version",
        "new_session_picks_up_latest",
    }
    assert names == expected, f"Missing or extra assertions: {names ^ expected}"
    for a in result.assertions:
        assert a.passed is True, a.evidence


def test_a4_metrics():
    result = demo_A4_hotreload()
    assert result.metrics["session_a_version"] == "v1"
    assert result.metrics["session_b_version"] == "v2"
    assert result.metrics["framework"] == "langgraph"


def _format_failures(result: DemoResult) -> str:
    lines = ["A4 demo failed:"]
    for a in result.assertions:
        if not a.passed:
            lines.append(f"  - {a.name}: {a.evidence}")
    return "\n".join(lines)
