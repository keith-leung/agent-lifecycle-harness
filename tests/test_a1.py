"""A1 isolation tests.

Runs in mock mode via config.ci.yaml. Real-LLM path is exercised by
`python -m agent_lifecycle_harness.run --demo A1` once `config.yaml` is filled.
"""

from __future__ import annotations

import pytest

from agent_lifecycle_harness.demos.a1_isolation import DemoResult, demo_A1_isolation


def test_a1_all_assertions_pass_mock():
    result: DemoResult = demo_A1_isolation()
    assert result.name == "A1_isolation"
    assert result.passed is True, _format_failures(result)
    names = {a.name for a in result.assertions}
    expected = {
        "resume_own_history_only",
        "no_cross_thread_key_leak",
        "independent_checkpoint_counts",
        "same_thread_writers_serialize",
        "accidental_reuse_then_fix",
    }
    assert names == expected, f"Missing or extra assertions: {names ^ expected}"
    for a in result.assertions:
        assert a.passed is True, a.evidence


def test_a1_metrics():
    result = demo_A1_isolation()
    assert result.metrics["threads"] == 3
    assert result.metrics["framework"] == "langgraph"


def _format_failures(result: DemoResult) -> str:
    lines = ["A1 demo failed:"]
    for a in result.assertions:
        if not a.passed:
            lines.append(f"  - {a.name}: {a.evidence}")
    return "\n".join(lines)
