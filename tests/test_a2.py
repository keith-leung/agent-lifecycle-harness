"""A2 compaction tests.

Runs in mock mode via config.ci.yaml. Real-LLM path is exercised by
`python -m agent_lifecycle_harness.run --demo A2` once `config.yaml` is filled.
"""

from __future__ import annotations

import pytest

from agent_lifecycle_harness.demos.a2_compaction import DemoResult, demo_A2_compaction


def test_a2_all_assertions_pass_mock():
    result: DemoResult = demo_A2_compaction()
    assert result.name == "A2_compaction"
    assert result.passed is True, _format_failures(result)
    names = {a.name for a in result.assertions}
    expected = {
        "compaction_shape",
        "coherence_after_compaction",
        "compactor_idempotent",
        "lossy_fields_enumerated",
    }
    assert names == expected, f"Missing or extra assertions: {names ^ expected}"
    for a in result.assertions:
        assert a.passed is True, a.evidence


def test_a2_metrics():
    result = demo_A2_compaction()
    assert result.metrics["n_turns"] == 10
    assert result.metrics["first_last_n"] == 3
    assert result.metrics["framework"] == "langgraph"


def _format_failures(result: DemoResult) -> str:
    lines = ["A2 demo failed:"]
    for a in result.assertions:
        if not a.passed:
            lines.append(f"  - {a.name}: {a.evidence}")
    return "\n".join(lines)
