"""A5 — Reasoning-degradation monitoring (detector-correctness redesign).

SPEC re-designed 2026-07-06: A5 tests the DegradationMonitor detector on
known fixture score sequences, not the agent's actual degradation under
truncation. The detector is app-owned; it accepts per-turn quality scores
(from any judge) + baseline, computes sustained-delta, and fires when
threshold + k consecutive samples are both met.

Assertions:
  (a) fire on known-degrading sequence
  (b) NOT fire on stable sequence (false-positive guard)
  (c) NOT fire on single-dip-then-recovery (trend-based, not single-sample)
  (d) mitigation loop: detector fires → mitigation hook is invoked (spy/mock)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_lifecycle_harness.degradation import DegradationMonitor


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


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _record_fixture(
    monitor: DegradationMonitor,
    session_id: str,
    scores: list[float],
    baseline: float = 0.9,
    truncated_flags: list[bool] | None = None,
) -> None:
    """Feed a fixture score sequence into the monitor."""
    if truncated_flags is None:
        truncated_flags = [s < baseline for s in scores]
    for turn, (score, truncated) in enumerate(zip(scores, truncated_flags), start=1):
        monitor.record_turn(
            session_id=session_id,
            turn=turn,
            score=score,
            baseline_score=baseline,
            truncated=truncated,
            model="fixture-judge",
        )


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_fire_on_degrading(monitor: DegradationMonitor, session_id: str) -> AssertionResult:
    """(a) fire on known-degrading sequence: turns 1-10 score 0.9, turns 11-20 score 0.5."""
    alerts = monitor.evaluate(session_id)
    if not alerts:
        return AssertionResult(
            name="degradation_detected",
            passed=False,
            evidence="No degradation alerts fired on known-degrading fixture.",
        )
    alert = alerts[0]
    if alert.sustained_count < 3:
        return AssertionResult(
            name="degradation_detected",
            passed=False,
            evidence=f"Sustained count {alert.sustained_count} < 3.",
        )
    return AssertionResult(
        name="degradation_detected",
        passed=True,
        evidence=f"Fired: delta={alert.actual:.3f}, sustained={alert.sustained_count}.",
    )


def assert_no_fire_on_stable(monitor: DegradationMonitor, session_id: str) -> AssertionResult:
    """(b) NOT fire on stable sequence: all turns score 0.9."""
    alerts = monitor.evaluate(session_id)
    if alerts:
        return AssertionResult(
            name="control_no_false_positive",
            passed=False,
            evidence=f"False positive: {len(alerts)} alert(s) fired on stable fixture.",
        )
    return AssertionResult(
        name="control_no_false_positive",
        passed=True,
        evidence="No alerts on stable fixture.",
    )


def assert_no_fire_on_single_dip(monitor: DegradationMonitor, session_id: str) -> AssertionResult:
    """(c) NOT fire on single-dip-then-recovery: turn 5 = 0.5, others 0.9."""
    alerts = monitor.evaluate(session_id)
    if alerts:
        return AssertionResult(
            name="trend_based",
            passed=False,
            evidence=f"Single-dip fixture fired {len(alerts)} alert(s).",
        )
    return AssertionResult(
        name="trend_based",
        passed=True,
        evidence="No alerts on single-dip fixture.",
    )


def assert_mitigation_hook_fires(
    monitor: DegradationMonitor,
    session_id: str,
    alerts: list[Any] | None = None,
) -> AssertionResult:
    """(d) mitigation loop: detector fires → mitigation hook is invoked (spy)."""
    if alerts is None:
        alerts = monitor.evaluate(session_id)
    if not alerts:
        return AssertionResult(
            name="mitigation_hook_fires",
            passed=False,
            evidence="Detector did not fire, cannot verify mitigation hook.",
        )
    hook_calls = getattr(monitor, "_hook_call_count", 0)
    if hook_calls == 0:
        return AssertionResult(
            name="mitigation_hook_fires",
            passed=False,
            evidence="Mitigation hook was not invoked despite detector firing.",
        )
    return AssertionResult(
        name="mitigation_hook_fires",
        passed=True,
        evidence=f"Hook invoked {hook_calls} time(s) after {alerts[0].sustained_count} sustained samples.",
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------

def demo_A5_degradation(
    harness: Any | None = None,
    n_turns: int = 30,
    truncate_after_turn: int = 20,
    truncate_window: int = 5,
) -> DemoResult:
    """Run the A5 detector-correctness scenario on fixture sequences.

    Note: this demo no longer runs a real 30-turn agent or uses _mock_judge_score.
    It feeds known score sequences directly into DegradationMonitor and asserts
    the detector behaves correctly.
    """
    assertions: list[AssertionResult] = []

    # (a) fire on known-degrading sequence
    monitor_a = DegradationMonitor(delta_threshold=0.05, min_sustained=3)
    degraded_scores = [0.9] * 10 + [0.5] * 10
    _record_fixture(monitor_a, "session-degrading", degraded_scores, baseline=0.9)
    assertions.append(assert_fire_on_degrading(monitor_a, "session-degrading"))

    # (b) NOT fire on stable sequence
    monitor_b = DegradationMonitor(delta_threshold=0.05, min_sustained=3)
    stable_scores = [0.9] * 20
    _record_fixture(monitor_b, "session-stable", stable_scores, baseline=0.9)
    assertions.append(assert_no_fire_on_stable(monitor_b, "session-stable"))

    # (c) NOT fire on single-dip-then-recovery
    monitor_c = DegradationMonitor(delta_threshold=0.05, min_sustained=3)
    dip_scores = [0.9, 0.9, 0.9, 0.9, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]
    dip_flags = [False, False, False, False, True] + [False] * 5
    _record_fixture(monitor_c, "session-dip", dip_scores, baseline=0.9, truncated_flags=dip_flags)
    assertions.append(assert_no_fire_on_single_dip(monitor_c, "session-dip"))

    # (d) mitigation loop: spy on the hook
    hook_spy: list[dict] = []

    def _spy_hook(alert: Any) -> None:
        hook_spy.append({
            "metric": alert.metric,
            "session_id": alert.session_id,
            "delta": alert.actual,
            "sustained": alert.sustained_count,
        })

    monitor_d = DegradationMonitor(
        delta_threshold=0.05,
        min_sustained=3,
        on_alert=_spy_hook,
    )
    _record_fixture(monitor_d, "session-mitigation", degraded_scores, baseline=0.9)
    alerts_d = monitor_d.evaluate("session-mitigation")
    assertions.append(assert_mitigation_hook_fires(monitor_d, "session-mitigation", alerts=alerts_d))

    passed = all(a.passed for a in assertions)
    metrics = {
        "delta_threshold": 0.05,
        "min_sustained": 3,
        "fixture_sequences": {
            "degrading": degraded_scores,
            "stable": stable_scores,
            "single_dip": dip_scores,
        },
        "framework": "langgraph",
    }
    return DemoResult(name="A5_degradation", passed=passed, assertions=assertions, metrics=metrics)
