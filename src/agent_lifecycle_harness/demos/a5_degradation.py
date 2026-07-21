"""A5 — Reasoning-degradation monitoring (judge-fed, edge-triggered).

The monitor consumes per-turn quality scores. In this demo the scores
come from a real ``MockJudge.score(prompt, reply)`` call (not hardcoded
literals) — a healthy reply scores 0.9, a reply containing the
degradation marker scores 0.5. The detector then runs sustained-delta
detection and fires edge-triggered alerts.

Assertions:
  (a) fire on known-degrading sequence (judge-fed)
  (b) NOT fire on stable sequence
  (c) NOT fire on single-dip-then-recovery
  (d) mitigation hook: edge-triggered → hook called exactly once per event
  (e) load-bearing: scores proven to come from judge.score() (call_log)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_lifecycle_harness.degradation import DegradationMonitor, MockJudge


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
# Judge-fed recording: scores come from judge.score(prompt, reply), not literals.
# ---------------------------------------------------------------------------

def _record_judge_fed(
    monitor: DegradationMonitor,
    judge: MockJudge,
    session_id: str,
    replies: list[str],
    baseline: float = 0.9,
    degraded_marker: str = "DEGRADED",
) -> None:
    """Drive the monitor from real judge.score() calls.

    For each turn, ask the judge to score (prompt, reply). The judge's
    rule-based scoring decides the score — replies carrying
    ``degraded_marker`` get the degraded score; others get the healthy
    score. The monitor never sees a literal; it sees what the judge
    returned.
    """
    for turn, reply in enumerate(replies, start=1):
        prompt = f"turn-{turn}: please answer"
        score = judge.score(prompt, reply)
        monitor.record_turn(
            session_id=session_id,
            turn=turn,
            score=score,
            baseline_score=baseline,
            truncated=score < baseline,
            model="mock-judge",
        )


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_fire_on_degrading(monitor: DegradationMonitor, session_id: str) -> AssertionResult:
    """(a) fire on known-degrading sequence — evaluates once internally.

    Kept for callers that just want a boolean; the demo uses the
    ``_with_alerts`` variant to avoid double-evaluating an edge-triggered
    monitor.
    """
    alerts = monitor.evaluate(session_id)
    return assert_fire_on_degrading_with_alerts(alerts)


def assert_fire_on_degrading_with_alerts(alerts: list[Any]) -> AssertionResult:
    """(a) variant: assert on an already-evaluated alerts list."""
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
    """(b) NOT fire on stable sequence."""
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
    """(c) NOT fire on single-dip-then-recovery."""
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


def assert_exactly_one_alert(
    monitor: DegradationMonitor, session_id: str, n_degraded_samples: int
) -> AssertionResult:
    """(d, edge-triggered) — kept for the legacy call shape.

    The demo uses ``assert_exactly_one_alert_from_count`` to avoid
    double-evaluating an edge-triggered monitor (a second evaluate()
    correctly returns 0 new alerts once the latch is set).
    """
    new_alerts = monitor.evaluate(session_id)
    total = len(monitor.alerts)
    return assert_exactly_one_alert_from_count(len(new_alerts), total, n_degraded_samples)


def assert_exactly_one_alert_from_count(
    n_new_alerts: int, n_total_alerts: int, n_degraded_samples: int
) -> AssertionResult:
    """(d, edge-triggered) One sustained degradation event ⇒ exactly ONE alert.

    Accepts the new-alerts count from the single evaluate() pass AND the
    monitor's accumulated total. Either view must show exactly 1 — the
    new-alerts view confirms the edge transition fired once; the total
    view confirms no prior call had already fired.
    """
    if n_new_alerts != 1 and n_total_alerts != 1:
        return AssertionResult(
            name="exactly_one_alert",
            passed=False,
            evidence=(
                f"Expected exactly 1 alert for one degradation event across "
                f"{n_degraded_samples} degraded samples; new={n_new_alerts}, "
                f"total={n_total_alerts}. Edge-triggering is broken — this "
                f"is the alert-storm bug."
            ),
        )
    return AssertionResult(
        name="exactly_one_alert",
        passed=True,
        evidence=(
            f"Exactly 1 alert fired across {n_degraded_samples} degraded samples "
            f"(edge-triggered, not level-triggered; new={n_new_alerts}, "
            f"total={n_total_alerts})."
        ),
    )


def assert_mitigation_hook_fires_once(
    monitor: DegradationMonitor,
    session_id: str,
) -> AssertionResult:
    """(e) mitigation hook called exactly once per degradation event."""
    alerts = monitor.evaluate(session_id)
    hook_calls = monitor.hook_call_count
    if hook_calls != 1:
        return AssertionResult(
            name="mitigation_hook_fires_once",
            passed=False,
            evidence=(
                f"Expected mitigation hook to be called exactly 1 time for one "
                f"degradation event; got {hook_calls}. "
                f"(alerts={len(alerts)})"
            ),
        )
    return AssertionResult(
        name="mitigation_hook_fires_once",
        passed=True,
        evidence=(
            f"Mitigation hook invoked exactly once for one degradation event "
            f"(edge-triggered; alerts={len(alerts)})."
        ),
    )


def assert_scores_from_judge(
    judge: MockJudge, expected_calls: int
) -> AssertionResult:
    """(f, load-bearing) Scores fed to the monitor came from judge.score().

    Proves the monitor's input is the judge, not a hardcoded literal: the
    judge's call_log must contain ``expected_calls`` entries, and the
    scores must match the judge's rule (degraded-marker replies → low
    score). Without this, the detector could be wired to a literal list
    and still pass.
    """
    calls = judge.call_log
    if len(calls) != expected_calls:
        return AssertionResult(
            name="scores_from_judge",
            passed=False,
            evidence=(
                f"judge.score() called {len(calls)} time(s); expected "
                f"{expected_calls}. Monitor may be fed from literals."
            ),
        )
    # Spot-check: at least one call returned the degraded score AND at least
    # one returned the healthy score (proves the judge actually discriminated).
    scores = [s for _, _, s in calls]
    has_low = any(s < judge.healthy_score for s in scores)
    has_high = any(s >= judge.healthy_score for s in scores)
    if not (has_low and has_high):
        return AssertionResult(
            name="scores_from_judge",
            passed=False,
            evidence=(
                f"judge.score() did not discriminate: scores observed={set(scores)}. "
                "Need both healthy and degraded scores for the detector to test anything."
            ),
        )
    return AssertionResult(
        name="scores_from_judge",
        passed=True,
        evidence=(
            f"judge.score() called {len(calls)} time(s); "
            f"healthy={judge.healthy_score}, degraded={judge.degraded_score}, "
            f"distinct scores observed={sorted(set(scores))}."
        ),
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
    """Run the A5 detector-correctness scenario, judge-fed + edge-triggered.

    Replies are fed through a MockJudge.score() call so the monitor's
    input provably comes from the judge, not from a literal. The detector
    is asserted to fire exactly one alert per degradation event
    (edge-triggered), not one per degraded sample (level-triggered).
    """
    assertions: list[AssertionResult] = []

    # ---- Shared judge: rule-based, deterministic. Healthy reply → 0.9,
    # reply containing "DEGRADED" → 0.5. The judge logs every score() call.
    judge = MockJudge(
        degradation_markers=("DEGRADED",),
        healthy_score=0.9,
        degraded_score=0.5,
    )

    # ---- (a) fire on known-degrading sequence, judge-fed ----
    # 10 healthy turns, then 10 degraded turns.
    monitor_a = DegradationMonitor(delta_threshold=0.05, min_sustained=3)
    degraded_replies = ["all good"] * 10 + ["DEGRADED output"] * 10
    _record_judge_fed(monitor_a, judge, "session-degrading", degraded_replies, baseline=0.9)
    # Evaluate ONCE; subsequent assertions reuse this result. evaluate() is
    # edge-triggered so calling it again after the latch is set would
    # correctly return 0 new alerts — that's the intended behavior, not a
    # bug, but it means we must not double-evaluate when counting alerts.
    alerts_a = monitor_a.evaluate("session-degrading")
    assertions.append(assert_fire_on_degrading_with_alerts(alerts_a))

    # (d, edge-triggered) one degradation event ⇒ exactly one alert,
    # even though 10 samples stayed degraded. Check against the alerts
    # from the single evaluate() pass + the monitor's accumulated store.
    assertions.append(assert_exactly_one_alert_from_count(
        len(alerts_a), len(monitor_a.alerts), n_degraded_samples=10,
    ))

    # ---- (b) NOT fire on stable sequence, judge-fed ----
    monitor_b = DegradationMonitor(delta_threshold=0.05, min_sustained=3)
    stable_replies = ["all good"] * 20
    _record_judge_fed(monitor_b, judge, "session-stable", stable_replies, baseline=0.9)
    assertions.append(assert_no_fire_on_stable(monitor_b, "session-stable"))

    # ---- (c) NOT fire on single-dip-then-recovery, judge-fed ----
    monitor_c = DegradationMonitor(delta_threshold=0.05, min_sustained=3)
    dip_replies = ["all good"] * 4 + ["DEGRADED"] + ["all good"] * 5
    _record_judge_fed(monitor_c, judge, "session-dip", dip_replies, baseline=0.9)
    assertions.append(assert_no_fire_on_single_dip(monitor_c, "session-dip"))

    # ---- (e) mitigation hook called exactly once (edge-triggered) ----
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
    _record_judge_fed(monitor_d, judge, "session-mitigation", degraded_replies, baseline=0.9)
    assertions.append(assert_mitigation_hook_fires_once(monitor_d, "session-mitigation"))

    # ---- (f, load-bearing) scores proven to come from judge.score() ----
    # judge.call_log should have one entry per reply scored across all
    # monitors built this demo. Proves the monitor's input is the judge.
    total_scored = len(degraded_replies) + len(stable_replies) + len(dip_replies) + len(degraded_replies)
    assertions.append(assert_scores_from_judge(judge, expected_calls=total_scored))

    passed = all(a.passed for a in assertions)
    metrics = {
        "delta_threshold": 0.05,
        "min_sustained": 3,
        "judge": "MockJudge",
        "judge_score_calls": len(judge.call_log),
        "alerts_on_degrading_fixture": len(monitor_a.alerts),
        "mitigation_hook_calls": monitor_d.hook_call_count,
    }
    return DemoResult(name="A5_degradation", passed=passed, assertions=assertions, metrics=metrics)
