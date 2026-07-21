"""A5 — Reasoning degradation monitoring.

Judge-fed: per-turn quality scores come from a judge's ``score()`` call
(or a MockJudge in tests), not from hardcoded literals. The monitor
computes sustained-delta and fires **edge-triggered** alerts — one alert
per degradation event (transition into degraded state), not one per
sample. Recovery clears the latch so a subsequent degradation fires
again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence


@dataclass
class MetricSample:
    session_id: str
    turn: int
    score: float
    baseline_score: float
    delta: float
    truncated: bool
    model: str
    ts: str


@dataclass
class Alert:
    metric: str
    session_id: str
    threshold: float
    actual: float
    sustained_count: int
    ts: str


@dataclass
class DegradationReport:
    alerts: list[Alert]
    samples: list[MetricSample]


# ---------------------------------------------------------------------------
# Judge interface + a MockJudge for tests.
# ---------------------------------------------------------------------------

class Judge:
    """Abstract judge: scores a (prompt, reply) pair in [0, 1]."""

    def score(self, prompt: str, reply: str) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class MockJudge(Judge):
    """Deterministic rule-based judge for tests.

    Replies containing any of ``degradation_markers`` score
    ``degraded_score``; otherwise ``healthy_score``. Used by the A5 demo
    to drive the monitor from real judge.score() calls instead of fixture
    literals.
    """

    def __init__(
        self,
        *,
        degradation_markers: Sequence[str] = ("DEGRADED",),
        healthy_score: float = 0.9,
        degraded_score: float = 0.5,
    ) -> None:
        self.degradation_markers = tuple(degradation_markers)
        self.healthy_score = healthy_score
        self.degraded_score = degraded_score
        # Track every score() call so tests can prove the monitor is fed
        # from the judge, not from a literal.
        self.call_log: list[tuple[str, str, float]] = []

    def score(self, prompt: str, reply: str) -> float:
        s = self.degraded_score if any(m in reply for m in self.degradation_markers) else self.healthy_score
        self.call_log.append((prompt, reply, s))
        return s


# ---------------------------------------------------------------------------
# DegradationMonitor — edge-triggered alerts.
# ---------------------------------------------------------------------------

class DegradationMonitor:
    """App-owned degradation monitor for context-truncation-driven quality loss.

    Tracks judge scores per turn. Fires when sustained delta exceeds
    threshold for ``min_sustained`` consecutive samples.

    Edge-triggered: a single sustained degradation event produces exactly
    ONE alert (the transition into the degraded state). Subsequent
    samples that stay degraded do not re-fire. If the series recovers
    (delta drops below threshold), the latch resets and a later
    degradation will fire again. This prevents alert storms: a 100-turn
    degraded run produces 1 alert, not 98.
    """

    def __init__(
        self,
        *,
        delta_threshold: float = 0.15,
        min_sustained: int = 3,
        on_alert: Callable[[Alert], None] | None = None,
    ) -> None:
        self.delta_threshold = delta_threshold
        self.min_sustained = min_sustained
        self._samples: list[MetricSample] = []
        self._alerts: list[Alert] = []
        self._on_alert = on_alert
        self._hook_call_count = 0
        # Per-session edge-trigger latch: once an alert has fired for a
        # sustained degradation, don't fire again until recovery clears it.
        self._alerted_sessions: set[str] = set()

    def record_turn(
        self,
        session_id: str,
        turn: int,
        *,
        score: float,
        baseline_score: float,
        truncated: bool,
        model: str = "",
    ) -> MetricSample:
        delta = baseline_score - score
        sample = MetricSample(
            session_id=session_id,
            turn=turn,
            score=score,
            baseline_score=baseline_score,
            delta=delta,
            truncated=truncated,
            model=model,
            ts=datetime.now(timezone.utc).isoformat(),
        )
        self._samples.append(sample)
        return sample

    def evaluate(self, session_id: str) -> list[Alert]:
        """Evaluate all samples for sustained degradation.

        Edge-triggered: returns at most one alert per degradation event.
        A session that stays degraded across N samples fires once on
        transition; it fires again only after recovery clears the latch.
        """
        session_samples = [s for s in self._samples if s.session_id == session_id]
        new_alerts: list[Alert] = []
        sustained = 0
        # Track recovery within this pass so the latch resets correctly.
        recovered = False
        for sample in session_samples:
            degraded = sample.delta > self.delta_threshold
            if degraded:
                sustained += 1
                if sustained >= self.min_sustained and session_id not in self._alerted_sessions:
                    alert = Alert(
                        metric="score_delta",
                        session_id=session_id,
                        threshold=self.delta_threshold,
                        actual=sample.delta,
                        sustained_count=sustained,
                        ts=sample.ts,
                    )
                    if self._on_alert is not None:
                        self._on_alert(alert)
                        self._hook_call_count += 1
                    new_alerts.append(alert)
                    # Latch: do not fire again for this session until recovery.
                    self._alerted_sessions.add(session_id)
            else:
                # Recovery: a non-degraded sample clears the sustained run
                # AND the latch, so a new degradation event can fire again.
                if sustained >= self.min_sustained:
                    recovered = True
                    self._alerted_sessions.discard(session_id)
                sustained = 0
        self._alerts.extend(new_alerts)
        return new_alerts

    def reset(self, session_id: str | None = None) -> None:
        """Clear samples (and latches). If session_id given, clear only that session."""
        if session_id is None:
            self._samples.clear()
            self._alerts.clear()
            self._alerted_sessions.clear()
            self._hook_call_count = 0
        else:
            self._samples = [s for s in self._samples if s.session_id != session_id]
            self._alerts = [a for a in self._alerts if a.session_id != session_id]
            self._alerted_sessions.discard(session_id)

    @property
    def alerts(self) -> list[Alert]:
        return list(self._alerts)

    @property
    def samples(self) -> list[MetricSample]:
        return list(self._samples)

    @property
    def hook_call_count(self) -> int:
        return self._hook_call_count
