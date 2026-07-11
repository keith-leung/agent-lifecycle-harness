"""A5 — Reasoning degradation monitoring.

SPEC: degradation = context truncation (history-growth failure mode).
End-to-end: a long-running agent whose visible history is truncated
from turn N onward should exhibit degraded output quality, detected
by a lightweight cross-vendor judge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence


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


class DegradationMonitor:
    """App-owned degradation monitor for context-truncation-driven quality loss.

    Tracks judge scores per turn. Fires when sustained delta exceeds
    threshold for k consecutive samples.
    """

    def __init__(
        self,
        *,
        delta_threshold: float = 0.15,
        min_sustained: int = 3,
        on_alert: Any = None,
    ) -> None:
        self.delta_threshold = delta_threshold
        self.min_sustained = min_sustained
        self._samples: list[MetricSample] = []
        self._alerts: list[Alert] = []
        self._on_alert = on_alert
        self._hook_call_count = 0

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

        Returns list of alerts fired.
        """
        session_samples = [s for s in self._samples if s.session_id == session_id]
        alerts: list[Alert] = []
        sustained = 0
        for sample in session_samples:
            if sample.delta > self.delta_threshold:
                sustained += 1
                if sustained >= self.min_sustained:
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
                    alerts.append(alert)
            else:
                sustained = 0
        self._alerts.extend(alerts)
        return alerts

    @property
    def alerts(self) -> list[Alert]:
        return list(self._alerts)

    @property
    def samples(self) -> list[MetricSample]:
        return list(self._samples)
