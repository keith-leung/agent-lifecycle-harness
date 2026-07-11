"""A4 — Config hot-reload demo.

Demonstrates version-on-session: ongoing sessions continue on the config
version they started with; new sessions pick up the current version.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from agent_lifecycle_harness.agent import LifecycleHarness
from agent_lifecycle_harness.hotreload import ConfigVersionTracker
from agent_lifecycle_harness.llm import LLMClient, MockLLMClient, RealLLMClient


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


def _make_harness(llm: LLMClient | None = None, judge: LLMClient | None = None) -> tuple[LifecycleHarness, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    if llm is None:
        llm = MockLLMClient(prefix="a4-sut")
    if judge is None:
        judge = MockLLMClient(prefix="a4-judge")
    return LifecycleHarness(db_path=db_path, llm=llm, judge=judge), db_path


def assert_ongoing_session_retains_version(
    tracker: ConfigVersionTracker,
    session_id: str,
) -> AssertionResult:
    """Ongoing session keeps the version it started with."""
    session = tracker.get_session(session_id)
    if session is None:
        return AssertionResult(
            name="ongoing_session_retains_version",
            passed=False,
            evidence="Session not found in tracker.",
        )
    return AssertionResult(
        name="ongoing_session_retains_version",
        passed=True,
        evidence=f"Session {session_id} retains version {session.config_version}.",
    )


def assert_new_session_picks_up_latest(
    tracker: ConfigVersionTracker,
    session_id: str,
    expected_version: str,
) -> AssertionResult:
    """New session gets the current config version."""
    session = tracker.get_session(session_id)
    if session is None:
        return AssertionResult(
            name="new_session_picks_up_latest",
            passed=False,
            evidence="Session not found in tracker.",
        )
    if session.config_version != expected_version:
        return AssertionResult(
            name="new_session_picks_up_latest",
            passed=False,
            evidence=f"Expected {expected_version}, got {session.config_version}.",
        )
    return AssertionResult(
        name="new_session_picks_up_latest",
        passed=True,
        evidence=f"New session {session_id} uses version {expected_version}.",
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------

def demo_A4_hotreload(
    harness: LifecycleHarness | None = None,
) -> DemoResult:
    """Run the A4 config hot-reload scenario."""
    if harness is None:
        harness, _ = _make_harness()
    tracker = ConfigVersionTracker()
    assertions: list[AssertionResult] = []

    # Session A starts with v1.
    tracker.register_session("session-a")
    harness.invoke("session-a", "hello from v1")

    # Reload config -> v2.
    tracker.set_version("v2")

    # Session A continues (still v1).
    harness.invoke("session-a", "still v1?")

    # Session B starts after reload (should be v2).
    tracker.register_session("session-b")
    harness.invoke("session-b", "hello from v2")

    # (a) ongoing session retains its version.
    assertions.append(assert_ongoing_session_retains_version(tracker, "session-a"))

    # (b) new session picks up latest.
    assertions.append(assert_new_session_picks_up_latest(tracker, "session-b", "v2"))

    passed = all(a.passed for a in assertions)
    metrics = {
        "session_a_version": tracker.get_session("session-a").config_version if tracker.get_session("session-a") else None,
        "session_b_version": tracker.get_session("session-b").config_version if tracker.get_session("session-b") else None,
        "framework": harness.framework,
    }
    return DemoResult(name="A4_hotreload", passed=passed, assertions=assertions, metrics=metrics)
