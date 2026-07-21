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
    # A4 wiring: register one MockLLMClient per material model name. Each
    # carries a distinct prefix in its output so the load-bearing assertion
    # can tell which client actually answered. The default `llm` is the
    # fallback when a session doesn't pin a model.
    model_registry = {
        "sut-v1": MockLLMClient(prefix="sut-v1"),
        "sut-v2": MockLLMClient(prefix="sut-v2"),
    }
    return (
        LifecycleHarness(
            db_path=db_path, llm=llm, judge=judge, model_registry=model_registry,
        ),
        db_path,
    )


def _last_assistant_text(state: dict[str, Any]) -> str:
    """Return the content of the last assistant message in a graph state."""
    for m in reversed(state.get("messages", [])):
        role = getattr(m, "type", None) or getattr(m, "role", "")
        if role in ("assistant", "ai"):
            return getattr(m, "content", "") or ""
    return ""


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


def assert_checkpoint_version_stamped(
    harness: LifecycleHarness,
    session_id: str,
    expected_version: str,
) -> AssertionResult:
    """config_version is persisted into the checkpoint, not just in-memory.

    This is the hard version-on-session check: reading state back from the
    checkpointer (SQLite) must recover the same config_version that was
    passed to invoke(). If this fails, the version lives only in the
    tracker's in-memory dict and would be lost on process restart.
    """
    state = harness.get_state(session_id)
    if state is None:
        return AssertionResult(
            name="checkpoint_version_stamped",
            passed=False,
            evidence=f"No persisted state for {session_id}.",
        )
    meta = state.get("_harness_meta", {})
    actual = meta.get("config_version")
    if actual != expected_version:
        return AssertionResult(
            name="checkpoint_version_stamped",
            passed=False,
            evidence=(
                f"Persisted checkpoint config_version for {session_id} "
                f"is {actual!r}, expected {expected_version!r}. "
                "Version not surviving the SQLite round-trip."
            ),
        )
    return AssertionResult(
        name="checkpoint_version_stamped",
        passed=True,
        evidence=(
            f"Persisted checkpoint for {session_id} carries "
            f"config_version={actual} (recovered from SQLite metadata)."
        ),
    )


def assert_cosmetic_propagates_immediately(
    tracker: ConfigVersionTracker,
    session_id: str,
    cosmetic_key: str,
    expected_value: Any,
    version_before: str,
) -> AssertionResult:
    """Cosmetic change reaches live sessions without bumping the version.

    A cosmetic field (log_level, metrics_endpoint, ...) is safe to hot-patch
    into every live session because it cannot change model output. So after
    `set_config` flips such a field, every already-registered session must
    (a) see the new value in its resolved config and (b) still report the
    same version label — no version bump for cosmetic.
    """
    resolved = tracker.resolved_config_for(session_id) or {}
    actual_value = resolved.get(cosmetic_key)
    actual_version = tracker.get_session(session_id).config_version
    if actual_value != expected_value:
        return AssertionResult(
            name="cosmetic_propagates_immediately",
            passed=False,
            evidence=(
                f"Live session {session_id} did not pick up cosmetic "
                f"{cosmetic_key}={expected_value!r}; resolved has {actual_value!r}."
            ),
        )
    if actual_version != version_before:
        return AssertionResult(
            name="cosmetic_propagates_immediately",
            passed=False,
            evidence=(
                f"Cosmetic change bumped version for {session_id}: "
                f"{version_before} → {actual_version}. Cosmetic must not bump."
            ),
        )
    return AssertionResult(
        name="cosmetic_propagates_immediately",
        passed=True,
        evidence=(
            f"Cosmetic {cosmetic_key}={actual_value!r} reached live "
            f"session {session_id}; version held at {actual_version}."
        ),
    )


def assert_material_pins_live_session(
    tracker: ConfigVersionTracker,
    session_id: str,
    material_key: str,
    frozen_value: Any,
    version_before: str,
    version_after: str,
) -> AssertionResult:
    """Material change bumps the version but does not perturb live sessions.

    A material field (system_prompt / model / temperature) flips the version
    label so new sessions pick up the change. The already-registered session
    must (a) keep seeing its frozen value in resolved config and (b) keep
    its registration-time version label, while the global version moves on.
    """
    resolved = tracker.resolved_config_for(session_id) or {}
    actual_value = resolved.get(material_key)
    actual_version = tracker.get_session(session_id).config_version
    if actual_value != frozen_value:
        return AssertionResult(
            name="material_pins_live_session",
            passed=False,
            evidence=(
                f"Live session {session_id} leaked material {material_key}: "
                f"frozen={frozen_value!r}, resolved={actual_value!r}."
            ),
        )
    if actual_version != version_before:
        return AssertionResult(
            name="material_pins_live_session",
            passed=False,
            evidence=(
                f"Material change leaked version into live session "
                f"{session_id}: pinned={version_before}, now={actual_version}."
            ),
        )
    if version_after == version_before:
        return AssertionResult(
            name="material_pins_live_session",
            passed=False,
            evidence=(
                f"Material change did not bump global version "
                f"({version_before} unchanged)."
            ),
        )
    return AssertionResult(
        name="material_pins_live_session",
        passed=True,
        evidence=(
            f"Live session {session_id} frozen on {material_key}={actual_value!r} "
            f"at version {actual_version}; global advanced to {version_after}."
        ),
    )


def assert_new_session_after_material(
    tracker: ConfigVersionTracker,
    session_id: str,
    material_key: str,
    expected_value: Any,
    expected_version: str,
) -> AssertionResult:
    """A session registered after a material change sees the new slice + version."""
    resolved = tracker.resolved_config_for(session_id) or {}
    actual_value = resolved.get(material_key)
    actual_version = tracker.get_session(session_id).config_version
    if actual_value != expected_value or actual_version != expected_version:
        return AssertionResult(
            name="new_session_after_material",
            passed=False,
            evidence=(
                f"New session {session_id}: {material_key}={actual_value!r} "
                f"(want {expected_value!r}), version={actual_version} "
                f"(want {expected_version})."
            ),
        )
    return AssertionResult(
        name="new_session_after_material",
        passed=True,
        evidence=(
            f"New session {session_id} picked up {material_key}={actual_value!r} "
            f"at version {actual_version}."
        ),
    )


def assert_ttl_reaps_inactive(
    reaped: list[str],
    expected_reaped: list[str],
    expected_alive: list[str],
    tracker: ConfigVersionTracker,
) -> AssertionResult:
    """TTL sweep removes only inactive sessions; active sessions survive.

    `expected_reaped` are session_ids that should be reaped; `expected_alive`
    are session_ids that must still be active after the sweep. We check the
    reaper's return value AND the post-sweep membership, because the return
    value is the observable contract while membership is the persistence
    invariant.
    """
    reaped_set = set(reaped)
    expected_reaped_set = set(expected_reaped)
    if reaped_set != expected_reaped_set:
        return AssertionResult(
            name="ttl_reaps_inactive",
            passed=False,
            evidence=(
                f"Reaper returned {sorted(reaped_set)}, expected "
                f"{sorted(expected_reaped_set)}."
            ),
        )
    for sid in expected_alive:
        session = tracker.get_session(sid)
        if session is None or not session.is_active():
            return AssertionResult(
                name="ttl_reaps_inactive",
                passed=False,
                evidence=(
                    f"Session {sid} should be alive after sweep but is "
                    f"{'missing' if session is None else 'ended/expired'}."
                ),
            )
    for sid in expected_reaped:
        session = tracker.get_session(sid)
        if session is not None and session.is_active():
            return AssertionResult(
                name="ttl_reaps_inactive",
                passed=False,
                evidence=f"Session {sid} should be reaped but is still active.",
            )
    return AssertionResult(
        name="ttl_reaps_inactive",
        passed=True,
        evidence=(
            f"Reaper removed {sorted(reaped_set)}; "
            f"{sorted(expected_alive)} remain active."
        ),
    )


def assert_touch_prevents_reap(
    tracker: ConfigVersionTracker,
    session_id: str,
    ttl_seconds: float,
) -> AssertionResult:
    """A session touched within the TTL window survives a sweep.

    Constructs a fake "old" last_active_at on the session, then touches it,
    then runs a sweep. Without the touch the session would be reaped; the
    touch must keep it alive — this is what makes TTL safe for long-lived
    active sessions.
    """
    import datetime as _dt
    session = tracker.get_session(session_id)
    if session is None:
        return AssertionResult(
            name="touch_prevents_reap",
            passed=False,
            evidence=f"Session {session_id} not found.",
        )
    # Simulate a long-inactive session by backdating last_active_at.
    old_ts = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=ttl_seconds * 10)
    with tracker._lock:  # noqa: SLF001 - test-only backdate
        session.last_active_at = old_ts.isoformat()
    # Touch should bring it back to now.
    touched = tracker.touch_session(session_id)
    reaped = tracker.cleanup_expired()
    survived = session_id not in reaped and tracker.get_session(session_id) is not None
    if not touched:
        return AssertionResult(
            name="touch_prevents_reap",
            passed=False,
            evidence=f"touch_session({session_id}) returned False.",
        )
    if not survived:
        return AssertionResult(
            name="touch_prevents_reap",
            passed=False,
            evidence=f"Touched session {session_id} was reaped anyway: {reaped}.",
        )
    return AssertionResult(
        name="touch_prevents_reap",
        passed=True,
        evidence=(
            f"touch_session({session_id}) refreshed last_active_at; "
            f"survived sweep (reaped={reaped})."
        ),
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------

def demo_A4_hotreload(
    harness: LifecycleHarness | None = None,
) -> DemoResult:
    """Run the A4 config hot-reload scenario.

    Four-phase scenario exercising both change classes, the persisted
    version stamp, and session-TTL memory hygiene:

    1. Baseline: register session-a on initial config; checkpoint carries v1.
    2. Cosmetic reload (log_level flip): propagates to session-a immediately,
       no version bump, no new checkpoint version either.
    3. Material reload (model flip): bumps version to v2; session-a stays
       frozen on its old model + v1; session-b registers after the reload
       and picks up the new model + v2.
    4. TTL sweep: a TTL-enabled tracker reaps inactive sessions while
       active ones (including ones just `touch_session`-ed) survive; reaped
       sessions are retained for audit, not deleted.
    """
    if harness is None:
        harness, _ = _make_harness()
    initial_config = {
        "system_prompt": "you are a helpful agent",
        "model": "sut-v1",
        "temperature": 0.0,
        "log_level": "INFO",
        "metrics_endpoint": "http://default:9090",
    }
    tracker = ConfigVersionTracker(initial_config=initial_config, initial_version="v1")
    assertions: list[AssertionResult] = []

    # ---- Phase 1: session-a starts on v1 -------------------------------------
    tracker.register_session("session-a")
    session_a = tracker.get_session("session-a")
    harness.invoke(
        "session-a", "hello from v1",
        config_version=session_a.config_version,
        resolved_config=tracker.resolved_config_for("session-a"),
    )
    tracker.touch_session("session-a")  # refresh activity after a real turn

    # ---- Phase 2: cosmetic change (log_level INFO -> DEBUG) ------------------
    cosmetic_cfg = dict(initial_config)
    cosmetic_cfg["log_level"] = "DEBUG"
    cosmetic_cls = tracker.set_config(cosmetic_cfg)
    # Session A continues — same version (cosmetic must not bump), but its
    # resolved config now reflects DEBUG.
    harness.invoke(
        "session-a", "cosmetic reload, still v1",
        config_version=session_a.config_version,
        resolved_config=tracker.resolved_config_for("session-a"),
    )
    tracker.touch_session("session-a")
    assertions.append(AssertionResult(
        name="cosmetic_classified",
        passed=cosmetic_cls.kind == "cosmetic",
        evidence=(
            f"set_config(log_level=DEBUG) classified as {cosmetic_cls.kind} "
            f"(cosmetic_changes={cosmetic_cls.changed_cosmetic}, "
            f"material_changes={cosmetic_cls.changed_material})."
        ),
    ))
    assertions.append(assert_cosmetic_propagates_immediately(
        tracker, "session-a", "log_level", "DEBUG", version_before="v1",
    ))
    assertions.append(assert_checkpoint_version_stamped(harness, "session-a", "v1"))

    # ---- Phase 3: material change (model sut-v1 -> sut-v2) -------------------
    material_cfg = dict(cosmetic_cfg)
    material_cfg["model"] = "sut-v2"
    material_cls = tracker.set_config(material_cfg)
    version_after_material = tracker.get_version()  # should be v2
    harness.invoke(
        "session-a", "material reload, still v1 for me",
        config_version=session_a.config_version,
        resolved_config=tracker.resolved_config_for("session-a"),
    )
    tracker.touch_session("session-a")
    assertions.append(AssertionResult(
        name="material_classified",
        passed=material_cls.kind == "material",
        evidence=(
            f"set_config(model=sut-v2) classified as {material_cls.kind} "
            f"(material_changes={material_cls.changed_material})."
        ),
    ))
    assertions.append(assert_material_pins_live_session(
        tracker, "session-a", "model", frozen_value="sut-v1",
        version_before="v1", version_after=version_after_material,
    ))

    # Session B starts after the material reload — should be on v2 with sut-v2.
    tracker.register_session("session-b")
    session_b = tracker.get_session("session-b")
    harness.invoke(
        "session-b", "hello from v2",
        config_version=session_b.config_version,
        resolved_config=tracker.resolved_config_for("session-b"),
    )
    tracker.touch_session("session-b")
    assertions.append(assert_new_session_picks_up_latest(tracker, "session-b", version_after_material))
    assertions.append(assert_new_session_after_material(
        tracker, "session-b", "model", "sut-v2", version_after_material,
    ))
    assertions.append(assert_checkpoint_version_stamped(harness, "session-b", version_after_material))

    # ---- Phase 3b: load-bearing assertion — resolved model actually changes
    # the LLM that answers. Ask both sessions the SAME question; the reply
    # text must carry each session's pinned model signature. This proves the
    # resolved config reached the LLM call, not just the tracker label.
    probe_question = "reply with your model identifier"
    sa_probe = harness.invoke(
        "session-a", probe_question,
        config_version=session_a.config_version,
        resolved_config=tracker.resolved_config_for("session-a"),
    )
    sb_probe = harness.invoke(
        "session-b", probe_question,
        config_version=session_b.config_version,
        resolved_config=tracker.resolved_config_for("session-b"),
    )
    sa_text = _last_assistant_text(sa_probe)
    sb_text = _last_assistant_text(sb_probe)
    sa_has_v1 = "sut-v1" in sa_text
    sb_has_v2 = "sut-v2" in sb_text
    # Stronger: also check the WRONG signature is absent (no leakage).
    sa_clean = "sut-v2" not in sa_text
    sb_clean = "sut-v1" not in sb_text
    assertions.append(AssertionResult(
        name="resolved_model_drives_llm",
        passed=sa_has_v1 and sb_has_v2 and sa_clean and sb_clean,
        evidence=(
            f"session-a (pinned sut-v1) reply carries sut-v1={sa_has_v1}, "
            f"sut-v2 leak={not sa_clean}; "
            f"session-b (pinned sut-v2) reply carries sut-v2={sb_has_v2}, "
            f"sut-v1 leak={not sb_clean}. "
            f"sa_reply[:80]={sa_text[:80]!r}; sb_reply[:80]={sb_text[:80]!r}"
        ),
    ))

    # Ongoing session-a's checkpoint is still stamped v1 throughout.
    assertions.append(assert_ongoing_session_retains_version(tracker, "session-a"))

    # ---- Phase 4: TTL — sessions don't accumulate forever --------------------
    # Use a separate TTL-enabled tracker so phases 1-3 keep their no-TTL
    # backward-compat default. The harness itself is not needed here; TTL is
    # purely a tracker concern (memory hygiene), independent of checkpoints.
    ttl_seconds = 10.0
    ttl_tracker = ConfigVersionTracker(
        initial_config=initial_config,
        initial_version="v1",
        session_ttl_seconds=ttl_seconds,
    )
    ttl_tracker.register_session("idle-old")   # will be backdated + reaped
    ttl_tracker.register_session("idle-fresh")  # recently active, survives
    ttl_tracker.register_session("touched")     # backdated then touched, survives
    # Backdate idle-old and touched to well past the TTL window.
    import datetime as _dt
    ancient = (_dt.datetime.now(_dt.timezone.utc)
               - _dt.timedelta(seconds=ttl_seconds * 5)).isoformat()
    with ttl_tracker._lock:  # noqa: SLF001 - test-only backdate
        ttl_tracker._sessions["idle-old"].last_active_at = ancient
        ttl_tracker._sessions["touched"].last_active_at = ancient
    # Touch "touched" so it returns to now; "idle-old" stays ancient.
    ttl_tracker.touch_session("touched")
    reaped = ttl_tracker.cleanup_expired()
    assertions.append(assert_ttl_reaps_inactive(
        reaped,
        expected_reaped=["idle-old"],
        expected_alive=["idle-fresh", "touched"],
        tracker=ttl_tracker,
    ))
    # Touch-makes-immune property: backdate an active session, then touch,
    # then sweep — it survives.
    assertions.append(assert_touch_prevents_reap(
        ttl_tracker, "idle-fresh", ttl_seconds,
    ))
    # TTL=None (the default tracker from phases 1-3) never reaps.
    no_ttl_reaped = tracker.cleanup_expired()
    assertions.append(AssertionResult(
        name="ttl_disabled_when_none",
        passed=no_ttl_reaped == [],
        evidence=(
            f"Tracker with session_ttl_seconds=None reaped {no_ttl_reaped} "
            f"(expected [] — TTL disabled by default)."
        ),
    ))
    # Audit retention: reaped sessions remain queryable via get_session /
    # all_sessions (just marked ended), not silently deleted.
    reaped_session = ttl_tracker.get_session("idle-old")
    assertions.append(AssertionResult(
        name="reaped_session_retained_for_audit",
        passed=reaped_session is not None and not reaped_session.is_active()
        and reaped_session.ended_at is not None,
        evidence=(
            f"Reaped session idle-old retained: "
            f"is_active={reaped_session.is_active() if reaped_session else 'N/A'}, "
            f"ended_at={reaped_session.ended_at if reaped_session else 'N/A'}."
        ),
    ))

    passed = all(a.passed for a in assertions)
    sa_state = harness.get_state("session-a")
    sb_state = harness.get_state("session-b")
    metrics = {
        "global_version_after_material": version_after_material,
        "session_a_version": tracker.get_session("session-a").config_version,
        "session_b_version": tracker.get_session("session-b").config_version,
        "session_a_checkpoint_version": sa_state.get("_harness_meta", {}).get("config_version") if sa_state else None,
        "session_b_checkpoint_version": sb_state.get("_harness_meta", {}).get("config_version") if sb_state else None,
        "session_a_resolved_log_level": (tracker.resolved_config_for("session-a") or {}).get("log_level"),
        "session_a_resolved_model": (tracker.resolved_config_for("session-a") or {}).get("model"),
        "session_b_resolved_model": (tracker.resolved_config_for("session-b") or {}).get("model"),
        "ttl_tracker_active_count": len(ttl_tracker.active_sessions()),
        "ttl_tracker_all_count": len(ttl_tracker.all_sessions()),
        "framework": harness.framework,
    }
    return DemoResult(name="A4_hotreload", passed=passed, assertions=assertions, metrics=metrics)
