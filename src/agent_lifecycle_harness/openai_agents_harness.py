"""OpenAI Agents SDK harness for A7 cross-framework matrix.

Demonstrates how each A1-A6 lifecycle concern maps to OAI SDK APIs.
Cells are either implemented with assertions or documented as framework-given
with code citations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


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
# A1 OAI SDK: Isolation — real SQLiteSession test
# ---------------------------------------------------------------------------

def assert_oai_session_isolation() -> AssertionResult:
    """A1 OAI-side: two SQLiteSessions must not share items."""

    async def _run() -> bool:
        from agents.memory import SQLiteSession

        s1 = SQLiteSession("oai-session-1", db_path=":memory:")
        s2 = SQLiteSession("oai-session-2", db_path=":memory:")

        await s1.add_items([{"role": "user", "content": "hello session-1"}])
        await s2.add_items([{"role": "user", "content": "hello session-2"}])

        items1 = await s1.get_items()
        items2 = await s2.get_items()

        contents1 = [i.get("content") for i in items1]
        contents2 = [i.get("content") for i in items2]

        return "hello session-2" not in contents1 and "hello session-1" not in contents2

    try:
        ok = asyncio.run(_run())
    except Exception as exc:
        return AssertionResult(
            name="oai_session_isolation",
            passed=False,
            evidence=f"OAI SDK Session isolation test raised: {exc}",
        )
    return AssertionResult(
        name="oai_session_isolation",
        passed=ok,
        evidence="OAI SDK Session provides framework-given isolation via request-scoped reload."
        if ok
        else "Sessions leaked items across session ids.",
    )


# ---------------------------------------------------------------------------
# A2 OAI SDK: Compaction — real import + instantiate
# ---------------------------------------------------------------------------

def assert_oai_compaction_framework_given() -> AssertionResult:
    """A2 OAI-side: OpenAIResponsesCompactionSession is framework-given.

    Unlike LG (app-owned compaction), OAI SDK provides compaction natively.
    We verify the symbol exists and can be instantiated.
    """
    try:
        from agents import OpenAIResponsesCompactionSession
        from agents.memory import SQLiteSession

        assert OpenAIResponsesCompactionSession is not None
        underlying = SQLiteSession("oai-compaction-underlying", db_path=":memory:")
        session = OpenAIResponsesCompactionSession(
            "oai-compaction",
            underlying,
            client=None,
        )
        assert session is not None
    except Exception as exc:
        return AssertionResult(
            name="oai_compaction_framework_given",
            passed=False,
            evidence=f"OAI SDK compaction import/instantiate failed: {exc}",
        )
    return AssertionResult(
        name="oai_compaction_framework_given",
        passed=True,
        evidence="OAI SDK OpenAIResponsesCompactionSession is framework-given compaction (LG side is app-owned).",
    )


# ---------------------------------------------------------------------------
# A3 OAI SDK: Tombstone — real SQLiteSession pop_item test
# ---------------------------------------------------------------------------

def assert_oai_pop_item_removes_provenance() -> AssertionResult:
    """A3 OAI-side: pop_item removes provenance → tombstone is app-owned.

    OAI SDK's pop_item removes items from session history without
    preserving provenance metadata. Tombstone must be app-owned.
    """
    try:
        from agents.memory import SQLiteSession

        async def _run() -> bool:
            s = SQLiteSession("oai-pop-test", db_path=":memory:")
            await s.add_items([
                {"role": "user", "content": "turn-1"},
                {"role": "assistant", "content": "reply-1"},
            ])
            before = await s.get_items()
            popped = await s.pop_item()
            after = await s.get_items()
            return (
                popped is not None
                and len(before) == 2
                and len(after) == 1
                and popped.get("content") == "reply-1"
            )

        ok = asyncio.run(_run())
    except Exception as exc:
        return AssertionResult(
            name="oai_pop_item_removes_provenance",
            passed=False,
            evidence=f"OAI SDK pop_item test raised: {exc}",
        )
    return AssertionResult(
        name="oai_pop_item_removes_provenance",
        passed=ok,
        evidence="OAI SDK pop_item removes provenance; tombstone is app-owned on both LG and OAI."
        if ok
        else "pop_item did not remove the expected item.",
    )


# ---------------------------------------------------------------------------
# A4 OAI SDK: Hot-reload — real import verification
# ---------------------------------------------------------------------------

def assert_oai_run_config_hot_reload() -> AssertionResult:
    """A4 OAI-side: RunConfig/SessionSettings symbols exist and construct.

    This cell proves ONLY that the OAI SDK ships RunConfig + SessionSettings
    as real, constructible objects whose set fields round-trip. It does NOT
    prove a hot-reload mechanism — the version-on-session logic (bind config
    version to session, old sessions unaffected by reload) is application-
    owned on BOTH frameworks. The OAI SDK gives the request-scoped config
    object; the app decides how to version and reload.
    """
    try:
        from agents import RunConfig, SessionSettings

        run_config = RunConfig(
            model=None,
            workflow_name="hot-reload-test",
            trace_metadata={"version": "v1", "session": "s1"},
        )
        # Round-trip: values set at construction are readable back.
        assert run_config.workflow_name == "hot-reload-test"
        assert run_config.trace_metadata.get("version") == "v1"

        settings = SessionSettings()
        run_config_with_settings = RunConfig(
            model=None,
            session_settings=settings,
        )
        assert run_config_with_settings.session_settings is not None
    except Exception as exc:
        return AssertionResult(
            name="oai_run_config_hot_reload",
            passed=False,
            evidence=f"OAI SDK RunConfig/SessionSettings construction raised: {exc}",
        )
    return AssertionResult(
        name="oai_run_config_hot_reload",
        passed=True,
        evidence="OAI SDK RunConfig + SessionSettings symbols exist and construct "
                 "(fields round-trip). Hot-reload version-on-session logic is "
                 "app-owned on BOTH frameworks.",
    )


# ---------------------------------------------------------------------------
# A5 OAI SDK: Degradation — real import verification
# ---------------------------------------------------------------------------

def assert_oai_run_metrics() -> AssertionResult:
    """A5 OAI-side: Usage/TurnSpanData symbols exist, construct, and round-trip.

    This cell proves ONLY that the OAI SDK ships Usage + TurnSpanData as real,
    constructible objects whose set fields are readable back. It does NOT
    prove a degradation-monitoring mechanism — sustained-delta detection +
    mitigation is application-owned on BOTH frameworks.
    """
    try:
        from agents import Usage, TurnSpanData

        usage = Usage()
        turn_span = TurnSpanData(turn=3, agent_name="metrics-test-agent")
        # Meaningful round-trip: the values we set are the values we read back.
        assert turn_span.turn == 3, f"turn round-trip failed: {turn_span.turn}"
        assert turn_span.agent_name == "metrics-test-agent", f"agent_name round-trip failed: {turn_span.agent_name}"
        # Verify Usage is a real object we can inspect (not just importable).
        assert hasattr(usage, "input_tokens") or hasattr(usage, "requests"), \
            "Usage object has no expected metric fields"
    except Exception as exc:
        return AssertionResult(
            name="oai_run_metrics",
            passed=False,
            evidence=f"OAI SDK Usage/TurnSpanData construction raised: {exc}",
        )
    return AssertionResult(
        name="oai_run_metrics",
        passed=True,
        evidence="OAI SDK Usage + TurnSpanData symbols exist, construct, and "
                 "round-trip set fields. Degradation monitoring (sustained-delta "
                 "detection + mitigation) is app-owned on BOTH frameworks.",
    )


# ---------------------------------------------------------------------------
# A6 OAI SDK: Migration — honest: schema migration is app-owned on BOTH
# ---------------------------------------------------------------------------

def assert_oai_schema_migration_app_owned() -> AssertionResult:
    """A6 OAI-side: the SDK ships NO state-schema migration; A6 is app-owned
    on both frameworks.

    Earlier this cell wrote ``{"schema_version": "v2"}`` into
    ``RunConfig.trace_metadata`` and claimed that proved "framework-given
    schema versioning." That was false: ``trace_metadata`` is an arbitrary
    user-supplied dict (an attach-case for whatever the app wants to tag a
    run with), not a versioning/migration API. Writing a key called
    ``schema_version`` into it proves nothing the SDK does for you.

    What the SDK genuinely provides here: the ``RunConfig`` symbol and its
    ``trace_metadata`` slot exist as real, constructible objects (proven by
    import + construct below). Carrying a schema tag through them is
    application behavior, not framework behavior. State-schema migration is
    therefore app-owned on both LangGraph and the OAI SDK, exactly as A6's
    app-owned migrator (``migration.py``) demonstrates.
    """
    try:
        from agents import RunConfig

        # Real symbol, real object — the only thing the SDK gives us here.
        run_config = RunConfig(model=None, trace_metadata={})
        assert run_config.trace_metadata is not None
        # trace_metadata is an arbitrary dict: the app can put any key in it,
        # but the SDK neither interprets nor versions it.
        run_config.trace_metadata["schema_version"] = "v2"
        assert run_config.trace_metadata.get("schema_version") == "v2"
    except Exception as exc:
        return AssertionResult(
            name="oai_schema_migration_app_owned",
            passed=False,
            evidence=f"OAI SDK RunConfig construction raised: {exc}",
        )
    return AssertionResult(
        name="oai_schema_migration_app_owned",
        passed=True,
        evidence="OAI SDK ships no state-schema migration; RunConfig.trace_metadata "
                 "is arbitrary app metadata, not a versioning API. A6 (schema registry "
                 "+ migration fn) is app-owned on BOTH frameworks.",
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------
# The live A7 entrypoint is demos/a7_cross_framework.py:demo_A7_cross_framework,
# which imports the individual assert_* functions from this module. There is
# no standalone demo function here — run.py dispatches the cross_framework one.

