"""OpenAI Agents SDK harness for A7 cross-framework matrix.

Each A1-A6 lifecycle concern maps to an OAI SDK cell. Cells fall into
exactly two categories — no third "import-only" middle ground:

  * **BehaviorCell** — runs against the installed SDK and asserts an
    OBSERVABLE behavior (isolation, pop-removes-item, usage-accumulates).
    Each carries a mutation check: breaking the asserted behavior makes
    the cell FAIL.
  * **DocCell** — the SDK symbol exists but the asserted behavior cannot
    be reproduced locally (needs a real OpenAI client, or the behavior is
    application-owned on BOTH frameworks so there's nothing for the SDK
    to be tested for). DocCells are honestly marked ``executed=False``
    with a verified citation (site-packages file:line), NOT returned as
    passing assertions. ``import X; assert X is not None`` is neither —
    it has been removed.
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


@dataclass
class BehaviorCell:
    """A cell that ran an observable behavior assertion against the SDK.

    ``passed`` reflects whether the behavior held. ``mutation_verified``
    is set True once the corresponding mutation proof (break the behavior
    → cell FAILs) has been run separately.
    """
    name: str
    passed: bool
    evidence: str
    mutation_verified: bool = False
    kind: str = "behavior"


@dataclass
class DocCell:
    """A cell honestly marked as documented, not executed.

    The SDK ships the referenced symbol (citation points into
    ``site-packages``), but the asserted behavior cannot be reproduced
    locally. This is NOT a passing test — it's a verified reference.
    Callers must not treat ``DocCell`` as ``passed``.
    """
    name: str
    citation: str  # site-packages file:line or official doc URL, verified
    note: str       # why this isn't a behavior test
    kind: str = "documented"
    executed: bool = False
    passed: bool = False  # explicitly False; never treated as a pass


# ---------------------------------------------------------------------------
# A1 OAI SDK: Isolation — real SQLiteSession behavior test (BehaviorCell)
# ---------------------------------------------------------------------------

async def _oai_session_isolation_behavior(shared_db: bool) -> dict[str, Any]:
    """Run two SQLiteSessions and check whether items leak across them.

    ``shared_db`` controls the mutation: when False (the real check) each
    session gets its own :memory: db and isolation holds; when True (the
    mutation) both sessions back onto the SAME shared file with the SAME
    session_id — so the second session's reads see the first session's
    rows. The mutation must break isolation, proving the cell tests
    per-session isolation rather than symbol existence.
    """
    from agents.memory import SQLiteSession
    import tempfile, os
    if shared_db:
        # Mutation: same db file + same session_id → both sessions read the
        # same row set. SQLiteSession scopes rows by session_id, so sharing
        # both the file and the id makes them see each other's items.
        shared_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        shared_file.close()
        try:
            s1 = SQLiteSession("oai-shared", db_path=shared_file.name)
            s2 = SQLiteSession("oai-shared", db_path=shared_file.name)
        finally:
            pass
    else:
        s1 = SQLiteSession("oai-session-1", db_path=":memory:")
        s2 = SQLiteSession("oai-session-2", db_path=":memory:")
        shared_file = None
    try:
        await s1.add_items([{"role": "user", "content": "hello session-1"}])
        await s2.add_items([{"role": "user", "content": "hello session-2"}])
        items1 = await s1.get_items()
        items2 = await s2.get_items()
        c1 = [i.get("content") for i in items1]
        c2 = [i.get("content") for i in items2]
        leak_into_1 = "hello session-2" in c1
        leak_into_2 = "hello session-1" in c2
        return {
            "isolated": not (leak_into_1 or leak_into_2),
            "leak_into_1": leak_into_1, "leak_into_2": leak_into_2,
            "s1_contents": c1, "s2_contents": c2,
        }
    finally:
        if shared_file is not None:
            try: os.unlink(shared_file.name)
            except OSError: pass


def assert_oai_session_isolation() -> BehaviorCell:
    """A1 OAI-side (behavior): two SQLiteSessions must not share items.

    Citation: agents/memory/sqlite_session.py (SQLiteSession.add_items at
    line 17; per-session row scoping via session_id arg).
    """
    try:
        result = asyncio.run(_oai_session_isolation_behavior(shared_db=False))
    except Exception as exc:
        return BehaviorCell(
            name="oai_session_isolation", passed=False,
            evidence=f"raised: {type(exc).__name__}: {exc}",
        )
    return BehaviorCell(
        name="oai_session_isolation",
        passed=result["isolated"],
        evidence=(
            f"s1={result['s1_contents']}, s2={result['s2_contents']}; "
            f"isolated={result['isolated']}"
        ),
    )


# ---------------------------------------------------------------------------
# A2 OAI SDK: Compaction — DocCell (run_compaction needs a real OpenAI client)
# ---------------------------------------------------------------------------

def doc_oai_compaction_framework_given() -> DocCell:
    """A2 OAI-side (documented): OpenAIResponsesCompactionSession is the SDK's
    framework-given compaction primitive. We cannot run ``run_compaction``
    locally because it calls the real OpenAI Responses API (it accepts a
    ``client`` argument and invokes it to summarize). Behavior of the
    compaction policy itself is therefore NOT executed here; the cell is an
    honest citation. LangGraph-side compaction is app-owned (see A2).
    """
    return DocCell(
        name="oai_compaction_framework_given",
        citation=(
            "site-packages/agents/memory/openai_responses_compaction_session.py:78 "
            "(class OpenAIResponsesCompactionSession; run_compaction method)"
        ),
        note=(
            "run_compaction(client=...) requires a real OpenAI Responses client; "
            "compaction-policy behavior cannot be reproduced locally. Symbol "
            "construction is not a behavior test."
        ),
    )


# ---------------------------------------------------------------------------
# A3 OAI SDK: Tombstone — real pop_item behavior test (BehaviorCell)
# ---------------------------------------------------------------------------

async def _oai_pop_item_behavior(broken_pop: bool) -> dict[str, Any]:
    """Add two items, pop the last, verify it's gone.

    ``broken_pop`` is the mutation: when True, we monkey-patch pop_item to
    a no-op so the item is NOT removed — the assertion must then fail,
    proving the cell really checks the deletion behavior.
    """
    from agents.memory import SQLiteSession
    s = SQLiteSession("oai-pop-test", db_path=":memory:")
    if broken_pop:
        async def _noop_pop():
            return None
        s.pop_item = _noop_pop  # mutation: pop does nothing
    await s.add_items([
        {"role": "user", "content": "turn-1"},
        {"role": "assistant", "content": "reply-1"},
    ])
    before = await s.get_items()
    popped = await s.pop_item()
    after = await s.get_items()
    return {
        "before_len": len(before),
        "after_len": len(after),
        "popped_content": (popped or {}).get("content"),
        "removed_correctly": (
            popped is not None
            and len(before) == 2
            and len(after) == 1
            and popped.get("content") == "reply-1"
        ),
    }


def assert_oai_pop_item_removes_provenance() -> BehaviorCell:
    """A3 OAI-side (behavior): pop_item removes the most recent item.

    Citation: agents/memory/sqlite_session.py — SQLiteSession.pop_item uses
    DELETE...RETURNING to atomically remove and return the most recent item.
    """
    try:
        result = asyncio.run(_oai_pop_item_behavior(broken_pop=False))
    except Exception as exc:
        return BehaviorCell(
            name="oai_pop_item_removes_provenance", passed=False,
            evidence=f"raised: {type(exc).__name__}: {exc}",
        )
    return BehaviorCell(
        name="oai_pop_item_removes_provenance",
        passed=result["removed_correctly"],
        evidence=(
            f"before={result['before_len']}, after={result['after_len']}, "
            f"popped={result['popped_content']!r}"
        ),
    )


# ---------------------------------------------------------------------------
# A4 OAI SDK: Hot-reload — DocCell (version-on-session is app-owned on BOTH)
# ---------------------------------------------------------------------------

def doc_oai_run_config_hot_reload() -> DocCell:
    """A4 OAI-side (documented): RunConfig is a request-scoped config object.

    Hot-reload's defining behavior — old sessions pinned to the config
    version they started with, new sessions pick up the current version —
    is APPLICATION-OWNED on both LangGraph (A4's ConfigVersionTracker) and
    OAI SDK. RunConfig has no version field, no session-version binding.
    Constructing it and round-tripping a field tests Python dataclass
    assignment, not hot-reload. So this is a citation, not a test.
    """
    return DocCell(
        name="oai_run_config_hot_reload",
        citation=(
            "site-packages/agents/run_config.py:211 (class RunConfig; "
            "no version/session-binding field — fields: model, workflow_name, "
            "trace_metadata, session_settings, ...)"
        ),
        note=(
            "RunConfig is request-scoped config; version-on-session pinning is "
            "app-owned on both frameworks (A4 ConfigVersionTracker). Constructing "
            "a dataclass and reading back a field is not a hot-reload test."
        ),
    )


# ---------------------------------------------------------------------------
# A5 OAI SDK: Usage accumulation — real behavior test (BehaviorCell)
# ---------------------------------------------------------------------------

async def _oai_usage_accumulates_behavior(broken_add: bool) -> dict[str, Any]:
    """Usage.add must accumulate token counts across calls.

    Mutation: when ``broken_add`` is True, monkey-patch ``add`` to a no-op
    so the totals stay 0 — the assertion must fail, proving the cell tests
    accumulation rather than symbol existence.

    Note: ``Usage.add`` takes another ``Usage`` object (not keyword args);
    ``add(self, other: Usage)`` at agents/usage.py:102.
    """
    from agents import Usage
    u = Usage()
    if broken_add:
        def _noop_add(*a, **kw):
            pass
        u.add = _noop_add  # mutation
    # add() takes another Usage; construct two with the fields we want to sum.
    u.add(Usage(input_tokens=100, output_tokens=50))
    u.add(Usage(input_tokens=200, output_tokens=150))
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "accumulated_correctly": u.input_tokens == 300 and u.output_tokens == 200,
    }


def assert_oai_usage_accumulates() -> BehaviorCell:
    """A5 OAI-side (behavior): Usage.add accumulates token counts.

    Citation: agents/usage.py:102 (class Usage, field input_tokens /
    output_tokens, method add).
    """
    try:
        result = asyncio.run(_oai_usage_accumulates_behavior(broken_add=False))
    except Exception as exc:
        return BehaviorCell(
            name="oai_usage_accumulates", passed=False,
            evidence=f"raised: {type(exc).__name__}: {exc}",
        )
    return BehaviorCell(
        name="oai_usage_accumulates",
        passed=result["accumulated_correctly"],
        evidence=(
            f"after two add() calls: input_tokens={result['input_tokens']} "
            f"(expect 300), output_tokens={result['output_tokens']} (expect 200)"
        ),
    )


# ---------------------------------------------------------------------------
# A5b OAI SDK: TurnSpanData export — real behavior test (BehaviorCell)
# ---------------------------------------------------------------------------

def _oai_turn_span_export_behavior(broken_export: bool) -> dict[str, Any]:
    """export() must serialize the span's fields into its output structure.

    Mutation: when ``broken_export`` is True, monkey-patch ``export`` to
    return a dict missing the ``data`` field — the assertion must fail,
    proving the cell checks the serialized payload, not symbol existence
    or __init__ round-trip.
    """
    from agents import TurnSpanData
    span = TurnSpanData(turn=3, agent_name="metrics-test-agent")
    if broken_export:
        def _broken():
            return {"type": "custom", "name": "turn"}  # missing data
        span.export = _broken
    exported = span.export()
    data = exported.get("data", {}) if isinstance(exported, dict) else {}
    return {
        "exported": exported,
        "data_turn": data.get("turn"),
        "data_agent_name": data.get("agent_name"),
        "serialized_correctly": data.get("turn") == 3 and data.get("agent_name") == "metrics-test-agent",
    }


def assert_oai_turn_span_exports() -> BehaviorCell:
    """A5b OAI-side (behavior): export() serializes the span's fields into
    its output dict under the ``data`` key.

    Citation: agents/tracing/span_data.py:119 (TurnSpanData.export returns
    ``{"type":"custom","name":"turn","data":{...}}``).
    """
    try:
        result = _oai_turn_span_export_behavior(broken_export=False)
    except Exception as exc:
        return BehaviorCell(
            name="oai_turn_span_exports", passed=False,
            evidence=f"raised: {type(exc).__name__}: {exc}",
        )
    return BehaviorCell(
        name="oai_turn_span_exports",
        passed=result["serialized_correctly"],
        evidence=(
            f"export()={result['exported']}; "
            f"data.turn={result['data_turn']}, data.agent_name={result['data_agent_name']!r}"
        ),
    )


# ---------------------------------------------------------------------------
# A6 OAI SDK: Schema migration — DocCell (app-owned on BOTH frameworks)
# ---------------------------------------------------------------------------

def doc_oai_schema_migration_app_owned() -> DocCell:
    """A6 OAI-side (documented): the SDK ships no state-schema migration API.

    RunConfig.trace_metadata is an arbitrary user-supplied dict — writing
    a key called ``schema_version`` into it doesn't make the SDK version
    anything. State-schema migration is app-owned on both LangGraph (A6's
    SchemaRegistry + migration fn) and OAI SDK.
    """
    return DocCell(
        name="oai_schema_migration_app_owned",
        citation=(
            "site-packages/agents/run_config.py:211 — RunConfig.trace_metadata "
            "field (arbitrary dict, no schema-versioning semantics)"
        ),
        note=(
            "SDK has no migration API; trace_metadata is unstructured app metadata. "
            "A6 is app-owned on both frameworks. Construction + dict assignment is "
            "not a migration test."
        ),
    )


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------
# The live A7 entrypoint is demos/a7_cross_framework.py:demo_A7_cross_framework,
# which imports the individual assert_* functions from this module. There is
# no standalone demo function here — run.py dispatches the cross_framework one.

