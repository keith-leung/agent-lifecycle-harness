"""A2 acceptance — fixed fixture, quantified non-determinism, three runs.

Loads a fixed 40-turn conversation from a committed fixture (no
re-generation), runs each compaction strategy 3 times, and reports
min/median/max of the final payload tokens. The coherence probe runs 5
times per strategy and reports pass rate.

Fixed assertions (definitions corrected from prior round):
  1. 40% < reduction < 85%      (reduction has both bounds)
  2. Structural deletion: turn-15 raw message id `h14` is NOT in the
     final payload's message list (id-based, not string-based)
  3. Information retention: MIDSEG string appears in summary text AND
     coherence probe replies with MIDSEG (pass-rate based)

Real-LLM-only.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent_lifecycle_harness.compaction import (
    AppendCompactionStrategy,
    CompactionStore,
    DigestCompactionStrategy,
    LangmemCompactor,
    NoCompactionStrategy,
)
from agent_lifecycle_harness.config import is_mock_mode, load_config
from agent_lifecycle_harness.llm import build_chat_model, tokenizer_kind_for

MIDSEG_MARKER = "MIDSEG_7f3a91"
MIDSEG_TURN = 15
REFERENT_TURN = 30
PRESERVE_TAIL_TURNS = 3
REDUCTION_MIN = 40
REDUCTION_MAX = 85

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "a2_conversation_40turns.json"
EXPECTED_MESSAGES_SHA = "311ab2a7ead91207add840a4a4bffa98b5a99c2658777bca9fe54ff9d9164453"

STRATEGY_RUNS = 3       # per strategy, for tokens_after min/median/max
COHERENCE_PROBES = 5    # per strategy, for pass rate


# ============================================================================
# Fixture loading.
# ============================================================================

def _load_fixture() -> tuple[list[BaseMessage], dict[str, Any]]:
    """Load the fixed conversation. Verifies sha256 of the messages block
    (recomputed from the file, NOT trusted from the fixture's own meta)
    and aborts on mismatch.
    """
    if not FIXTURE_PATH.exists():
        raise SystemExit(f"FIXTURE MISSING: {FIXTURE_PATH}")
    raw_bytes = FIXTURE_PATH.read_bytes()
    data = json.loads(raw_bytes.decode("utf-8"))
    meta = data.get("meta", {})
    msgs_raw = data.get("messages", [])

    # Recompute the fingerprint from the messages block. Serialization must
    # be canonical: sort_keys + no whitespace + UTF-8. The fixture's own
    # meta.messages_sha256 is NOT trusted here — the verification is only
    # meaningful if it's computed independently from the data.
    canonical = json.dumps(msgs_raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    computed_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    print(f"[fixture] path: {FIXTURE_PATH.name}")
    print(f"[fixture] computed messages sha256: {computed_sha}")
    print(f"[fixture] expected (spec constant): {EXPECTED_MESSAGES_SHA}")

    if computed_sha != EXPECTED_MESSAGES_SHA:
        print(
            "[fixture] ABORT: computed messages sha256 does not match the "
            "spec constant. The fixture has been modified or is wrong."
        )
        raise SystemExit(3)

    print("[fixture] sha256 OK")
    msgs: list[BaseMessage] = []
    for m in msgs_raw:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"], id=m["id"]))
        else:
            msgs.append(AIMessage(content=m["content"], id=m["id"]))
    return msgs, meta


# ============================================================================
# Tail-preserving wrapper (exempt mechanism — no summary-text duplication).
# ============================================================================

class _TailPreservingStrategy:
    """Guarantees the last N raw turns survive in the payload verbatim,
    WITHOUT duplication, by passing their ids as ``exempt_message_ids``
    to the inner strategy's compactor.
    """

    def __init__(self, inner: Any, *, preserve_tail_turns: int, token_counter: Any) -> None:
        self.inner = inner
        self.preserve_tail_turns = preserve_tail_turns
        self.token_counter = token_counter
        self.name = getattr(inner, "name", "tail")
        self.last_outcome: dict[str, Any] = {}

    def build_replay_context(self, thread_id: str, full_history: list[BaseMessage]):
        tail_size = self.preserve_tail_turns * 2
        exempt_ids: set[str] = set()
        if len(full_history) > tail_size:
            exempt_ids = {
                getattr(m, "id", None)
                for m in full_history[-tail_size:]
                if getattr(m, "id", None) is not None
            }
        compactor = getattr(self.inner, "compactor", None)
        if compactor is not None and exempt_ids:
            compactor._pending_exempt = exempt_ids  # type: ignore[attr-defined]
        replay = self.inner.build_replay_context(thread_id, full_history)
        if compactor is not None:
            compactor._pending_exempt = None  # type: ignore[attr-defined]
        from agent_lifecycle_harness.compaction import ReplayOutcome
        merged = ReplayOutcome(
            messages=list(replay.messages),
            dropped_raw_count=replay.dropped_raw_count,
            digest_messages_added=replay.digest_messages_added,
            tokens_before=self.token_counter(full_history),
            tokens_after=self.token_counter(replay.messages),
            strategy=self.name,
            digest=replay.digest,
            stable_prefix_tokens=0,
        )
        self.last_outcome[thread_id] = merged
        return merged


def _make_strategy(name: str, store: CompactionStore, chat_model: Any) -> _TailPreservingStrategy:
    compactor = LangmemCompactor(
        store, chat_model,
        max_tokens=6000, max_tokens_before_summary=3000, max_summary_tokens=500,
    )
    if name == "none":
        inner: Any = NoCompactionStrategy(token_counter=chat_model.get_num_tokens_from_messages)
    elif name == "merge":
        inner = DigestCompactionStrategy(compactor)
    elif name == "append":
        inner = AppendCompactionStrategy(compactor)
    else:
        raise ValueError(name)
    return _TailPreservingStrategy(
        inner, preserve_tail_turns=PRESERVE_TAIL_TURNS,
        token_counter=chat_model.get_num_tokens_from_messages,
    )


# ============================================================================
# Run one strategy turn-by-turn over the fixture.
# ============================================================================

@dataclass
class StrategyRun:
    strategy: str
    rows: list[dict] = field(default_factory=list)
    final_payload: list[BaseMessage] = field(default_factory=list)
    final_tokens: int = 0
    summary_tokens_final: int = 0
    n_llm_calls_total: int = 0


def _run_strategy_once(
    strategy_name: str,
    chat_model: Any,
    dialogue_n_turns: int,
    seed: list[BaseMessage],
) -> StrategyRun:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = CompactionStore(tmp.name)
    strategy = _make_strategy(strategy_name, store, chat_model)

    run = StrategyRun(strategy=strategy_name)
    history: list[BaseMessage] = []

    for i in range(dialogue_n_turns):
        history.append(seed[i * 2])
        history.append(seed[i * 2 + 1])
        replay = strategy.build_replay_context("acceptance", list(history))
        payload = list(replay.messages)
        tokens_in = chat_model.get_num_tokens_from_messages(payload)
        summary_messages = [
            m for m in payload
            if getattr(m, "id", None) is None or getattr(m, "type", "") == "system"
        ]
        summary_tokens = (
            chat_model.get_num_tokens_from_messages(summary_messages)
            if summary_messages else 0
        )
        run.rows.append({
            "turn": i + 1, "tokens_in": tokens_in, "summary_tokens": summary_tokens,
            "n_llm": replay.digest_messages_added,
        })
        run.n_llm_calls_total += replay.digest_messages_added
        run.final_payload = payload
        run.final_tokens = tokens_in
        run.summary_tokens_final = summary_tokens

    try:
        Path(tmp.name).unlink()
    except OSError:
        pass
    return run


def _payload_text(messages: list[BaseMessage]) -> str:
    return "".join(getattr(m, "content", "") or "" for m in messages)


# ============================================================================
# Main.
# ============================================================================

def main() -> int:
    config = load_config("config.yaml")
    if is_mock_mode(config):
        print("REJECTED: acceptance requires real LLM, not mock mode.")
        return 2

    provider_names = list(config["providers"].keys())
    chat_model = build_chat_model(provider_names[0], "medium", config=config, temperature=0.0)
    model_name = getattr(chat_model, "model_name", "")
    tok_kind = tokenizer_kind_for(model_name)

    print("=" * 80)
    print(f"[setup] model={model_name}")
    print(f"[setup] tokenizer={tok_kind}")
    print(f"[setup] temperature={getattr(chat_model, 'temperature', '?')}")
    print(f"[setup] provider: {provider_names[0]} via gpt-agent.cc gateway")
    print("[setup] determinism: deepseek-v4-flash does NOT document greedy-sampling "
          "determinism; the gpt-agent.cc gateway also does not publish a determinism "
          "guarantee. temperature=0 reduces but does not eliminate output variance.")
    print(f"[setup] strategy_runs={STRATEGY_RUNS} coherence_probes={COHERENCE_PROBES}")
    print(f"[setup] reduction window: {REDUCTION_MIN}% < red < {REDUCTION_MAX}%")
    print("=" * 80)

    seed, meta = _load_fixture()
    n_turns = meta.get("n_turns", len(seed) // 2)
    baseline_final_tokens = chat_model.get_num_tokens_from_messages(seed)
    print(f"[fixture] {len(seed)} messages, {n_turns} turns, "
          f"{baseline_final_tokens} tokens (baseline final payload)")
    print(f"[fixture] MIDSEG seeded at turn {MIDSEG_TURN} (message id h{MIDSEG_TURN-1})")
    print(f"[fixture] referent question at turn {REFERENT_TURN}")

    # First 3 turns verbatim.
    print("\n--- first 3 turns (user side, verbatim) ---")
    for i in range(3):
        hm = seed[i * 2]
        am = seed[i * 2 + 1]
        print(f"\n[turn {i+1} user] {hm.content}")
        print(f"[turn {i+1} asst] {(am.content or '')[:300]}{'...' if len(am.content or '') > 300 else ''}")

    # ---- Per-strategy: STRATEGY_RUNS runs each, collect tokens_after ----
    # Each run is checkpointed to runs/a2_v5/run_<strategy>_<k>.json so a
    # kill mid-suite resumes without re-spending LLM calls. Each run starts
    # from a fresh store (no cross-run state), so re-loading a saved run is
    # equivalent to having just executed it.
    journal_dir = Path("runs/a2_v5")
    journal_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- running each strategy {STRATEGY_RUNS}x (journal: {journal_dir}) ---")
    runs_by_strategy: dict[str, list[StrategyRun]] = {}
    for name in ("none", "append", "merge"):
        runs_by_strategy[name] = []
        for k in range(STRATEGY_RUNS):
            jpath = journal_dir / f"run_{name}_{k+1}.json"
            if jpath.exists():
                try:
                    d = json.loads(jpath.read_text(encoding="utf-8"))
                    run = StrategyRun(strategy=d["strategy"], final_tokens=d["final_tokens"],
                                      summary_tokens_final=d["summary_tokens"],
                                      n_llm_calls_total=d["n_llm_calls_total"])
                    run.final_payload = [
                        HumanMessage(content=m["content"], id=m["id"]) if m["role"] == "user"
                        else AIMessage(content=m["content"], id=m["id"])
                        for m in d["final_payload"]
                    ]
                    runs_by_strategy[name].append(run)
                    print(f"  [{name} run {k+1}/{STRATEGY_RUNS}] CACHED final_tokens={run.final_tokens}", flush=True)
                    continue
                except Exception as exc:
                    print(f"  [{name} run {k+1}/{STRATEGY_RUNS}] cache corrupt ({exc}); re-running", flush=True)
            print(f"  [{name} run {k+1}/{STRATEGY_RUNS}]", flush=True)
            run = _run_strategy_once(name, chat_model, n_turns, seed)
            runs_by_strategy[name].append(run)
            print(f"    final_tokens={run.final_tokens} summary_tokens={run.summary_tokens_final} "
                  f"n_llm_total={run.n_llm_calls_total}", flush=True)
            jpath.write_text(json.dumps({
                "strategy": run.strategy, "final_tokens": run.final_tokens,
                "summary_tokens": run.summary_tokens_final,
                "n_llm_calls_total": run.n_llm_calls_total,
                "final_payload": [
                    {"role": "user" if isinstance(m, HumanMessage) else "assistant",
                     "id": getattr(m, "id", None), "content": getattr(m, "content", "")}
                    for m in run.final_payload
                ],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- tokens_after min/median/max table ----
    print("\n" + "=" * 80)
    print(f"{'strategy':<10} {'min':>8} {'median':>8} {'max':>8} {'spread%':>8} {'reduction%':>12}")
    print("-" * 80)
    reduction_table: dict[str, float] = {}
    for name in ("none", "append", "merge"):
        finals = [r.final_tokens for r in runs_by_strategy[name]]
        med = statistics.median(finals)
        spread = 100 * (max(finals) - min(finals)) / med if med else 0
        red = 100 * (1 - med / baseline_final_tokens)
        reduction_table[name] = red
        flag = " *** non-reproducible" if spread > 20 else ""
        print(f"{name:<10} {min(finals):>8} {med:>8.0f} {max(finals):>8} "
              f"{spread:>7.1f}% {red:>11.1f}%{flag}")
    print("=" * 80)

    # ---- Assertion 1: reduction window (median-based) ----
    print("\n--- assertion 1: reduction window (median-based) ---")
    for name in ("append", "merge"):
        red = reduction_table[name]
        ok = REDUCTION_MIN < red < REDUCTION_MAX
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} reduction {red:.1f}% in ({REDUCTION_MIN}, {REDUCTION_MAX})")

    # ---- Assertion 2: structural deletion (h14 not in final payload) ----
    # Check all runs of append and merge.
    print("\n--- assertion 2: structural deletion (h14 NOT in final payload) ---")
    print(f"  baseline (none) keeps h14: "
          f"{any(getattr(m, 'id', None) == 'h14' for m in runs_by_strategy['none'][0].final_payload)}")
    for name in ("append", "merge"):
        results = []
        for k, run in enumerate(runs_by_strategy[name]):
            ids = {getattr(m, "id", None) for m in run.final_payload}
            h14_absent = "h14" not in ids
            results.append(h14_absent)
            print(f"  [{name} run {k+1}] h14 absent: {h14_absent} "
                  f"(payload ids sample: {sorted(i for i in ids if i)[:8]})")
        all_pass = all(results)
        print(f"  [{'PASS' if all_pass else 'FAIL'}] {name}: h14 absent in all {STRATEGY_RUNS} runs")

    # ---- Information retention: MIDSEG in summary text ----
    print("\n--- information retention: MIDSEG in summary text ---")
    for name in ("append", "merge"):
        results = []
        for k, run in enumerate(runs_by_strategy[name]):
            summ_text = "".join(
                getattr(m, "content", "") or ""
                for m in run.final_payload
                if getattr(m, "id", None) is None or getattr(m, "type", "") == "system"
            )
            has = MIDSEG_MARKER in summ_text
            results.append(has)
            print(f"  [{name} run {k+1}] MIDSEG in summary text: {has}")
        print(f"  [{'PASS' if all(results) else 'FAIL'}] {name}: MIDSEG retained in summary in all runs")

    # ---- Assertion 3: coherence probe (5x per strategy) ----
    print(f"\n--- assertion 3: coherence probe ({COHERENCE_PROBES}x per strategy) ---")
    referent_question = seed[(REFERENT_TURN - 1) * 2].content
    print(f"  probe question: {referent_question[:120]}")
    coherence_results: dict[str, int] = {}
    for name in ("append", "merge"):
        base_payload = runs_by_strategy[name][-1].final_payload
        passes = 0
        for k in range(COHERENCE_PROBES):
            jpath = journal_dir / f"probe_{name}_{k+1}.json"
            if jpath.exists():
                try:
                    d = json.loads(jpath.read_text(encoding="utf-8"))
                    ok = bool(d["pass"])
                    snippet = d.get("snippet", "")
                    passes += int(ok)
                    print(f"  [{name} probe {k+1}] CACHED {'PASS' if ok else 'FAIL'}: {snippet!r}")
                    continue
                except Exception:
                    pass
            prompt = list(base_payload) + [HumanMessage(content=referent_question, id="probe")]
            try:
                reply = chat_model.bind(timeout=120).invoke(prompt)
                reply_text = getattr(reply, "content", "") or ""
            except Exception as exc:
                reply_text = f"[err: {type(exc).__name__}]"
            ok = MIDSEG_MARKER in reply_text
            passes += int(ok)
            snippet = reply_text[:80].replace("\n", " ")
            print(f"  [{name} probe {k+1}] {'PASS' if ok else 'FAIL'}: {snippet!r}", flush=True)
            jpath.write_text(json.dumps({"pass": ok, "snippet": snippet}, ensure_ascii=False), encoding="utf-8")
        coherence_results[name] = passes
        print(f"  [{name}] coherence pass rate: {passes}/{COHERENCE_PROBES}")

    # ---- Summary-only coherence probe (5x per strategy) ----
    # Same question, but ONLY the summary message(s) as context — every raw
    # message (id is not None) is stripped. This tests whether the summary
    # itself retained the MIDSEG fact, as opposed to the full-payload probe
    # which can answer from any raw message that happens to mention it.
    print(f"\n--- summary-only coherence probe ({COHERENCE_PROBES}x per strategy) ---")
    print(f"  probe question: {referent_question[:120]}")
    print(f"  context: [summary messages only, raw messages stripped]")
    summary_coherence_results: dict[str, int] = {}
    for name in ("append", "merge"):
        base_payload = runs_by_strategy[name][-1].final_payload
        summary_only = [
            m for m in base_payload
            if getattr(m, "id", None) is None or getattr(m, "type", "") == "system"
        ]
        if not summary_only:
            print(f"  [{name}] no summary messages in payload; skipping (0/{COHERENCE_PROBES})")
            summary_coherence_results[name] = 0
            continue
        summary_token_count = chat_model.get_num_tokens_from_messages(summary_only)
        print(f"  [{name}] summary-only context: {len(summary_only)} message(s), "
              f"{summary_token_count} tokens")
        passes = 0
        for k in range(COHERENCE_PROBES):
            jpath = journal_dir / f"probe_summary_{name}_{k+1}.json"
            if jpath.exists():
                try:
                    d = json.loads(jpath.read_text(encoding="utf-8"))
                    ok = bool(d["pass"])
                    snippet = d.get("snippet", "")
                    passes += int(ok)
                    print(f"  [{name} summary-probe {k+1}] CACHED {'PASS' if ok else 'FAIL'}: {snippet!r}")
                    continue
                except Exception:
                    pass
            prompt = list(summary_only) + [HumanMessage(content=referent_question, id="probe")]
            try:
                reply = chat_model.bind(timeout=120).invoke(prompt)
                reply_text = getattr(reply, "content", "") or ""
            except Exception as exc:
                reply_text = f"[err: {type(exc).__name__}]"
            ok = MIDSEG_MARKER in reply_text
            passes += int(ok)
            snippet = reply_text[:80].replace("\n", " ")
            print(f"  [{name} summary-probe {k+1}] {'PASS' if ok else 'FAIL'}: {snippet!r}", flush=True)
            jpath.write_text(json.dumps({"pass": ok, "snippet": snippet}, ensure_ascii=False), encoding="utf-8")
        summary_coherence_results[name] = passes
        print(f"  [{name}] summary-only coherence pass rate: {passes}/{COHERENCE_PROBES}")

    # ---- merge final payload composition (turn-1-of-3 run, for diagnosis) ----
    print("\n--- merge final payload composition (run 1) ---")
    merge_run1 = runs_by_strategy["merge"][0]
    for m in merge_run1.final_payload:
        mid = getattr(m, "id", None)
        typ = getattr(m, "type", "?")
        c = getattr(m, "content", "") or ""
        n_tok = chat_model.get_num_tokens_from_messages([m])
        has_midseg = MIDSEG_MARKER in c
        print(f"  id={mid!r:8} type={typ:9} tokens={n_tok:5} has_MIDSEG={has_midseg} content_head={c[:60]!r}")

    print("\n--- summary ---")
    print(f"  append tokens_after min/med/max: "
          f"{min(r.final_tokens for r in runs_by_strategy['append'])}/"
          f"{statistics.median(r.final_tokens for r in runs_by_strategy['append']):.0f}/"
          f"{max(r.final_tokens for r in runs_by_strategy['append'])}")
    print(f"  merge  tokens_after min/med/max: "
          f"{min(r.final_tokens for r in runs_by_strategy['merge'])}/"
          f"{statistics.median(r.final_tokens for r in runs_by_strategy['merge']):.0f}/"
          f"{max(r.final_tokens for r in runs_by_strategy['merge'])}")
    print(f"  append coherence (full payload): {coherence_results['append']}/{COHERENCE_PROBES}")
    print(f"  merge  coherence (full payload): {coherence_results['merge']}/{COHERENCE_PROBES}")
    print(f"  append coherence (summary only): {summary_coherence_results['append']}/{COHERENCE_PROBES}")
    print(f"  merge  coherence (summary only): {summary_coherence_results['merge']}/{COHERENCE_PROBES}")

    print("\n--- probe comparison table ---")
    print(f"{'strategy':<10} {'full-payload':>14} {'summary-only':>14}")
    print("-" * 40)
    for name in ("append", "merge"):
        print(f"{name:<10} {coherence_results[name]:>13}/{COHERENCE_PROBES} "
              f"{summary_coherence_results[name]:>13}/{COHERENCE_PROBES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
