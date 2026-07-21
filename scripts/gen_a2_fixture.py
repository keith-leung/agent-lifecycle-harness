"""Generate the A2 conversation fixture — ONE TIME, then commit and never regenerate.

Run manually from a terminal (not from a background agent task):

    conda activate agent-lifecycle-harness
    cd repos/A_agent-lifecycle-harness
    python -u scripts/gen_a2_fixture.py

Writes:  tests/fixtures/a2_conversation_40turns.json
Resumes: runs/a2_fixture_checkpoint.json  (checkpointed after EVERY turn)

Why this exists
---------------
The conversation is a FIXTURE, not the system under test. Regenerating it on
every acceptance run (a) costs 40 accumulating LLM calls, (b) makes numbers
across rounds incomparable because the ruler changes each time. Generate once,
commit, and every future acceptance run loads it in milliseconds.

Contract
--------
Each turn sends the FULL accumulated history, so the model can actually answer
the back-references written into the dialogue. Turn 15 seeds a transaction id;
turn 30 asks the model to recall it. If turn 30's reply does not contain the
id, the history was not accumulated and the fixture is rejected.

Safe to Ctrl-C: progress is checkpointed after every turn; re-running resumes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent_lifecycle_harness.config import is_mock_mode, load_config
from agent_lifecycle_harness.llm import build_chat_model, tokenizer_kind_for

MIDSEG_MARKER = "MIDSEG_7f3a91"
N_TURNS = 40
MIDSEG_TURN = 15      # seeds the id
REFERENT_TURN = 30    # asks the model to recall it

PER_CALL_TIMEOUT = 240   # late turns send ~46k tokens; be generous
PROBE_ABORT_SECONDS = 15  # if a trivial call takes longer, something is wrong

FIXTURE_PATH = Path("tests/fixtures/a2_conversation_40turns.json")
CHECKPOINT_PATH = Path("runs/a2_fixture_checkpoint.json")


# ---------------------------------------------------------------------------
# The dialogue. 40 turns building a ToyDB design doc, with dense back-references.
# Turn 15 seeds the marker; turn 30 quizzes it. Kept verbatim here so this
# script does not depend on a file that is still being edited.
# ---------------------------------------------------------------------------

DIALOGUE: list[str] = [
    "We're designing a small toy database called ToyDB. Let's start with the storage layer: should we use a B-tree or an LSM-tree for the primary index? Pick one and defend it.",
    "OK, you picked one. Now: how does that choice affect our write path? Walk me through what happens when a single row insert arrives.",
    "Following on from the write path you just described: where does the WAL fit in, and what guarantees does it give us on crash?",
    "Switching to reads: given our storage choice, how would you implement a point lookup, and what's the latency shape?",
    "Tying turns 2 and 4 together: what happens to a point lookup that arrives WHILE the write from turn 2 is mid-flush? Do we read stale or fresh data?",
    "Let's add concurrency control. Given what you said in turn 5 about in-flight writes, should ToyDB use MVCC or strict 2PL? Defend the choice.",
    "Continuing turn 6: if we go MVCC, what does a transaction abort look like mechanically? Walk me through the cleanup.",
    "Pivoting to query parsing: given our storage layout, how should the parser represent a range scan so the executor can use the index from turn 4?",
    "Following turn 8: now add a filter predicate on top of the range scan. Where does predicate pushdown happen, and why?",
    "Connecting turns 5 and 9: if a filter is pushed down but the underlying row is being written (the race from turn 5), whose version does the filter see?",
    "Let's talk schema. For our toy schema, should primary keys be surrogate (auto-increment) or natural? Reference your storage choice from turn 1.",
    "From turn 11: if we pick surrogate keys, what does a secondary index on a natural column cost us at write time, given the write path from turn 2?",
    "Scaling up: at what row count does the storage choice from turn 1 start to hurt, and what's the first symptom we'd observe?",
    "Following turn 13: when that symptom appears, what's the cheapest mitigation that does NOT change the storage engine?",
    f"Operational concern. I'm going to track a specific long-running transaction through the system to test the abort path from turn 7. For this conversation, the transaction id is {MIDSEG_MARKER}. Please acknowledge the id and tell me which component would log it during an abort.",
    f"Connecting turn 15 to turn 7's abort path: if {MIDSEG_MARKER} aborts, exactly which data structures from turn 6's MVCC design get touched?",
    "Now caching. Given the read path from turn 4 and the race from turn 5, what's the safest cache invalidation strategy for ToyDB?",
    "Following turn 17: what's the cache hit ratio at which the cache stops paying for itself, given our write rate from turn 2?",
    "Network layer. If ToyDB accepts writes over the network, what does the WAL-from-turn-3 story look like when the network drops mid-ack?",
    "Tying turns 19 and 17 together: after a network drop, how do we repopulate the cache without serving stale data per turn 5's race?",
    "Replication. Should ToyDB use async or sync replication, given the WAL guarantees from turn 3? Defend it.",
    "From turn 21: if we go async, what's the data-loss window on a primary crash, expressed in terms of the write rate from turn 2?",
    "Following turn 22: how does the secondary's MVCC (turn 6) behave when it receives a batch of async WAL records out of order?",
    "Back to the query side. Given the out-of-order replication from turn 23, can a read-only replica serve the range scan from turn 8 consistently?",
    "If turn 24 says yes, what does that imply about our predicate pushdown from turn 9 — does it need to change?",
    "Transactions spanning replicas: given turns 21-25, can ToyDB offer distributed transactions, or should we explicitly scope transactions to one node?",
    "From turn 26: if transactions are single-node, what does that cost the application layer in terms of the schema choices from turn 11?",
    "Observability. To debug the race from turn 5 in production, what three metrics would you add to ToyDB, and why those three?",
    f"Following turn 28: how would each of those metrics behave during the abort scenario involving {MIDSEG_MARKER} from turn 15?",
    "Quiz time, no preamble: what was the exact transaction id I gave you earlier in this conversation? Reply with just the id.",
    "Good. Now: when that transaction aborts (per turn 16), which of the three metrics from turn 28 would spike first?",
    "Compaction. ToyDB's WAL from turn 3 grows forever — what's the checkpointing strategy, and how does it interact with turn 7's abort cleanup?",
    "From turn 32: how does checkpointing interact with the cache invalidation from turn 17?",
    "Schema evolution. If we add a column to the table from turn 11, does the storage layout from turn 1 require a rewrite? Why or why not?",
    "Following turn 34: does the secondary index from turn 12 need rebuilding during that schema change, given the write path from turn 2?",
    "Failure mode: if the checkpoint from turn 32 corrupts mid-write, what's the recovery story, referencing the WAL guarantees from turn 3?",
    f"From turn 36: during that recovery, what does the MVCC cleanup from turn 7 do with half-applied transactions like {MIDSEG_MARKER}?",
    "Performance. Given everything from turns 1-37, what's the single biggest throughput bottleneck in ToyDB, and is it the storage, the WAL, or the cache?",
    "Final synthesis: in one paragraph, summarize ToyDB's design as a chain of the trade-offs you made across turns 1, 6, 17, 21, and 32.",
    "Last question: of all the choices we made, which one would you revisit first if ToyDB had to handle 1000x the write rate from turn 2, and why?",
]

assert len(DIALOGUE) == N_TURNS, f"dialogue has {len(DIALOGUE)} turns, expected {N_TURNS}"
assert MIDSEG_MARKER in DIALOGUE[MIDSEG_TURN - 1]
assert "transaction id" in DIALOGUE[REFERENT_TURN - 1].lower()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _to_records(messages: list[BaseMessage]) -> list[dict]:
    out = []
    for m in messages:
        out.append({
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "id": getattr(m, "id", None),
            "content": m.content or "",
        })
    return out


def _from_records(records: list[dict]) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for r in records:
        if r["role"] == "user":
            msgs.append(HumanMessage(content=r["content"], id=r["id"]))
        else:
            msgs.append(AIMessage(content=r["content"], id=r["id"]))
    return msgs


def _messages_sha256(records: list[dict]) -> str:
    canonical = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Baseline probe — settle "is the gateway slow" BEFORE blaming anything.
# ---------------------------------------------------------------------------

def probe(chat_model) -> None:
    print("[probe] three trivial calls to establish gateway baseline...", flush=True)
    times = []
    for i in range(3):
        t0 = time.time()
        try:
            r = chat_model.bind(max_tokens=20).invoke([HumanMessage(content="say hi")])
            dt = time.time() - t0
            times.append(dt)
            print(f"  probe {i+1}: {dt:.2f}s  out={len((r.content or ''))} chars", flush=True)
        except Exception as exc:
            print(f"  probe {i+1}: FAILED after {time.time()-t0:.2f}s — {type(exc).__name__}: {exc}", flush=True)
            sys.exit(1)
    avg = sum(times) / len(times)
    print(f"[probe] avg {avg:.2f}s", flush=True)
    if avg > PROBE_ABORT_SECONDS:
        print(f"[probe] ABORT: baseline {avg:.1f}s > {PROBE_ABORT_SECONDS}s. "
              f"The gateway itself is slow — fix that before seeding 40 turns.", flush=True)
        sys.exit(1)
    print("[probe] gateway healthy. Any slowness from here is in the accumulating "
          "history (expected) or in this script (a bug).\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    config = load_config("config.yaml")
    if is_mock_mode(config):
        print("REJECTED: fixture generation needs a real LLM, not mock mode.")
        return 2

    provider_names = list(config["providers"].keys())
    chat_model = build_chat_model(provider_names[0], "medium", config=config, temperature=0.0)
    model_name = getattr(chat_model, "model_name", "?")

    print(f"[setup] model={model_name}")
    print(f"[setup] tokenizer={tokenizer_kind_for(model_name)}")
    print(f"[setup] turns={N_TURNS}  midseg_turn={MIDSEG_TURN}  referent_turn={REFERENT_TURN}")
    print(f"[setup] per-call timeout={PER_CALL_TIMEOUT}s\n", flush=True)

    probe(chat_model)

    # ---- resume ----------------------------------------------------------
    messages: list[BaseMessage] = []
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.exists():
        try:
            ck = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            if ck.get("model") == model_name and ck.get("n_turns") == N_TURNS:
                messages = _from_records(ck["messages"])
                print(f"[resume] loaded {len(messages)//2} completed turns from checkpoint\n", flush=True)
            else:
                print("[resume] checkpoint is for a different model/config — starting fresh\n", flush=True)
        except Exception as exc:
            print(f"[resume] checkpoint unreadable ({exc}) — starting fresh\n", flush=True)

    start_turn = len(messages) // 2

    # ---- seed ------------------------------------------------------------
    for i in range(start_turn, N_TURNS):
        hm = HumanMessage(content=DIALOGUE[i], id=f"h{i}")
        # FULL accumulated history — this is what makes the back-references real.
        payload = messages + [hm]
        tok_in = chat_model.get_num_tokens_from_messages(payload)

        t0 = time.time()
        status = "ok"
        try:
            ai = chat_model.bind(timeout=PER_CALL_TIMEOUT).invoke(payload)
            ai_text = getattr(ai, "content", "") or ""
            if not ai_text.strip():
                status = "EMPTY"
        except Exception as exc:
            ai_text = ""
            status = f"{type(exc).__name__}"
        dt = time.time() - t0

        if status != "ok":
            print(f"\n[FAIL] turn {i+1} failed after {dt:.1f}s — {status}", flush=True)
            print("[FAIL] Not writing a fixture with a broken turn. "
                  "Progress is checkpointed; fix the cause and re-run to resume.", flush=True)
            return 1

        am = AIMessage(content=ai_text, id=f"a{i}")
        messages.extend([hm, am])

        tok_out = chat_model.get_num_tokens_from_messages([am])
        tps = tok_out / dt if dt > 0 else 0.0
        print(f"turn {i+1:>2}/{N_TURNS} | {dt:>6.1f}s | in={tok_in:>6} out={tok_out:>5} "
              f"| {tps:>5.1f} tok/s | {status}", flush=True)

        # Checkpoint after EVERY turn.
        CHECKPOINT_PATH.write_text(
            json.dumps({"model": model_name, "n_turns": N_TURNS,
                        "messages": _to_records(messages)},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    # ---- integrity -------------------------------------------------------
    print("\n--- fixture integrity ---", flush=True)
    records = _to_records(messages)
    checks: list[tuple[str, bool, str]] = []

    checks.append(("message count == 2 * turns",
                   len(records) == 2 * N_TURNS,
                   f"{len(records)} messages"))

    empty = [r["id"] for r in records if not (r["content"] or "").strip()]
    checks.append(("no empty replies", not empty, f"empty: {empty[:5]}"))

    seed_reply = records[(MIDSEG_TURN - 1) * 2 + 1]["content"]
    checks.append((f"turn {MIDSEG_TURN} assistant acknowledges the id",
                   MIDSEG_MARKER in seed_reply,
                   f"snippet: {seed_reply[:120]!r}"))

    recall_reply = records[(REFERENT_TURN - 1) * 2 + 1]["content"]
    checks.append((f"turn {REFERENT_TURN} assistant RECALLS the id  <-- proves history accumulated",
                   MIDSEG_MARKER in recall_reply,
                   f"reply: {recall_reply[:160]!r}"))

    refs = sum(1 for d in DIALOGUE if "turn " in d.lower() or "earlier" in d.lower())
    checks.append(("dialogue has >= 8 back-references", refs >= 8, f"{refs} refs"))

    total_tok = chat_model.get_num_tokens_from_messages(messages)
    checks.append(("total tokens >= 20000 (long enough to need compaction)",
                   total_tok >= 20000, f"{total_tok} tokens"))

    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}\n         {detail}", flush=True)
        ok = ok and passed

    if not ok:
        print("\n[FAIL] integrity checks failed — fixture NOT written.", flush=True)
        print("       The turn-30 recall check is the important one: if it fails, "
              "history was not accumulated and the dialogue's back-references are fake.", flush=True)
        return 1

    # ---- write -----------------------------------------------------------
    digest = _messages_sha256(records)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps({
            "meta": {
                "model": model_name,
                "n_turns": N_TURNS,
                "midseg_marker": MIDSEG_MARKER,
                "midseg_turn": MIDSEG_TURN,
                "referent_turn": REFERENT_TURN,
                "total_tokens": total_tok,
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "messages_sha256": digest,
            },
            "messages": records,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print(f"\n[OK] wrote {FIXTURE_PATH}")
    print(f"[OK] messages_sha256 = {digest}")
    print(f"[OK] total tokens    = {total_tok}")
    print("\nNext: commit this file. Never regenerate it — every future acceptance")
    print("run must load it, so numbers stay comparable across rounds.")
    print(f"Record the sha256 in the repo README so tampering is visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
