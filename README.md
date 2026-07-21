# agent-lifecycle-harness

Stateful-agent lifecycle concerns built as explicit application-owned layers on top of LangGraph's persistence primitives, with a cross-framework matrix mapping each concern to its OpenAI Agents SDK equivalent.

LangGraph ships persistence primitives (`SqliteSaver`, `BaseStore`, `thread_id`) but deliberately does not ship compaction policy, poison-tombstone propagation, state-schema migration, degradation monitoring, or config-version pinning. This repo implements those layers, each with machine-checked assertions runnable in mock mode (no API keys) or against a real LLM.

## Status

| Demo | Concern | Mock | Real LLM | Tests |
|------|---------|------|----------|-------|
| A1 | Concurrent-user isolation | PASS | PASS | `tests/test_a1.py` |
| A2 | Checkpoint compaction (langmem) | PASS | PASS | `tests/test_a2.py` |
| A2∩A3 | Compaction ↔ tombstone interop | PASS | PASS | `tests/test_a2a3_interop.py` |
| A3 | Poison tombstone + DAG traversal | PASS | PASS | `tests/test_a3.py` |
| A4 | Config hot-reload (version-on-session) | PASS | PASS | `tests/test_a4.py` |
| A5 | Degradation monitoring (edge-triggered) | PASS | PASS | `tests/test_a5.py` |
| A6 | State schema migration | PASS | PASS | `tests/test_a6.py` |
| A7 | Cross-framework matrix (LG ↔ OAI SDK) | PASS | PASS | `tests/test_a7.py` |

Both modes are run by the same entry point; mock mode exercises detector logic and wiring against a deterministic client, real mode exercises the same code paths against the configured provider.

## Dependencies

| Package | Version | Role |
|---|---|---|
| `langgraph` | 1.2.7 | Checkpointer, `thread_id` isolation, state graph |
| `langmem` | 0.0.30 | A2 summarization engine (`summarize_messages`, `RunningSummary`) |
| `langchain-openai` | 1.2.2 | langmem-compatible chat model adapter |
| `openai-agents` | 0.17.7 | A7 cross-framework matrix target |
| `transformers` | 5.14.1 | DeepSeek native tokenizer (token counting in A2) |
| `tiktoken` | 0.13.0 | Fallback tokenizer for unknown models |

## Setup

```bash
conda create -n agent-lifecycle-harness python=3.11 -y
conda activate agent-lifecycle-harness
pip install -e ".[dev]"
```

Copy `config.example.yaml` to `config.yaml` and fill in real keys (gitignored). `config.ci.yaml` (mock mode, committed) needs no keys.

## Running

```bash
# Mock mode (deterministic, no API keys, ~1s)
python -m agent_lifecycle_harness.run --config config.ci.yaml --all

# Real LLM mode (requires filled config.yaml)
python -m agent_lifecycle_harness.run --config config.yaml --all

# Single demo
python -m agent_lifecycle_harness.run --config config.ci.yaml --demo A4
```

Each run prints per-demo timing and an `[heartbeat]` line if any invoke exceeds 30s, so a hang is distinguishable from slow-but-working. All LLM calls are logged to `debug.log` in the repo root.

## Architecture

### Core harness (`agent.py`)

`LifecycleHarness` wraps a single-node LangGraph `StateGraph` compiled with `SqliteSaver`. The node (`_call_model`) is strategy-agnostic — it asks the configured `CompactionStrategy` what the LLM should see, picks the LLM client (per-invoke via the model registry, falling back to the fixed default), and forwards the result.

The node never inlines lifecycle policy. Adding a policy means adding a strategy, not editing the node.

### Strategy pattern (A2, `compaction.py`)

Compaction is exposed behind a `CompactionStrategy` ABC with `build_replay_context(thread_id, full_history) -> ReplayOutcome`. Three implementations:

- **`NoCompactionStrategy`** — baseline; passes full history through unchanged. Reference point for reduction claims.
- **`DigestCompactionStrategy`** — merge semantics. Drives `langmem.short_term.summarize_messages`, which rewrites a single running summary in place on each fold. `RunningSummary.summarized_message_ids` is the source of truth for which messages are folded (range bookkeeping comes from the library, not hand-rolled).
- **`AppendCompactionStrategy`** — append semantics. Each fold produces an independent summary message appended to the list; prior summaries keep their bytes forever. Same `RunningSummary` id-tracking, different composition policy.

Token counting uses the model's native tokenizer when one is loadable (DeepSeek via `transformers.AutoTokenizer("deepseek-ai/DeepSeek-V4-Flash")`, verified byte-identical to V3, vocab 128000), falling back to `cl100k_base` with the difference surfaced via `tokenizer_kind_for()`.

`langmem.short_term.summarize_messages` is the only summarization engine; the repo does not reimplement summarization, token counting, or range arithmetic.

### Model registry (A4)

`LifecycleHarness` holds a `model_registry: dict[str, LLMClient]`. When `invoke(resolved_config=...)` carries a `model` key, the node looks up the registered client for that model and invokes it instead of the fixed default. This is the wire that makes a material config reload actually change the LLM answering a session — without it the resolved model is just a tracker label.

### Persistence boundary

- **LangGraph checkpoint table** (SqliteSaver): opaque checkpoint blobs, `thread_id` scoping, time-travel via `get_state_history`.
- **App-owned `CompactionStore`** (separate SQLite tables in the same db): `RunningSummary` state + a `message_id ↔ checkpoint_id` map so A3's tombstone DAG can resolve "did this compaction fold the message produced by checkpoint X?"
- **App-owned `ProvenanceStore`**: provenance DAG + tombstone audit log (in-memory in mock mode, `BaseStore`-backed in real mode).

---

## A1 — Concurrent-user isolation

**Failure mode.** Multi-user agents with no defined isolation unit leak state across users. Concurrent writers to one thread cause torn writes. Reusing one id across users silently mixes state.

**Implementation.** The isolation unit is `thread_id` (LangGraph). The app layer adds:

- **Namespaced ids**: `user:<uid>:session:<sid>` so accidental reuse is a visible bug, not silent mixing.
- **Per-thread write lock**: concurrent writers to the same thread serialize (`threading.Lock` keyed by `thread_id`); writers to different threads do not block.
- **Accidental-reuse reproduction**: the demo first runs two sessions on a bare shared id (state mixes), then on namespaced ids (state isolated) — the bug is demonstrated, then fixed.

**Assertions** (5):
1. `resume_own_history_only` — each thread resumes only its own history
2. `no_cross_thread_key_leak` — no thread's state contains a key seeded only by another thread
3. `independent_checkpoint_counts` — per-thread checkpoint counts are independent
4. `same_thread_writers_serialize` — concurrent writers to the same thread complete without torn writes
5. `accidental_reuse_then_fix` — bug reproduced on a bare id, fixed with namespaced ids

**Checkpointer note.** `SqliteSaver` is used for single-file reproducibility. `PostgresSaver` / `AsyncPostgresSaver` are the production-grade equivalents (LangGraph Platform itself uses Postgres). SqliteSaver-for-repro / PostgresSaver-for-production is the intended deployment story.

## A2 — Checkpoint compaction (langmem)

**Failure mode.** Conversations grow unbounded; without a retention/compaction policy, persisted state balloons and context windows overflow.

**Implementation.** Summarization is driven entirely by `langmem.short_term.summarize_messages`. The repo's `LangmemCompactor` adapter:

- Calls `summarize_messages(messages, running_summary=..., model=..., token_counter=model.get_num_tokens_from_messages, ...)` — token counting uses the model's own counter, not langmem's default approximate counter.
- Persists the resulting `RunningSummary` (summary text + `summarized_message_ids` + `last_summarized_message_id`) to `CompactionStore`, so fold state survives across invokes.
- Records a `message_id → checkpoint_id` map so A3 can resolve which checkpoint produced each folded message.

The strategy ABC (§Strategy pattern) makes the compaction policy pluggable: `DigestCompactionStrategy` (merge) and `AppendCompactionStrategy` (append) share the same langmem engine, differing only in how summary messages compose.

`exempt_message_ids` (used by A2's acceptance harness) physically removes the last N raw turns from langmem's input, folding only the older segment — guaranteeing the tail survives verbatim without summary-text/raw-tail duplication.

**Assertions** (4):
1. `compaction_shape` — one digest folds the producing checkpoints; running summary tracks folded message ids
2. `coherence_after_compaction` — post-compaction invoke still returns a coherent reply
3. `compactor_idempotent` — driving compaction to saturation, then one more pass, is a no-op
4. `lossy_fields_enumerated` — digest metadata enumerates which state fields were not preserved

**Acceptance harness (`demos/a2_acceptance.py`).** A separate real-LLM-only script exercises compaction on a fixed 40-turn fixture (`tests/fixtures/a2_conversation_40turns.json`, sha256-verified at load). It runs each strategy 3× and reports `tokens_after` min/median/max, runs the coherence probe 5× per strategy, and includes a summary-only probe that strips raw messages to test whether the summary itself retained information. Findings from this harness are reported as data, not claims — see the script output.

## A2∩A3 — Compaction + tombstone interop

**Scenario.** Seed 10 turns with a poison sentinel at turn 5; compact; tombstone turn 5; assert the digest is identified as affected (the poison message was folded into it) AND re-running affected checkpoints produces poison-free output.

**Key edge.** `RunningSummary.summarized_message_ids` carries the message ids langmem folded. The `compaction_msg_map` table resolves those ids back to producing checkpoint ids, so A3's tombstone traversal can ask "did this compaction absorb the message from the poisoned checkpoint?"

**Assertions** (2): `digest_identified_as_affected`, `rerun_poison_removed`.

## A3 — Poison-item tombstoning

**Failure mode.** A poisoned context (bad data, bad tool result, hallucinated seed) is cached in checkpoint history. Subsequent turns that consumed it propagate corruption silently.

**Implementation.**

- **Provenance DAG**: each checkpoint gets a provenance record (`sha256`, `parent_ids`, `produced_by`, `produced_at`) in `ProvenanceStore`.
- **Soft tombstone**: mark an entry poisoned (flag, not delete — reversible + auditable).
- **DAG traversal (BFS)**: find all downstream checkpoints whose parent-chain transitively includes the poisoned one.
- **Re-run policy**: re-run affected turns on a poison-free reconstruction of their input context. The proof is on the INPUT side (the rebuilt context must not contain the POISON sentinel) plus a non-empty model reply — deterministic and mode-independent, because a real LLM does not echo the POISON token verbatim so output-side proof is unsound.

**Assertions** (4): `dag_traversal_finds_affected`, `rerun_poison_removed`, `tombstone_soft`, `audit_log_has_op_actor_ts`.

## A4 — Config hot-reload (version-on-session)

**Failure mode.** Changing config without versioning silently alters the behavior of ongoing sessions.

**Implementation.** `ConfigVersionTracker` owns config-version state at the application layer (LangGraph provides no config-version scoping). It distinguishes:

- **Material changes** (`system_prompt`, `model`, `temperature`, plus any unknown key as fail-safe) → bump the version label. Ongoing sessions stay pinned to the version they registered with; only new sessions pick up the change.
- **Cosmetic changes** (`log_level`, `metrics_endpoint`, …) → propagate immediately to all sessions, no version bump, because they cannot affect model output.

`SessionConfig` freezes the material slice at registration time; cosmetic fields are overlaid live via `resolved_config()`. The app layer owns version tracking — `set_config()` classifies the diff and bumps or doesn't, callers read `resolved_config_for(session_id)`.

Sessions also carry a TTL (`session_ttl_seconds`, default None = never expire). `touch_session()` refreshes `last_active_at` on every successful turn so active long-lived sessions are never reaped. `cleanup_expired()` marks ended sessions (retained for audit, not deleted). An optional background sweeper (`start_sweeper` / `stop_sweeper`) drives `cleanup_expired` periodically.

**The wire to the LLM.** `resolved_config["model"]` is forwarded through `invoke(resolved_config=...)` into `config["configurable"]["model"]`. The node reads it and picks the registered `LLMClient` from `model_registry`, falling back to the default. Without this wire, the resolved model is just a tracker label. The load-bearing assertion `resolved_model_drives_llm` proves two sessions pinned to different models get replies with different model signatures — and a mutation that strips the wire makes it FAIL.

**Assertions** (14): `cosmetic_classified`, `cosmetic_propagates_immediately`, `checkpoint_version_stamped` ×2, `material_classified`, `material_pins_live_session`, `new_session_picks_up_latest`, `new_session_after_material`, `resolved_model_drives_llm` (load-bearing), `ongoing_session_retains_version`, `ttl_reaps_inactive`, `touch_prevents_reap`, `ttl_disabled_when_none`, `reaped_session_retained_for_audit`.

`config_version` is persisted into every checkpoint via LangGraph's `config["metadata"]` channel — `get_state()` recovers it from SQLite, not from an in-memory field.

## A5 — Reasoning-degradation monitoring

**Failure mode.** Silent degradation (e.g. from context truncation) is invisible until user complaints.

**Implementation.** `DegradationMonitor` consumes per-turn quality scores from a `Judge.score(prompt, reply)` call (the demo uses `MockJudge`, a rule-based judge that scores replies containing a degradation marker lower). Scores are not hardcoded literals — the load-bearing `scores_from_judge` assertion proves the monitor's input comes from `judge.score()` by checking `judge.call_log`.

Detection is **edge-triggered**: a sustained degradation event produces exactly ONE alert (the transition into the degraded state), not one per sample. A latch (`_alerted_sessions`) prevents alert storms — a 100-turn degraded run produces 1 alert, not 98. Recovery clears the latch so a subsequent degradation fires again.

**Assertions** (6): `degradation_detected`, `exactly_one_alert`, `control_no_false_positive`, `trend_based`, `mitigation_hook_fires_once`, `scores_from_judge`.

## A6 — State schema migration

**Failure mode.** State schema changes without migration break old persisted checkpoints.

**Implementation.** `SchemaRegistry` registers schema versions with backward-compatibility flags. `TransactionalMigrator` applies per-version migration functions; the original state is not mutated during migration (transactional), and the migrated state can be used immediately (resumable). Accessing old v1-only fields on migrated state raises, preventing silent shape confusion.

**Assertions** (5): `migration_preserves_data`, `backward_compatible`, `transactional`, `resumable`, `v1_shape_read_on_migrated_raises`.

## A7 — Cross-framework lifecycle matrix

Maps each A1–A6 concern to its OpenAI Agents SDK equivalent. Each OAI-side cell is exactly one of two kinds — there is no third "import-only" middle ground:

- **`BehaviorCell`** — runs against the installed SDK and asserts an OBSERVABLE behavior (session isolation, pop-removes-item, usage-accumulates, export-serializes-fields). Each carries a mutation proof: breaking the asserted behavior makes the cell FAIL.
- **`DocCell`** — the SDK symbol exists (citation is `site-packages` file:line) but the asserted behavior cannot be reproduced locally (needs a real OpenAI client, or the behavior is app-owned on BOTH frameworks). DocCells are honestly marked `executed=False` with a verified citation.

| Demo | LG concept | OAI SDK equivalent | OAI cell kind |
|------|------------|---------------------|---------------|
| A1 | `thread_id` isolation + write lock | `SQLiteSession` (per-session row scoping) | behavior (`add_items`/`get_items` isolation + shared-db mutation) |
| A2 | app-owned compaction + langmem | `OpenAIResponsesCompactionSession` | documented (`run_compaction` needs a real Responses client) |
| A3 | provenance DAG + soft tombstone | `Session.pop_item` (DELETE…RETURNING) | behavior (pop really removes + broken-pop mutation) |
| A4 | config-version tracker per session | `RunConfig` (no version field) | documented (version-on-session is app-owned on both) |
| A5 | DegradationMonitor (edge-triggered) | `Usage.add` + `TurnSpanData.export` | behavior (accumulation + serialization + mutations) |
| A6 | SchemaRegistry + migration fn | none | documented (SDK has no migration API; `trace_metadata` is arbitrary dict) |

**Verified symbols** (openai-agents 0.17.7, all confirmed at the cited file:line):

| Symbol | Location |
|---|---|
| `SQLiteSession` | `agents/memory/sqlite_session.py:17` |
| `OpenAIResponsesCompactionSession` | `agents/memory/openai_responses_compaction_session.py:78` |
| `RunConfig` | `agents/run_config.py:211` |
| `SessionSettings` | `agents/memory/session_settings.py:24` |
| `Usage` (`add(other: Usage)`) | `agents/usage.py:102` |
| `TurnSpanData` (`export`) | `agents/tracing/span_data.py:98` |

**Assertions** (10): 3 structural (`matrix_complete`, `oai_cells_run_or_cited`, `app_layer_boundary_documented`) + 4 behavior cells + 3 doc cells.

## Mutation testing

Every load-bearing assertion has a corresponding mutation that must make it FAIL. Mutations are run as one-off scripts that monkey-patch the asserted behavior and re-run the relevant cell/demo. Examples:

- **A4** `resolved_model_drives_llm`: strip the model-registry lookup so the node always uses the fixed client → both sessions get the same signature → FAIL.
- **A5** `exactly_one_alert`: restore the old level-triggered `evaluate` → 8 alerts for one event → FAIL.
- **A5** `scores_from_judge`: replace `MockJudge.score` with a constant → no degradation signal → FAIL.
- **A7** behavior cells: break isolation (shared db), break pop (no-op), break add (no-op), break export (drop `data`) → each FAILs its cell.

A PASS-ing assertion that survives its mutation is treated as not actually testing the asserted behavior.

## Project layout

```
src/agent_lifecycle_harness/
├── agent.py                      # LifecycleHarness, _build_graph, model registry, heartbeat
├── compaction.py                 # CompactionStrategy ABC + langmem-driven strategies
├── provenance.py                 # ProvenanceStore, provenance DAG, tombstone audit
├── tombstone.py                  # tombstone_items_matching + BFS traversal + rerun
├── hotreload.py                  # ConfigVersionTracker, ChangeClassifier, SessionConfig, TTL
├── degradation.py                # DegradationMonitor (edge-triggered), Judge, MockJudge
├── migration.py                  # SchemaRegistry + TransactionalMigrator
├── openai_agents_harness.py      # A7 BehaviorCell / DocCell cells
├── llm.py                        # LLMClient hierarchy, ChatModel (native tokenizer), MockChatModel
├── config.py                     # config loader
├── debug_log.py                  # debug.log writer
├── run.py                        # CLI entrypoint
└── demos/
    ├── a1_isolation.py ... a7_cross_framework.py
    ├── a2a3_interop.py
    └── a2_acceptance.py          # real-LLM-only acceptance harness (fixture-driven)
tests/
├── fixtures/a2_conversation_40turns.json   # fixed 40-turn fixture (sha256-verified)
└── test_a1.py ... test_a7.py
```

## Reproducibility notes

- **Mock mode is fully deterministic.** `MockLLMClient` produces a content-derived echo; `MockChatModel` produces a deterministic summary. Mock-mode runs are byte-stable across invocations.
- **Real-LLM mode is not deterministic.** DeepSeek-V4-Flash does not document greedy-sampling determinism even at `temperature=0`; MTP speculative decoding, non-deterministic logprobs, and floating-point non-associativity all introduce variance. Real-mode runs that depend on LLM output (A2 acceptance, A4 coherence) are run multiple times with min/median/max reported.
- **Gateway reliability.** The configured gateway (`gpt-agent.cc`) is a shared upstream and intermittently rate-limits or drops routes. `MultiVendorLLMClient.__init__` probes each provider but treats timeout/connection errors as transient (keeps the client) so a gateway hiccup at startup doesn't make every provider unusable.
