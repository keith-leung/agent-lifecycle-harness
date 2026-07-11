# agent-lifecycle-harness

A stateful-agent lifecycle harness demonstrating where framework persistence guarantees end and the application layer must begin.

## Problem

A stateful agent that runs is not a stateful agent that is production-shaped. LangGraph (and OpenAI Agents SDK) give persistence primitives — `SqliteSaver`, `BaseStore`, `thread_id` on one side; `Session`, `RunConfig`, request-scoped reload on the other — but deliberately do not give compaction policy, poison-tombstone propagation, state-schema migration, or reasoning-degradation monitoring. Those are application-owned.

This repo builds those application-owned layers, demo by demo, with machine-checkable assertions on real LLM calls. Each demo is end-to-end: the agent runs a long session, lifecycle events happen naturally, and the harness verifies behavior.

## Setup

```bash
conda create -n agent-lifecycle-harness python=3.11 -y
conda activate agent-lifecycle-harness
pip install -e ".[dev]"
```

Copy `config.example.yaml` to `config.yaml` and fill in real keys. `config.yaml` is gitignored.

## Architecture

### End-to-end pattern

Each demo follows the same pattern:
1. **Long-running agent**: runs N turns naturally (not manually orchestrated)
2. **Lifecycle event**: compaction, truncation, tombstone, migration happens automatically or as a natural consequence of running
3. **Verification**: assertions check that the agent's behavior reflects the lifecycle event (not just that a side-store record exists)

### Replay integration

The harness injects compaction digests into the prompt via `_build_inputs()` in [agent.py](file:///c:/Users/orange_forever/Documents/Workspaces/AgenticPortfolios202607/repos/A_agent-lifecycle-harness/src/agent_lifecycle_harness/agent.py#L67). When a thread has digests, the graph node receives a system message containing digest summaries, making compaction part of the replay path rather than a side-store decoration.

### Cross-vendor judge

A5 accepts per-turn quality scores from any judge (MiniMax in production). The detector's correctness is tested on fixture sequences; the judge's correctness is B's territory.

## A1 — Concurrent-user isolation

### Failure mode

An agent serving multiple users with no defined isolation unit suffers cross-user state leakage. Concurrent writers to the same thread cause torn writes. Accidental reuse of one id across users silently mixes state.

### Architecture

The isolation unit is `thread_id` (LangGraph). The app layer adds:

- **Namespaced ids**: `user:<uid>:session:<sid>` to make accidental reuse a visible bug.
- **Per-thread write lock**: concurrent writers to the same thread serialize; writers to different threads do not block.
- **Shared vs thread-local boundary**: `BaseStore` (cross-thread) holds provenance + tombstone audit + migration ledger — explicitly documented as shared trust boundary. Thread-local = agent conversational state.

**Checkpointer note:** `SqliteSaver` is used for single-file demo reproducibility. `PostgresSaver` / `AsyncPostgresSaver` are the production-grade checkpointer (LangGraph Platform itself uses Postgres). Pairing SqliteSaver-for-demo with PostgresSaver-for-production is the intended production story.

### Running

**CI-MOCK (no API keys needed):**

```bash
conda activate agent-lifecycle-harness
$env:AGENT_HARNESS_CI_MOCK="1"
python -m agent_lifecycle_harness.run --demo A1
```

**Real LLM (requires valid `config.yaml`):**

```bash
conda activate agent-lifecycle-harness
python -m agent_lifecycle_harness.run --demo A1
```

### Assertions

A1 exercises five assertions:

- (a) each thread resumes only its own history
- (b) no thread's state contains a key seeded only by another thread
- (c) per-thread checkpoint counts are independent and thread-scoped
- (d) two concurrent writers to the SAME thread serialize without torn writes
- (e) accidental-reuse bug is reproduced on a bare id, then fixed with namespaced ids

## A2 — Checkpoint retention & compaction

### Failure mode

Conversations grow unbounded; with no retention/compaction policy, persisted state balloons without limit or, if naively truncated, breaks replay.

### Architecture

LangGraph does not provide compaction. The app layer adds:

- **first-N + last-N policy**: the oldest N and newest N raw checkpoints are preserved; the middle is compressed.
- **model-generated digest**: the middle segment is summarized into ONE digest entry that embeds the replaced raw checkpoint ids.
- **idempotent compactor**: re-running on an already-compacted thread is a no-op.
- **lossy fields metadata**: the digest records which state fields were not preserved in the summary.
- **replay integration**: digests are injected into the prompt via `_build_inputs()` in [agent.py](file:///c:/Users/orange_forever/Documents/Workspaces/AgenticPortfolios202607/repos/A_agent-lifecycle-harness/src/agent_lifecycle_harness/agent.py#L67), making compaction part of the replay path.

### Running

```bash
$env:AGENT_HARNESS_CI_MOCK="1"
python -m agent_lifecycle_harness.run --demo A2
```

### Assertions

A2 exercises four assertions:

- (a) after compaction, exactly one digest covers the middle raw checkpoints
- (b) subsequent invoke produces a coherent reply (agent "remembers" via digest)
- (c) re-running compaction on the same range is a no-op (idempotent)
- (d) lossy fields are enumerated in the digest metadata

## A2∩A3 — Compaction + Tombstone Interop

### Scenario

Poison at turn 3; run turns 4-10; compact turns 4-8 into digest; tombstone turn-3; assert digest is identified as affected AND re-run produces poison-free output.

### Key insight

Digest embeds `replaced_raw_ids` so A3 can identify which digests are affected when a raw checkpoint inside a digest range is tombstoned.

### Running

```bash
$env:AGENT_HARNESS_CI_MOCK="1"
python -m agent_lifecycle_harness.run --demo A2_A3
```

## A3 — Poison-item tombstoning

### Failure mode

A poisoned context (bad data, bad tool result, hallucinated seed) is cached in checkpoint history. Subsequent turns that consumed it propagate corruption silently.

### Architecture

- **provenance DAG**: each checkpoint entry gets a provenance record (`sha256`, `parent_ids`, `produced_by`, `produced_at`) stored in app-owned BaseStore indices.
- **soft tombstone**: mark an entry poisoned (flag, not delete — reversible + auditable).
- **DAG traversal**: find all downstream checkpoints whose parent-chain transitively includes the poisoned one.
- **re-run policy**: re-run affected turns in post-poison context; compare outputs to pre-tombstone to confirm divergence.

### Running

```bash
$env:AGENT_HARNESS_CI_MOCK="1"
python -m agent_lifecycle_harness.run --demo A3
```

### Assertions

- (a) DAG traversal finds poisoned checkpoint + downstream as affected
- (b) re-run of affected turns yields outputs ≠ pre-tombstone
- (c) tombstone is soft (recoverable from audit log)
- (d) audit log records op + actor + ts

## A4 — Config hot-reload

### Failure mode

Changing config without versioning can silently alter behavior of ongoing sessions.

### Architecture

- **version-on-session**: each session is tagged with the config version active at start time.
- **ongoing sessions continue on old version**: reload does not mutate existing sessions.
- **new sessions pick up latest**: fresh sessions get the current version.

### Running

```bash
$env:AGENT_HARNESS_CI_MOCK="1"
python -m agent_lifecycle_harness.run --demo A4
```

### Assertions

- (a) ongoing session retains its version after reload
- (b) new session picks up the latest version

## A5 — Reasoning-degradation monitoring

### Failure mode

Silent degradation from context truncation (history-growth failure mode) is invisible until user complaints.

### Architecture

**Detector correctness, not agent degradation (re-designed 2026-07-06):** the prior version required "agent runs 30 turns, history truncated from turn 20, judge scores drop" — but real agents don't deterministically degrade under truncation, so this is not a reliable demo. The re-designed A5 tests the **DegradationMonitor detector on known fixture score sequences**, not the agent's degradation:

- **DegradationMonitor (app-owned):** takes per-turn quality scores (however produced) + baseline, computes sustained-delta over ≥k consecutive samples, fires `degradation_detected` when threshold + k both met.
- **Fixture-driven assertions:**
  - (a) **fire on known-degrading sequence:** turns 1-10 score 0.9, turns 11-20 score 0.5 → detector fires with sustained_count ≥ k + delta > threshold.
  - (b) **NOT fire on stable sequence (false-positive guard):** all turns score 0.9 → detector does NOT fire.
  - (c) **trend-based, not single-sample:** one dip then recovery (turn 5 = 0.5, others 0.9) → detector does NOT fire.
  - (d) **mitigation loop:** feed degrading sequence → detector fires → mitigation hook (auto-trigger A2 compaction) is actually invoked (verified via spy/mock on the hook, not via real compaction succeeding).

### Running

```bash
$env:AGENT_HARNESS_CI_MOCK="1"
python -m agent_lifecycle_harness.run --demo A5
```

### Assertions

- (a) degradation_detected fires on known-degrading fixture: delta=0.400, sustained=3
- (b) control group (stable fixture, all 0.9) does NOT fire (false-positive guard)
- (c) trend-based: single-dip fixture does NOT fire (trend-based, not single-sample)
- (d) mitigation hook fires (spy count=8 after 3 sustained samples)

## A6 — State schema migration

### Failure mode

State schema changes without migration break old persisted checkpoints.

### Architecture

- **schema registry**: register schema versions with backward-compatibility flags.
- **migration fn**: per-version migration transforms old state to new.
- **backward compatibility**: old state can be read by new code without data loss of unmigrated fields.
- **transactional**: original state is not mutated during migration.
- **resumable**: migrated state can be used immediately (new fields accessible).
- **v1-shape-read-on-migrated raises**: accessing old v1-only fields on migrated state raises an error, preventing silent shape confusion.

### Running

```bash
$env:AGENT_HARNESS_CI_MOCK="1"
python -m agent_lifecycle_harness.run --demo A6
```

### Assertions

- (a) migration preserves data not covered by the migration
- (b) latest schema is backward compatible
- (c) transactional: original state is not mutated
- (d) resumable: migrated state can be used (new fields accessible)
- (e) v1-shape-read-on-migrated raises: old field access raises on migrated state

## A7 — Cross-framework lifecycle matrix

Maps A1-A6 lifecycle boundaries to OpenAI Agents SDK equivalents. Each OAI-side cell either runs with assertion or is documented as "framework-given, here's the API used" with a citation in code.

| Demo | LangGraph concept | OpenAI Agents SDK equivalent | App-layer boundary | Framework-owned | App-owned | OAI API citation |
|------|-------------------|------------------------------|--------------------|-----------------|-----------|------------------|
| A1 | thread_id isolation + per-thread write lock | Session + RunConfig (request-scoped reload) | namespaced ids + write serialization | thread_id (LG) / Session (OAI) | namespacing + locks | `openai.agents.Session` |
| A2 | SqliteSaver checkpoint history + app-owned CompactionStore | OpenAIResponsesCompactionSession (framework-given) | first-N + last-N + middle-digest | checkpoint table (LG) / OpenAIResponsesCompactionSession (OAI) | compaction policy + digest (LG only) | `openai.agents.OpenAIResponsesCompactionSession` |
| A3 | BaseStore provenance + soft tombstone + DAG traversal | Session.pop_item (removes provenance, no tombstone) | provenance DAG + audit log | opaque run records | provenance + tombstone + rerun (both) | `openai.agents.Session.pop_item` |
| A4 | config-version tracker per thread_id | RunConfig (request-scoped, framework-given) | version tracker + session registration | thread_id / Session | config version mapping (LG only) | `openai.agents.RunConfig` |
| A5 | invoke instrumentation + threshold alerting | run metrics + alerting hooks (framework-given) | DegradationMonitor | execution trace | metrics + alerts (both) | `openai.agents.Run metrics` |
| A6 | state schema registry + migration fn | run schema versioning (framework-given) | SchemaRegistry | serialized state blob | schema + migration fn (both) | `openai.agents.Schema versioning` |

**Key reference points:**
- OAI's `OpenAIResponsesCompactionSession` is framework-given (LG side is app-owned)
- OAI's `pop_item` removes provenance → tombstone is app-owned on both frameworks
- OAI's request-scoped `RunConfig` is framework-given hot-reload (LG side is app-owned)

### Running

```bash
$env:AGENT_HARNESS_CI_MOCK="1"
python -m agent_lifecycle_harness.run --demo A7
```

## A8 — Build-vs-buy justification

### Framing

I intentionally used LangGraph for checkpoint/store primitives and OpenAI Agents SDK for its request-scoped primitives, then implemented the lifecycle layers current frameworks do not natively ship. This is staff-level judgment about where frameworks end and the application layer begins — not NIH syndrome.

### Build-vs-buy table (mid-2026 survey)

| Lifecycle layer | Surveyed (mid-2026) | Verdict |
|---|---|---|
| Checkpoint persistence + retention/compaction | LangGraph Platform (TTL delete only, not semantic compaction); OpenAI Agents SDK `OpenAIResponsesCompactionSession` (conversation-history compaction, not checkpoint-DAG); DBOS/Temporal (durable execution, not session compaction) | **Build** — no framework ships compaction policy over checkpoint DAGs |
| Poison tombstone + downstream propagation | LangGraph (no invalidation hooks); DBOS ForkWorkflow (rerun mechanics, not tombstone semantics); Temporal Signals (event push, not provenance-driven invalidation) | **Build** — universally application-owned |
| Concurrent-user isolation + writer serialization | LangGraph `thread_id` (framework-given); OAI SDK `Session` (framework-given) | **Use framework** — thread_id/Session is the isolation primitive; app adds namespacing + per-thread write locks |
| Config hot-reload (version-on-session) | OAI SDK `RunConfig` (request-scoped, framework-given); LangGraph (no hot-reload protocol) | **Use OAI SDK where given; build on LG** |
| Reasoning-degradation monitoring | AgentOps/LangSmith/Arize (tracing/eval, not in-loop auto-mitigation); no framework ships judge-based drift detection | **Build** — universally application-owned |
| State-schema migration | No framework ships transactional agent-state migration | **Build** — universally application-owned |

## Keywords

- `BaseCheckpointSaver`
- `SqliteSaver`
- `PostgresSaver` / `AsyncPostgresSaver`
- `thread_id` isolation
- concurrent-user isolation
- compaction
- first-N + last-N + middle-digest
- lossy fields
- replay integration
- provenance
- tombstone
- DAG traversal
- re-run policy
- config-version tracker
- version-on-session
- reasoning degradation
- context truncation
- cross-vendor judge
- sustained delta
- DegradationMonitor
- schema migration
- backward compatibility
- transactional
- resumable
- cross-framework lifecycle matrix
- OpenAI Agents SDK
- `OpenAIResponsesCompactionSession`
- `Session.pop_item`
- `RunConfig`
