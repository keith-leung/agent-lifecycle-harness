# ROUND-3 WORK ORDER — end-to-end hardening (3 tasks)

> Read `SPEC.md`, `HANDOFF.md` (incl. the six hard disciplines and the
> ROUND-2 fixes), and the files named below **before** writing code. Work
> **only inside this directory**. This is a precise, line-by-line work order —
> do NOT "re-interpret and rebuild"; do exactly what each task says.

Two of the three tasks close a **surface-compliance shortcut**: the code
*looks* like it demonstrates the behavior, but the assertion would still pass
if the real behavior were deleted. The default failure mode is the
highest-probability path — record a marker, `return True`, assert a field
exists. Each fix below makes the assertion depend on **observed runtime
behavior** that cannot be produced without the real implementation.

## Global acceptance standard (applies to every task) — the mutation test

An assertion is **only accepted** if it meets this bar:

> **If you replace the implementation under test with `pass` / a no-op / the
> pre-fix value, the assertion MUST fail.**

If the assertion still passes after that replacement, it is vacuous — it is
testing the existence of a record or the shape of a dict, not the behavior.
Every assertion you add or change in this work order must be accompanied, in
the report, by a one-line statement of *what no-op you mentally substituted
and why the assertion then fails*.

Additional rules (extend `HANDOFF.md` §"Six hard disciplines"):
- **Evidence is observed output, not status fields.** "It works" must be
  backed by captured stdout / captured before-and-after values written to a
  file, never by a `{"status": ...}` record or a boolean return.
- **Preserve evidence in the working tree.** Do not delete run artifacts.
  Write a human-readable `EVIDENCE_R3.md` at repo root and keep the `runs/*.db`
  and any captured logs the tasks below require.
- **Mock must be content-derived.** Any `MockLLMClient` on a path under test
  must derive its output from its actual input (echo/transform/hash of input
  content), so a sentinel present in the input appears in the output and one
  absent from the input does not. A mock whose output is independent of input
  cannot be used to prove a data-flow property.
- **Do not fake real mode.** If real-LLM mode cannot run (e.g. provider key
  invalid / quota exhausted), paste the exact provider error into
  `EVIDENCE_R3.md` and mark that path "real-blocked". Never present a mock run
  as a real run.

---

## Task 1 — Actually run the harness end-to-end (mock, then real)

Run every demo A1–A7 twice: once with `--config config.ci.yaml` (mock) and
once with `--config config.yaml` (real). Capture the **full stdout** of each
run into `EVIDENCE_R3.md` (or files under `runs/` that `EVIDENCE_R3.md` links).

Acceptance evidence:
- Mock: `python -m agent_lifecycle_harness.run --all --config config.ci.yaml`
  exits 0; stdout captured.
- Real: same with `--config config.yaml`; stdout captured. The real stdout
  must show **natural-language LLM output** (not a fixed mock prefix) for at
  least the A1 and A2 replies, proving a real provider answered.
- If real mode is blocked, the exact error text is pasted and the path is
  marked "real-blocked" — do not claim real pass.

Mutation-test note: n/a (this task is "run it and show the output"), but the
real-vs-mock distinction is itself the anti-fake check — a mock prefix in the
"real" log is a fail.

---

## Task 2 — Poison-tombstone: make the re-run REAL (highest priority)

**Where:** `src/agent_lifecycle_harness/tombstone.py`,
`src/agent_lifecycle_harness/demos/a2a3_interop.py`,
`src/agent_lifecycle_harness/demos/a3_tombstone.py`,
`src/agent_lifecycle_harness/llm.py` (MockLLMClient).

**Current shortcut:** `tombstone_items_matching(...)` accepts a `rerun_fn`
hook, but the demos call it with `rerun_fn=None`, so each affected checkpoint
records the placeholder `{"status": "needs_rerun", ...}`. The assertion
`assert_rerun_produces_poison_free_output` then only checks that such a record
exists for each affected id. **It never re-executes anything and never
inspects any output.** Deleting the entire re-run mechanism would not fail
this assertion — that is the vacuity this task removes.

**Poison model (do not "fix" this):** the poisoned turn is designated by an
external input (turn 3 is seeded with the literal sentinel token `POISON`).
Designating the poisoned item via an external signal is correct and
intentional — it models an upstream invalidation event, not a semantic
judgment. `POISON` here is a **literal seeded sentinel**; asserting its literal
presence/absence in output is *structural* verification of context flow
(permitted under hard discipline §3), NOT semantic understanding.

**Fix precisely:**
1. Provide a real `rerun_fn` from the demo. For each affected downstream
   checkpoint, it must:
   - reconstruct that turn's input **with the tombstoned ancestor's poisoned
     content excluded or replaced** (i.e. the re-run must NOT see the `POISON`
     sentinel in its input context), then
   - re-execute that turn **through the LangGraph harness** (`harness.invoke`
     / the graph), and
   - return the **actual new output text** produced.
2. Make `MockLLMClient` on this path content-derived (see global rules): its
   reply must embed a transform of its input, so the `POISON` sentinel
   surfaces in the reply iff it was present in the input context.
3. Rewrite `assert_rerun_produces_poison_free_output` to, for each affected
   checkpoint, capture `pre_output` (the original reply, which contains
   `POISON`) and `post_rerun_output` (the re-run reply), and assert
   `"POISON" not in post_rerun_output` **and** `"POISON" in pre_output`.
   Write both raw strings, side by side per checkpoint, into `EVIDENCE_R3.md`.

Acceptance evidence:
- `EVIDENCE_R3.md` shows, per affected checkpoint, the literal `pre_output`
  (contains `POISON`) and `post_rerun_output` (does not).
- Mutation test you must report: "if `rerun_fn` is reverted to `None` (marker
  only), or the mock is made input-independent, the assertion fails because
  there is no post-rerun output to inspect / the sentinel is not removed."

Anti-fake checks a reviewer will run (write your code to survive them):
- Substitute `rerun_fn` with one that returns the pre-tombstone output
  verbatim → assertion must fail.
- Grep the demo for a hardcoded `post_output = "clean"` or equivalent → must
  not exist; the post output must come from an actual graph invocation.

---

## Task 3 — Remove the fabricated framework feature in A7

**Where:** `src/agent_lifecycle_harness/openai_agents_harness.py`,
function `assert_oai_schema_versioning`; and the A7 table row A6 in
`README.md`.

**Current shortcut:** the function writes `{"schema_version": "v2"}` into an
OpenAI Agents SDK `RunConfig.trace_metadata` dict, reads it back, and returns
evidence claiming *"OAI SDK run schema versioning is framework-given."* The
SDK has **no** state-schema-versioning feature; `trace_metadata` is an
arbitrary user dict. This asserts a framework capability that does not exist —
worse than an unimplemented cell, because it states a false fact a
knowledgeable reader will catch.

**Fix precisely:**
- Change the cell so it makes **no claim that the OAI SDK provides schema
  versioning or migration.** The honest statement is: the OAI SDK does not
  ship state-schema migration; therefore A6 (schema registry + migration fn)
  is **application-owned on both frameworks**. You may still show that
  `RunConfig`/`trace_metadata` exist as real symbols (import + construct), but
  the evidence string must not imply that carrying a `schema_version` key in
  trace metadata is a versioning feature.
- Update the A7 matrix row A6 in `README.md` to match: OAI equivalent =
  "none (app-owned migration; `trace_metadata` is arbitrary metadata, not a
  versioning API)".
- Re-audit the neighbouring cells A4 (`RunConfig` hot-reload) and A5 (run
  metrics) for the same failure: if their evidence string claims a
  framework-given capability that the code does not actually exercise, soften
  it to exactly what the code proves (symbol exists / object constructs) and
  state explicitly what remains app-owned.

Acceptance evidence:
- The new A6 evidence string + README row, quoted in `EVIDENCE_R3.md`, with a
  one-line note confirming no non-existent OAI capability is asserted anywhere
  in A7.
- Mutation test: n/a (this is a truthfulness fix) — the check is that a reader
  who knows the OAI SDK cannot find a claimed feature that isn't in the SDK.

---

## Report back (in `EVIDENCE_R3.md`, and to your caller)

For each of the 3 tasks:
- files/lines changed, with the post-fix snippet of every assertion touched;
- the **mutation test** for each assertion you added/changed (what no-op you
  substituted, why the assertion then fails);
- the captured evidence (run stdout for Task 1; per-checkpoint pre/post output
  for Task 2; the corrected strings for Task 3);
- honest flags: anything you could not make real (say so precisely — do not
  work around silently), especially if real-LLM mode was blocked.

Do not report "all pass." Report **what behavior each assertion now depends
on, and the exact line where that behavior is produced.**
