"""A2 — Short-term memory compaction, driven by ``langmem``.

All summarization goes through ``langmem.short_term.summarize_messages``.
This module does NOT reimplement summarization, token counting, or range
arithmetic — those are langmem's job:

* summarization policy (when to fold, what to fold, the folded text):
  ``summarize_messages`` (``langmem/short_term/summarization.py:337``)
* token counting: ``ChatOpenAI.get_num_tokens_from_messages`` (passed as
  ``token_counter=``, overriding the default approximate counter)
* incremental compaction / range overlap: ``RunningSummary`` carries
  ``summarized_message_ids`` and ``last_summarized_message_id``
  (``summarization.py:62-66``); langmem skips already-summarized messages
  on the next call, so overlapping ranges are handled by construction.

This module only owns:

* **Persistence** — ``CompactionStore`` writes ``RunningSummary`` state and
  a ``message_id -> checkpoint_id`` map to SQLite so A3's tombstone DAG can
  ask "did this compaction fold the message produced by checkpoint X?".
* **The langmem adapter** — ``LangmemCompactor`` builds the ``ChatOpenAI``
  model used both for summary generation and for exact token counting,
  calls ``summarize_messages``, and records the provenance edge.
* **Prefix-cache breakpoint strategy** — the only piece that no library
  manages: when a summary appears, the previous prefix cache is broken at
  some position. Merge vs Append is an application policy choice; both are
  provided (see ``prefix_cache.py``).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import AnyMessage
from langmem.short_term import RunningSummary, summarize_messages


@dataclass
class DigestEntry:
    """Compatibility view onto a ``RunningSummary`` for A3's DAG traversal.

    A3 needs "this compaction folded the message produced by checkpoint X".
    In langmem terms that is "checkpoint X's message id ∈
    ``running_summary.summarized_message_ids``". We surface the same edge
    through ``replaced_raw_ids`` so A3's existing traversal code is
    unchanged, but the source of truth is ``running_summary`` below.
    """
    digest_id: str
    thread_id: str
    summary: str
    replaced_raw_ids: list[str]
    lossy_fields: list[str]
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    running_summary: RunningSummary | None = None


class CompactionStore:
    """Persistence for ``RunningSummary`` state + message/checkpoint map.

    Backed by a dedicated SQLite table, separate from LangGraph's
    checkpoint table. Two relations:

    * ``compaction_running_summary`` — one row per (thread_id): the latest
      ``RunningSummary`` (summary text, summarized message ids, last
      summarized message id, created_at).
    * ``compaction_msg_map`` — (thread_id, message_id, checkpoint_id): the
      checkpoint that produced each message, recorded on every compaction
      call so A3 can resolve "message id → checkpoint id" backwards.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS compaction_running_summary (
                thread_id TEXT PRIMARY KEY,
                digest_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                summarized_message_ids TEXT NOT NULL,
                last_summarized_message_id TEXT,
                lossy_fields TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS compaction_msg_map (
                thread_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                checkpoint_id TEXT,
                PRIMARY KEY (thread_id, message_id)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------ msg map

    def record_message_checkpoints(
        self,
        thread_id: str,
        mapping: dict[str, str | None],
    ) -> None:
        """Persist message_id → checkpoint_id for a thread.

        Called on every compaction with the full current mapping so A3 can
        resolve any summarized message id back to its producing checkpoint.
        """
        rows = [
            (thread_id, mid, cid) for mid, cid in mapping.items()
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO compaction_msg_map VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def checkpoint_for_message(self, thread_id: str, message_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT checkpoint_id FROM compaction_msg_map "
            "WHERE thread_id = ? AND message_id = ?",
            (thread_id, message_id),
        ).fetchone()
        return row[0] if row else None

    def message_ids_for_checkpoint(self, thread_id: str, checkpoint_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT message_id FROM compaction_msg_map "
            "WHERE thread_id = ? AND checkpoint_id = ?",
            (thread_id, checkpoint_id),
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------ running summary

    def save_running_summary(
        self,
        thread_id: str,
        digest_id: str,
        running_summary: RunningSummary,
        *,
        lossy_fields: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO compaction_running_summary VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                thread_id,
                digest_id,
                running_summary.summary,
                json.dumps(sorted(running_summary.summarized_message_ids)),
                running_summary.last_summarized_message_id,
                json.dumps(lossy_fields),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()

    def get_running_summary(self, thread_id: str) -> RunningSummary | None:
        row = self._conn.execute(
            "SELECT summary, summarized_message_ids, last_summarized_message_id "
            "FROM compaction_running_summary WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if not row:
            return None
        return RunningSummary(
            summary=row[0],
            summarized_message_ids=set(json.loads(row[1])),
            last_summarized_message_id=row[2],
        )

    def digests_for_thread(self, thread_id: str) -> list[DigestEntry]:
        """A3-facing view: each thread has at most one live summary.

        langmem uses a single rolling ``RunningSummary`` (see
        ``summarization.py:482-488`` — every fold updates the same object),
        so there is exactly one digest row per thread at any time. We return
        it as a single-element list to preserve A3's iteration shape.
        """
        row = self._conn.execute(
            "SELECT digest_id, summary, summarized_message_ids, "
            "       last_summarized_message_id, lossy_fields, created_at, metadata "
            "FROM compaction_running_summary WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if not row:
            return []
        summarized_ids = set(json.loads(row[2]))
        # Map summarized message ids back to checkpoint ids for A3.
        replaced_raw_ids = self._resolve_checkpoint_ids(thread_id, summarized_ids)
        rs = RunningSummary(
            summary=row[1],
            summarized_message_ids=summarized_ids,
            last_summarized_message_id=row[3],
        )
        return [
            DigestEntry(
                digest_id=row[0],
                thread_id=thread_id,
                summary=row[1],
                replaced_raw_ids=replaced_raw_ids,
                lossy_fields=json.loads(row[4]),
                created_at=row[5],
                metadata=json.loads(row[6]) if row[6] else {},
                running_summary=rs,
            )
        ]

    def _resolve_checkpoint_ids(
        self, thread_id: str, message_ids: set[str]
    ) -> list[str]:
        if not message_ids:
            return []
        placeholders = ",".join("?" * len(message_ids))
        rows = self._conn.execute(
            f"SELECT DISTINCT checkpoint_id FROM compaction_msg_map "
            f"WHERE thread_id = ? AND message_id IN ({placeholders}) "
            f"AND checkpoint_id IS NOT NULL",
            (thread_id, *message_ids),
        ).fetchall()
        return [r[0] for r in rows if r[0]]

    def is_range_compacted(self, thread_id: str, raw_ids: Sequence[str]) -> bool:
        """True if every checkpoint id in ``raw_ids`` produced at least one
        message that the current running summary has folded."""
        rs = self.get_running_summary(thread_id)
        if rs is None:
            return False
        covered_checkpoints = set(
            self._resolve_checkpoint_ids(thread_id, rs.summarized_message_ids)
        )
        return all(rid in covered_checkpoints for rid in raw_ids)


class LangmemCompactor:
    """Adapter that drives ``langmem.short_term.summarize_messages``.

    Owns nothing the library already owns:

    * summary generation → ``model.invoke`` inside langmem
    * token counting → ``model.get_num_tokens_from_messages`` (passed via
      ``token_counter=``, NOT the default ``count_tokens_approximately``)
    * incremental / overlapping ranges → ``RunningSummary`` flows back in
      via ``running_summary=`` and langmem skips already-summarized ids

    Owns:

    * the ``ChatOpenAI`` model handle (shared by summary-gen + token count)
    * persisting the resulting ``RunningSummary`` + the message→checkpoint
      map that A3's DAG traversal needs

    This is the **engine**. The user-facing contract is
    ``CompactionStrategy.build_replay_context`` (see §6.1); this class is
    an internal collaborator of ``DigestCompactionStrategy``, not a
    public interface.
    """

    def __init__(
        self,
        store: CompactionStore,
        summarization_model: Any,
        *,
        max_tokens: int,
        max_tokens_before_summary: int | None = None,
        max_summary_tokens: int = 256,
    ) -> None:
        self.store = store
        self.model = summarization_model
        self.max_tokens = max_tokens
        self.max_tokens_before_summary = max_tokens_before_summary
        self.max_summary_tokens = max_summary_tokens
        # Exact token counting from the model itself, per task requirement.
        self.token_counter = summarization_model.get_num_tokens_from_messages

    def compact(
        self,
        thread_id: str,
        messages: Sequence[AnyMessage],
        *,
        message_to_checkpoint: dict[str, str | None] | None = None,
        exempt_message_ids: set[str] | None = None,
    ) -> CompactionOutcome:
        """Run one summarization pass and persist results.

        ``message_to_checkpoint`` maps each message's ``id`` to the
        checkpoint that produced it; recorded so A3 can walk the
        compaction edge backward. Returns a ``CompactionOutcome`` carrying
        the messages the LLM should actually see this turn (already folded
        by langmem) plus the structural report.

        ``exempt_message_ids`` — message ids that must NEVER be folded
        (e.g. the preserved raw tail). They are physically removed from
        the input passed to langmem, then spliced back verbatim after the
        fold. This is the ONLY way to guarantee no duplication: if they
        were left in, langmem's summary could reference their content, and
        re-injecting the raw message would duplicate it.

        The exemption interacts with the running summary by truncating
        ``summarized_message_ids`` to the non-exempt prefix: any id that
        is no longer in the input (because it's exempt) is dropped from
        the running tracker so langmem's "already summarized" check
        (summarization.py:173) doesn't fire on the next call.
        """
        if message_to_checkpoint:
            self.store.record_message_checkpoints(thread_id, message_to_checkpoint)

        # Split exempt messages out of the input BEFORE handing to langmem.
        # Exempt messages are spliced back raw after the fold.
        if exempt_message_ids:
            fold_input = [m for m in messages if getattr(m, "id", None) not in exempt_message_ids]
            exempt_msgs = [m for m in messages if getattr(m, "id", None) in exempt_message_ids]
        else:
            fold_input = list(messages)
            exempt_msgs = []

        running_summary = self.store.get_running_summary(thread_id)
        # If any previously-summarized id is now exempt (no longer in
        # fold_input), drop it from the running tracker so langmem doesn't
        # raise "already summarized" when it re-encounters the id in a
        # future non-exempt position. Also reset last_summarized_message_id
        # to the latest still-present summarized message so langmem's cutoff
        # lookup (summarization.py:152-155) succeeds.
        if running_summary is not None and exempt_message_ids:
            fold_input_ids = {getattr(m, "id", None) for m in fold_input}
            pruned_ids = running_summary.summarized_message_ids & fold_input_ids
            pruned_last = running_summary.last_summarized_message_id
            if pruned_last not in fold_input_ids:
                # Find the latest summarized id still in fold_input, by
                # scanning fold_input in order.
                pruned_last = None
                for m in reversed(fold_input):
                    if getattr(m, "id", None) in pruned_ids:
                        pruned_last = getattr(m, "id", None)
                        break
            if pruned_ids != running_summary.summarized_message_ids or pruned_last != running_summary.last_summarized_message_id:
                running_summary = RunningSummary(
                    summary=running_summary.summary,
                    summarized_message_ids=pruned_ids,
                    last_summarized_message_id=pruned_last,
                )

        pre_fold_token_count = self.token_counter(messages)
        result = summarize_messages(
            fold_input,
            running_summary=running_summary,
            model=self.model,
            max_tokens=self.max_tokens,
            max_tokens_before_summary=self.max_tokens_before_summary,
            max_summary_tokens=self.max_summary_tokens,
            token_counter=self.token_counter,
        )
        folded_head = result.messages
        new_rs = result.running_summary

        # Splice exempt (raw, never-folded) messages back. They go at the
        # end because they are the most recent turns.
        folded_messages = list(folded_head) + list(exempt_msgs)
        post_fold_token_count = self.token_counter(folded_messages)

        folded_something_new = (
            new_rs is not None
            and (
                running_summary is None
                or new_rs.summary != running_summary.summary
                or new_rs.summarized_message_ids != running_summary.summarized_message_ids
            )
        )

        if folded_something_new:
            assert new_rs is not None
            digest_id = _make_digest_id(thread_id, sorted(new_rs.summarized_message_ids))
            lossy_fields = ["messages"]
            self.store.save_running_summary(
                thread_id,
                digest_id,
                new_rs,
                lossy_fields=lossy_fields,
                metadata={
                    "max_tokens": self.max_tokens,
                    "max_tokens_before_summary": self.max_tokens_before_summary,
                    "max_summary_tokens": self.max_summary_tokens,
                    "compacted_message_count": len(new_rs.summarized_message_ids),
                },
            )
            digest_view = self.store.digests_for_thread(thread_id)[0]
        else:
            digest_view = None

        return CompactionOutcome(
            messages=folded_messages,
            digest=digest_view,
            tokens_before=pre_fold_token_count,
            tokens_after=post_fold_token_count,
            folded_new=folded_something_new,
        )


@dataclass
class CompactionOutcome:
    """What one compaction pass produced for a single turn.

    ``messages`` is what the LLM sees (already structurally folded by
    langmem — folded raw messages are NOT in this list). The token counts
    are exact, taken from ``model.get_num_tokens_from_messages`` before
    and after the fold, so reduction % is observable not approximate.
    """
    messages: list[AnyMessage]
    digest: DigestEntry | None
    tokens_before: int
    tokens_after: int
    folded_new: bool


# ============================================================================
# §6.1 — Strategy pattern
# ============================================================================
# The node is strategy-agnostic: it calls ``build_replay_context`` and
# forwards whatever the strategy returns. New policies (sliding window,
# cache-aware, etc.) are added by subclassing ``CompactionStrategy`` —
# the node never needs editing. This is the prerequisite for every later
# A2 upgrade (§6.2 structural replacement, §6.3 incremental compaction,
# §6.4 cache-aware): without this interface, each upgrade would have to
# edit the node and risk breaking the others.
# ============================================================================


@dataclass
class ReplayOutcome:
    """Result of ``CompactionStrategy.build_replay_context``.

    Carries the messages the LLM should see this turn plus an observable
    structural report (``dropped_raw_count`` etc.) so demos/tests can
    assert on *what the strategy did*, not just on downstream LLM output.
    Without this channel, "architecture C actually deleted the middle"
    would be unverifiable.

    ``stable_prefix_tokens`` is the strategy's own honest report of how
    many tokens in the replay payload will remain byte-identical on the
    NEXT fold. merge rewrites its single summary in place → 0. append
    preserves all prior summary messages verbatim → their combined token
    count. This is the prefix-cache-stability signal the task requires
    strategies to self-report.
    """
    messages: list[AnyMessage]
    dropped_raw_count: int
    digest_messages_added: int
    tokens_before: int
    tokens_after: int
    strategy: str
    digest: DigestEntry | None = None
    stable_prefix_tokens: int = 0


class CompactionStrategy(ABC):
    """§6.1 interface. The node never knows which concrete strategy is in use."""

    name: str = "abstract"

    @abstractmethod
    def build_replay_context(
        self,
        thread_id: str,
        full_history: Sequence[AnyMessage],
    ) -> ReplayOutcome:
        """Return the messages the LLM should see this turn, plus a report."""
        ...


class NoCompactionStrategy(CompactionStrategy):
    """Baseline: pass the full history through unchanged.

    ``dropped_raw_count`` is always 0. Required as the A/B reference point
    for ``DigestCompactionStrategy`` — without a no-op baseline, the
    reduction claims of digest compaction are unverifiable.
    """

    name = "none"

    def __init__(self, token_counter: Any = None) -> None:
        # Optional token counter so the baseline can still report token
        # shape (tokens_before == tokens_after for the no-op path).
        self._token_counter = token_counter

    def build_replay_context(self, thread_id, full_history):
        history = list(full_history)
        toks = self._token_counter(history) if self._token_counter else 0
        return ReplayOutcome(
            messages=history,
            dropped_raw_count=0,
            digest_messages_added=0,
            tokens_before=toks,
            tokens_after=toks,
            strategy=self.name,
            digest=None,
        )


class DigestCompactionStrategy(CompactionStrategy):
    """Architecture C driven by langmem (merge semantics), §6.1 interface.

    On every turn the strategy asks the ``LangmemCompactor`` engine to fold
    the history (langmem decides whether to fold at all based on token
    thresholds and the running summary). The folded message list is what
    the LLM sees: folded raw messages are structurally gone (not prefixed),
    exactly as §6.2 requires.

    This is the **merge** variant: langmem rewrites the single summary
    message in place on every new fold (`DEFAULT_EXISTING_SUMMARY_PROMPT`,
    `summarization.py:31-40`). Old summary text is overwritten, so the
    prefix breaks at the summary position. Use ``AppendCompactionStrategy``
    when prefix-cache stability matters more than summary freshness.
    """

    name = "merge"

    def __init__(
        self,
        compactor: LangmemCompactor,
        *,
        checkpoint_resolver: Any = None,
    ) -> None:
        self.compactor = compactor
        self._checkpoint_resolver = checkpoint_resolver
        self.last_outcome: dict[str, ReplayOutcome] = {}

    def build_replay_context(self, thread_id, full_history):
        history = list(full_history)
        msg_to_ckpt = (
            self._checkpoint_resolver(thread_id)
            if self._checkpoint_resolver is not None
            else None
        )
        exempt = getattr(self.compactor, "_pending_exempt", None)
        outcome = self.compactor.compact(
            thread_id, history,
            message_to_checkpoint=msg_to_ckpt,
            exempt_message_ids=exempt,
        )
        digest_added = 1 if (outcome.folded_new and outcome.digest is not None) else 0
        dropped = max(0, len(history) - (len(outcome.messages) - digest_added))
        # merge: the single summary message is rewritten on every fold,
        # so nothing in the payload is guaranteed stable → 0.
        replay = ReplayOutcome(
            messages=outcome.messages,
            dropped_raw_count=dropped,
            digest_messages_added=digest_added,
            tokens_before=outcome.tokens_before,
            tokens_after=outcome.tokens_after,
            strategy=self.name,
            digest=outcome.digest,
            stable_prefix_tokens=0,
        )
        self.last_outcome[thread_id] = replay
        return replay


class AppendCompactionStrategy(CompactionStrategy):
    """Architecture C with **append** semantics: prior summary messages
    are preserved verbatim, new folds append a new summary message.

    langmem's `summarize_messages` always rewrites a single summary in
    place (merge). Append needs N summary messages that accumulate. We
    reuse langmem to GENERATE each fold's summary text (calling it with
    `running_summary=None` so each summary stands alone) and reuse
    `RunningSummary.summarized_message_ids` to track which messages are
    already folded — the library-provided range bookkeeping, not a
    hand-rolled one. What we add is purely the **summary-message
    composition policy**: append, not overwrite.

    Prefix-cache consequence: prior summary messages keep their exact
    bytes, so the prefix cache stays warm across new folds. The cost is
    slower summary freshness (old summaries are never rewritten to
    incorporate later context).
    """

    name = "append"

    def __init__(
        self,
        compactor: LangmemCompactor,
        *,
        checkpoint_resolver: Any = None,
    ) -> None:
        # The compactor's model + token_counter are reused for summary
        # generation and counting. We do NOT call compactor.compact() —
        # that drives the merge path. Instead we call langmem directly
        # with running_summary=None for each independent fold.
        self.compactor = compactor
        self.model = compactor.model
        self.token_counter = compactor.token_counter
        self.max_tokens = compactor.max_tokens
        self.max_tokens_before_summary = compactor.max_tokens_before_summary
        self.max_summary_tokens = compactor.max_summary_tokens
        self._store = compactor.store
        self._checkpoint_resolver = checkpoint_resolver
        # Per-thread accumulated summary messages (append grows this list).
        self._summaries: dict[str, list[AnyMessage]] = {}
        # Per-thread RunningSummary carried forward for id tracking only.
        self._running: dict[str, RunningSummary] = {}
        self.last_outcome: dict[str, ReplayOutcome] = {}

    def _fold_next_segment(
        self, thread_id: str, messages: list[AnyMessage],
        exempt_ids: set[str] | None = None,
    ) -> tuple[AnyMessage | None, RunningSummary | None, list[AnyMessage]]:
        """Drive langmem with running_summary=None on the not-yet-folded
        suffix, returning (new_summary_message, new_running, remaining_msgs)
        or (None, prev_running, messages) if langmem declined to fold.

        ``exempt_ids`` are removed from the input BEFORE handing to langmem
        — they are never folded, so their content cannot leak into a
        summary (which would duplicate when spliced back raw).
        """
        # Strip exempt + already-folded from input.
        prev = self._running.get(thread_id)
        already_folded = prev.summarized_message_ids if prev else set()
        input_msgs = [
            m for m in messages
            if getattr(m, "id", None) not in already_folded
            and (exempt_ids is None or getattr(m, "id", None) not in exempt_ids)
        ]

        result = summarize_messages(
            input_msgs,
            running_summary=None,  # independent fold — append, not merge
            model=self.model,
            max_tokens=self.max_tokens,
            max_tokens_before_summary=self.max_tokens_before_summary,
            max_summary_tokens=self.max_summary_tokens,
            token_counter=self.token_counter,
        )
        new_rs = result.running_summary
        if new_rs is None:
            return None, prev, messages
        # Merge the new fold's ids into our running tracker.
        all_summarized = (
            (prev.summarized_message_ids if prev else set())
            | new_rs.summarized_message_ids
        )
        merged_rs = RunningSummary(
            summary=new_rs.summary,
            summarized_message_ids=all_summarized,
            last_summarized_message_id=new_rs.last_summarized_message_id,
        )
        # The new summary message is the system message langmem synthesized
        # at the head of result.messages (everything before the first
        # non-summary message that has an id matching an input id).
        summary_msg = None
        for m in result.messages:
            if getattr(m, "id", None) is None or getattr(m, "type", "") == "system":
                summary_msg = m
                break
        return summary_msg, merged_rs, result.messages

    def build_replay_context(self, thread_id, full_history):
        history = list(full_history)
        msg_to_ckpt = (
            self._checkpoint_resolver(thread_id)
            if self._checkpoint_resolver is not None
            else None
        )
        if msg_to_ckpt:
            self._store.record_message_checkpoints(thread_id, msg_to_ckpt)

        # Read the exempt set stashed by _TailPreservingStrategy (if any).
        exempt_ids = getattr(self.compactor, "_pending_exempt", None)

        tokens_before = self.token_counter(history)
        # Repeatedly fold new segments until langmem produces nothing new.
        added_this_turn = 0
        prev_running = self._running.get(thread_id)
        passes = 0
        while passes < 10:
            summary_msg, new_running, _ = self._fold_next_segment(
                thread_id, history, exempt_ids=exempt_ids,
            )
            if summary_msg is None or new_running is None:
                break
            # Detect progress: did summarized_message_ids grow?
            prev_ids = prev_running.summarized_message_ids if prev_running else set()
            if new_running.summarized_message_ids == prev_ids:
                break
            self._summaries.setdefault(thread_id, []).append(summary_msg)
            self._running[thread_id] = new_running
            prev_running = new_running
            added_this_turn += 1
            passes += 1

        running = self._running.get(thread_id)
        if running is not None:
            # Compose: accumulated summary messages + unsummarized tail.
            # The tail includes exempt messages (which were never folded)
            # AND any messages folded in a prior turn but not yet summarized.
            summaries = self._summaries.get(thread_id, [])
            tail = [
                m for m in history
                if getattr(m, "id", None) not in running.summarized_message_ids
            ]
            folded_messages = list(summaries) + tail
        else:
            folded_messages = list(history)

        tokens_after = self.token_counter(folded_messages)
        # Build a DigestEntry view for A3 compatibility (single summary text
        # = concatenation of all appended summaries; replaced_raw_ids from
        # the msg→checkpoint map).
        digest = None
        if running is not None:
            replaced = self._store._resolve_checkpoint_ids(
                thread_id, running.summarized_message_ids
            )
            digest = DigestEntry(
                digest_id=_make_digest_id(thread_id, sorted(running.summarized_message_ids)),
                thread_id=thread_id,
                summary=" || ".join(
                    getattr(m, "content", "")[:200] for m in self._summaries.get(thread_id, [])
                ),
                replaced_raw_ids=replaced,
                lossy_fields=["messages"],
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata={"append_summary_count": len(self._summaries.get(thread_id, []))},
                running_summary=running,
            )
        digest_added = added_this_turn
        dropped = max(0, len(history) - (len(folded_messages) - digest_added))
        # append: all accumulated summary messages are immutable across
        # subsequent folds — they keep their exact bytes forever. Their
        # combined token count is the honest "stable prefix" report.
        summaries = self._summaries.get(thread_id, [])
        stable_prefix_tokens = (
            self.token_counter(summaries) if summaries else 0
        )
        replay = ReplayOutcome(
            messages=folded_messages,
            dropped_raw_count=dropped,
            digest_messages_added=digest_added,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            strategy=self.name,
            digest=digest,
            stable_prefix_tokens=stable_prefix_tokens,
        )
        self.last_outcome[thread_id] = replay
        return replay


def _make_digest_id(thread_id: str, message_ids: Sequence[str]) -> str:
    payload = thread_id + "|" + "|".join(message_ids)
    return "digest-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def coerce_checkpoint_snapshot(snap: Any) -> dict[str, Any]:
    """Normalize a StateSnapshot or dict into a plain checkpoint dict."""
    if isinstance(snap, dict):
        return snap
    return {
        "checkpoint_id": getattr(snap, "config", {})
        .get("configurable", {})
        .get("checkpoint_id"),
        "next": getattr(snap, "next", []),
        "values": getattr(snap, "values", {}),
    }
