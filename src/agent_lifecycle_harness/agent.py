"""Core harness wrapping a LangGraph agent with SqliteSaver."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph

from agent_lifecycle_harness.llm import LLMClient, LLMResponse, build_chat_model
from agent_lifecycle_harness.compaction import (
    CompactionStore,
    CompactionStrategy,
    DigestCompactionStrategy,
    LangmemCompactor,
    NoCompactionStrategy,
)


def _build_graph(
    llm_client: LLMClient,
    compaction_strategy: CompactionStrategy | None = None,
    model_registry: dict[str, LLMClient] | None = None,
) -> StateGraph:
    """Minimal single-node agent graph for lifecycle experiments.

    The node is strategy-agnostic: it asks the configured
    ``CompactionStrategy`` what the LLM should see, then forwards that. All
    policy (no-compaction / digest / future strategies) lives behind the
    strategy interface, so adding a policy never edits this node.

    ``model_registry`` enables per-invoke client selection: if a caller
    puts ``model`` in ``config["configurable"]``, the node looks up the
    matching LLMClient in the registry and invokes THAT instead of the
    fixed ``llm_client``. This is what makes A4's resolved config actually
    reach the LLM — flipping the resolved model flips the client that
    answers. A missing model key falls back to the fixed client so demos
    that don't pass a model (A1/A3/A6/...) behave identically to before.
    """

    strategy = compaction_strategy or NoCompactionStrategy()
    registry = model_registry or {}

    def _call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        messages = state["messages"]

        # Some providers (e.g. MiMo) reject `role: system`. Strip system
        # messages before strategy rewrite so a synthesized digest message
        # (role: user) survives, but genuine system prompts don't 422.
        filtered = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", None)
            if role == "system":
                continue
            filtered.append(m)

        # Strategy decides what the LLM sees (architecture C for digest).
        thread_id = config.get("configurable", {}).get("thread_id", "")
        decision = strategy.build_replay_context(thread_id, filtered)

        # Per-invoke client selection (A4): if the caller pinned a model via
        # config["configurable"]["model"], use the registered client for it.
        # Otherwise fall through to the harness's default fixed client.
        pinned_model = config.get("configurable", {}).get("model")
        active_client = registry.get(pinned_model, llm_client) if pinned_model else llm_client

        response: LLMResponse = active_client.invoke_sync(decision.messages)
        return {"messages": [AIMessage(content=response.content)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", _call_model)
    builder.add_edge("__start__", "agent")
    builder.add_edge("agent", END)
    return builder


class LifecycleHarness:
    """LangGraph-backed lifecycle harness using SqliteSaver.

    Isolation unit is `thread_id`. A namespaced id is recommended
    (e.g. ``user:<uid>:session:<sid>``) so accidental cross-user reuse
    becomes a visible bug rather than silent state mixing.
    """

    def __init__(
        self,
        db_path: str | Path,
        llm: LLMClient,
        judge: LLMClient,
        *,
        framework: str = "langgraph",
        compaction_store: Any | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        summarization_chat_model: Any | None = None,
        max_tokens: int = 384,
        max_tokens_before_summary: int | None = None,
        max_summary_tokens: int = 200,
        model_registry: dict[str, LLMClient] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.llm = llm
        self.judge = judge
        self.framework = framework
        # Expose the summarization chat model so demos can build their own
        # LangmemCompactor against the same model the harness would use.
        self.summarization_chat_model = summarization_chat_model
        # Per-invoke model→client registry (A4). When a caller pins a model
        # via invoke(resolved_config=...), the node uses the registered
        # client for that model instead of the fixed default `llm`. Empty
        # by default → no per-invoke selection → backward compatible.
        self.model_registry: dict[str, LLMClient] = dict(model_registry or {})
        # Per-thread serialization locks (same-thread writers serialize).
        self._thread_locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        # Compaction persistence (shared SQLite table, separate from LG's
        # checkpoint table). All CompactionStore instances at the same
        # db_path see the same rows.
        self.compaction_store = compaction_store or CompactionStore(self.db_path)

        # Strategy selection (§6.1):
        #   * caller-provided strategy → use as-is (full control)
        #   * summarization_chat_model available → DigestCompactionStrategy
        #     backed by a LangmemCompactor engine (architecture C, real
        #     folding)
        #   * otherwise → NoCompactionStrategy (baseline; used in mock mode
        #     without a real chat model, or when a caller wants the A/B
        #     reference)
        if compaction_strategy is not None:
            self.compaction_strategy = compaction_strategy
        elif summarization_chat_model is not None:
            compactor = LangmemCompactor(
                self.compaction_store,
                summarization_chat_model,
                max_tokens=max_tokens,
                max_tokens_before_summary=max_tokens_before_summary,
                max_summary_tokens=max_summary_tokens,
            )
            self.compaction_strategy = DigestCompactionStrategy(
                compactor,
                checkpoint_resolver=self._resolve_message_checkpoints,
            )
        else:
            self.compaction_strategy = NoCompactionStrategy()

        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._graph = _build_graph(
            llm, self.compaction_strategy, model_registry=self.model_registry
        ).compile(
            checkpointer=SqliteSaver(conn)
        )

    def _resolve_message_checkpoints(self, thread_id: str) -> dict[str, str | None]:
        """Map every message id in the thread's latest state to the
        checkpoint id that produced it. Feeds A3's DAG the
        message↔checkpoint edge for compaction-aware tombstone traversal.
        """
        mapping: dict[str, str | None] = {}
        for cp in self.list_checkpoints(thread_id):
            cp_id = cp.get("checkpoint_id")
            for m in cp.get("values", {}).get("messages", []):
                mid = getattr(m, "id", None) or (
                    m.get("id") if isinstance(m, dict) else None
                )
                if mid:
                    # Later checkpoints win (most recent producing cp).
                    mapping[mid] = cp_id
        return mapping

    def _thread_lock(self, thread_id: str) -> threading.Lock:
        with self._global_lock:
            if thread_id not in self._thread_locks:
                self._thread_locks[thread_id] = threading.Lock()
            return self._thread_locks[thread_id]

    def _build_inputs(self, thread_id: str, user_msg: str, *, truncate_after: int | None = None) -> dict[str, Any]:
        """Build inputs for the graph, optionally truncating history."""
        user_message = {"role": "user", "content": user_msg}
        
        if truncate_after is not None and truncate_after > 0:
            # Get current state and truncate
            state = self.get_state(thread_id)
            if state and "messages" in state:
                messages = state["messages"]
                # Keep only the last truncate_after messages
                if len(messages) > truncate_after:
                    messages = messages[-truncate_after:]
                return {"messages": messages + [user_message]}
        
        return {"messages": [user_message]}

    def invoke(
        self,
        thread_id: str,
        user_msg: str,
        *,
        config_version: str = "v1",
        truncate_after: int | None = None,
        resolved_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one turn and return the full state snapshot.

        `config_version` is stamped into every checkpoint produced by this
        invoke via LangGraph's `config["metadata"]` channel, so it persists
        across sessions and survives process restarts. Reading the state back
        with `get_state()` recovers it from checkpoint metadata, not from an
        in-memory field — that is what makes version-on-session observable
        rather than asserted-only.

        `resolved_config` (A4): the per-session resolved config dict. If it
        carries a ``model`` key, that model name is forwarded into
        ``config["configurable"]["model"]`` so the node picks the matching
        registered LLMClient (see ``model_registry``). This is the wire that
        makes a material config reload actually change the LLM answering the
        session — without it the resolved model is just a tracker label.
        """
        configurable: dict[str, Any] = {"thread_id": thread_id}
        if resolved_config is not None and "model" in resolved_config:
            configurable["model"] = resolved_config["model"]
        config: dict[str, Any] = {
            "configurable": configurable,
            # LangGraph persists config["metadata"] into each checkpoint's
            # metadata blob, so config_version survives in SQLite.
            "metadata": {"config_version": config_version},
        }
        inputs = self._build_inputs(thread_id, user_msg, truncate_after=truncate_after)
        # Same-thread writers serialize via per-thread lock.
        lock = self._thread_lock(thread_id)
        import sys
        import time as _time
        import datetime as _dt
        _t0 = _time.time()
        # Heartbeat: if the invoke takes >30s, print a marker so the run
        # doesn't look frozen. Real-LLM calls can legitimately take 20-60s
        # under gateway load; this distinguishes "slow but working" from
        # "hung" by emitting a line every 30s until the invoke returns.
        import threading as _threading
        _stop_hb = _threading.Event()
        def _heartbeat():
            i = 0
            while not _stop_hb.wait(30):
                i += 1
                print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] "
                      f"[heartbeat] invoke thread={thread_id!r} still running "
                      f"({_t0 and int(_time.time()-_t0)}s elapsed)",
                      file=sys.stderr, flush=True)
        _hb = _threading.Thread(target=_heartbeat, daemon=True)
        _hb.start()
        try:
            with lock:
                result = self._graph.invoke(inputs, config=config)
        finally:
            _stop_hb.set()
        # Attach runtime metadata for downstream inspection. The authoritative
        # copy lives in checkpoint metadata (read back via get_state); the
        # in-memory echo here is a convenience for the caller.
        result.setdefault("_harness_meta", {})
        result["_harness_meta"]["thread_id"] = thread_id
        result["_harness_meta"]["config_version"] = config_version

        return result

    def get_state(self, thread_id: str) -> dict[str, Any] | None:
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        if state is None:
            return None
        # Normalize into a plain dict for easier assertions.
        values = dict(state.values) if state.values else {}
        # config_version is read back from checkpoint metadata — the same
        # blob that was written by invoke()'s config["metadata"] channel.
        # This is the persisted source of truth, not an in-memory echo.
        meta = dict(getattr(state, "metadata", {}) or {})
        values["_harness_meta"] = {
            "thread_id": thread_id,
            "next": state.next,
            "config_version": meta.get("config_version"),
        }
        return values

    def list_checkpoints(self, thread_id: str) -> list[dict[str, Any]]:
        """List checkpoint tuples for a thread (best-effort wrapper)."""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            history = list(self._graph.get_state_history(config))
        except Exception:
            return []
        items: list[dict[str, Any]] = []
        for snap in history:
            meta = dict(getattr(snap, "metadata", {}) or {})
            items.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_id": getattr(snap, "config", {}).get(
                        "configurable", {}
                    ).get("checkpoint_id"),
                    "next": getattr(snap, "next", []),
                    "values": getattr(snap, "values", {}),
                    "config_version": meta.get("config_version"),
                }
            )
        return items
