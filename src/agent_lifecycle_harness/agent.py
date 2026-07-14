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

from agent_lifecycle_harness.llm import LLMClient, LLMResponse
from agent_lifecycle_harness.compaction import CompactionStore, CheckpointCompactor


def _build_graph(llm_client: LLMClient, compaction_store: Any | None = None) -> StateGraph:
    """Minimal single-node agent graph for lifecycle experiments."""

    def _call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        messages = state["messages"]
        
        # Read compaction digests from the store inside the graph node.
        # This makes compaction a first-class graph concern rather than a
        # harness-layer side effect.
        prefix = ""
        if compaction_store is not None:
            thread_id = config.get("configurable", {}).get("thread_id", "")
            digests = compaction_store.digests_for_thread(thread_id)
            if digests:
                digest_text = "\n".join(
                    f"[DIGEST {d.digest_id}]: {d.summary}" for d in digests
                )
                prefix = f"Context digests:\n{digest_text}\n\n"
        
        # Some providers (e.g. MiMo) reject `role: system`. Strip it here.
        filtered = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", None)
            if role == "system":
                continue
            filtered.append(m)
        
        # Inject digest context into the first message if present.
        if prefix and filtered:
            first = filtered[0]
            if hasattr(first, "content"):
                first.content = f"{prefix}{first.content}"
            elif isinstance(first, dict):
                first["content"] = f"{prefix}{first.get('content', '')}"
        
        response: LLMResponse = llm_client.invoke_sync(filtered)
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
        compaction_threshold: int = 0,
    ) -> None:
        self.db_path = Path(db_path)
        self.llm = llm
        self.judge = judge
        self.framework = framework
        self.compaction_store = compaction_store
        self.compaction_threshold = compaction_threshold
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._graph = _build_graph(llm, self.compaction_store).compile(
            checkpointer=SqliteSaver(conn)
        )
        # Per-thread serialization locks (same-thread writers serialize).
        self._thread_locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        # Auto-compaction infrastructure: if threshold > 0, create a
        # CompactionStore backed by the same DB and a CheckpointCompactor
        # that uses the judge for digest generation.
        if self.compaction_threshold > 0 and self.compaction_store is None:
            self.compaction_store = CompactionStore(self.db_path)
            self._compactor = CheckpointCompactor(
                self.compaction_store, self.judge, first_last_n=3
            )
        else:
            self._compactor = None

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
    ) -> dict[str, Any]:
        """Run one turn and return the full state snapshot."""
        config = {"configurable": {"thread_id": thread_id}}
        inputs = self._build_inputs(thread_id, user_msg, truncate_after=truncate_after)
        # Same-thread writers serialize via per-thread lock.
        lock = self._thread_lock(thread_id)
        with lock:
            result = self._graph.invoke(inputs, config=config)
        # Attach runtime metadata for downstream inspection.
        result.setdefault("_harness_meta", {})
        result["_harness_meta"]["thread_id"] = thread_id
        result["_harness_meta"]["config_version"] = config_version

        # Auto-compaction: if checkpoint count exceeds threshold, compact
        # the middle segment into a digest entry.
        if self.compaction_threshold > 0 and self._compactor is not None:
            checkpoints = self.list_checkpoints(thread_id)
            if len(checkpoints) > self.compaction_threshold:
                self._compactor.compact(thread_id, checkpoints)

        return result

    def get_state(self, thread_id: str) -> dict[str, Any] | None:
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        if state is None:
            return None
        # Normalize into a plain dict for easier assertions.
        values = dict(state.values) if state.values else {}
        values["_harness_meta"] = {
            "thread_id": thread_id,
            "next": state.next,
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
            items.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_id": getattr(snap, "config", {}).get(
                        "configurable", {}
                    ).get("checkpoint_id"),
                    "next": getattr(snap, "next", []),
                    "values": getattr(snap, "values", {}),
                }
            )
        return items
