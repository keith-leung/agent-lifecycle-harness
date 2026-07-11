"""A4 — Config hot-reload with version-on-session.

Ongoing sessions continue on the config version they started with.
New sessions pick up the current version. The app layer owns version
tracking; LangGraph does not provide config-version scoping.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionConfig:
    session_id: str
    config_version: str
    started_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigVersionReport:
    version: str
    active_sessions: list[SessionConfig]
    new_sessions_after_reload: list[SessionConfig]


class ConfigVersionTracker:
    """App-owned config-version tracker.

    LangGraph sessions are identified by thread_id. We map thread_id to
    the config version that was active when the session started.
    """

    def __init__(self) -> None:
        self._current_version = "v1"
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionConfig] = {}

    def set_version(self, version: str) -> None:
        with self._lock:
            self._current_version = version

    def get_version(self) -> str:
        with self._lock:
            return self._current_version

    def register_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> SessionConfig:
        with self._lock:
            version = self._current_version
            import datetime
            cfg = SessionConfig(
                session_id=session_id,
                config_version=version,
                started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                metadata=metadata or {},
            )
            self._sessions[session_id] = cfg
            return cfg

    def get_session(self, session_id: str) -> SessionConfig | None:
        with self._lock:
            return self._sessions.get(session_id)

    def active_sessions(self) -> list[SessionConfig]:
        with self._lock:
            return list(self._sessions.values())
