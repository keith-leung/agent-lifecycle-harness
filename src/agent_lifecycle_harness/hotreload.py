"""A4 — Config hot-reload with version-on-session.

Two change classes are distinguished:

* **material** — changes a field that affects model output
  (``system_prompt``, ``model``, ``temperature``). A material change bumps
  the global version. Ongoing sessions stay pinned to the version they
  registered with; only new sessions pick up the change.
* **cosmetic** — changes a field that does not affect model output
  (``log_level``, ``metrics_endpoint``, …). A cosmetic change propagates
  immediately to all sessions, live or new, because reading the merged
  config always overlays the current cosmetic fields.

Sessions also have a TTL. A background sweeper (opt-in) expires sessions
that have been inactive longer than the configured TTL, so memory does
not grow unboundedly with traffic.

The app layer owns version tracking; LangGraph does not provide
config-version scoping.
"""

from __future__ import annotations

import copy
import datetime
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


# Fields that change model output → must pin to session at registration time.
# Any change here bumps the version and only affects new sessions.
MATERIAL_KEYS: frozenset[str] = frozenset({"system_prompt", "model", "temperature"})

# Fields that don't affect model output → safe to propagate to live sessions.
# Listed explicitly so the classifier is auditable; ``COSMETIC_KEYS`` is the
# known set, unknown keys default to material (fail-safe: treat unknown as
# behavior-affecting rather than silently hot-patching it).
COSMETIC_KEYS: frozenset[str] = frozenset({
    "log_level", "metrics_endpoint", "trace_sample_rate", "telemetry_tags",
})


@dataclass(frozen=True)
class ChangeClassification:
    """Result of classifying a config diff.

    `kind` is one of ``"material"``, ``"cosmetic"``, ``"no_change"``.
    `changed_material` / `changed_cosmetic` are the concrete field names that
    flipped — surfaced so callers can log *why* a version bumped.
    """
    kind: str
    changed_material: tuple[str, ...]
    changed_cosmetic: tuple[str, ...]


class ChangeClassifier:
    """Decide whether a config diff forces a version bump.

    Fail-safe policy: any key outside both ``MATERIAL_KEYS`` and
    ``COSMETIC_KEYS`` is treated as material. Better to over-pin than to
    hot-patch an unknown field into a live session.
    """

    def __init__(
        self,
        material_keys: frozenset[str] = MATERIAL_KEYS,
        cosmetic_keys: frozenset[str] = COSMETIC_KEYS,
    ) -> None:
        self.material_keys = material_keys
        self.cosmetic_keys = cosmetic_keys

    def classify(self, old: dict[str, Any], new: dict[str, Any]) -> ChangeClassification:
        changed_material: list[str] = []
        changed_cosmetic: list[str] = []
        # Union of keys present on either side — a key disappearing is itself
        # a change.
        for key in set(old) | set(new):
            if old.get(key) == new.get(key):
                continue
            if key in self.cosmetic_keys:
                changed_cosmetic.append(key)
            else:
                # material includes explicitly-listed material keys AND any
                # unknown key (fail-safe).
                changed_material.append(key)
        if changed_material:
            kind = "material"
        elif changed_cosmetic:
            kind = "cosmetic"
        else:
            kind = "no_change"
        return ChangeClassification(
            kind=kind,
            changed_material=tuple(sorted(changed_material)),
            changed_cosmetic=tuple(sorted(changed_cosmetic)),
        )

    def is_material(self, old: dict[str, Any], new: dict[str, Any]) -> bool:
        return self.classify(old, new).kind == "material"


@dataclass
class SessionConfig:
    """Per-session frozen material config + view of current cosmetic config.

    `config_version` is the version label at registration time. `material`
    is the deep-copied material slice of the global config at that moment,
    frozen for the session's lifetime. Cosmetic fields are *not* frozen —
    they're overlaid at read time via `resolved_config()`.

    `started_at` / `last_active_at` are ISO-8601 UTC timestamps. TTL
    expiry is computed against `last_active_at`, refreshed on every
    `touch_session()` so an active long-lived session is never reaped.
    `ended_at` is set when the session is explicitly ended or expired;
    `None` means the session is live.
    """
    session_id: str
    config_version: str
    started_at: str
    material: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_active_at: str = ""
    ended_at: str | None = None

    def __post_init__(self) -> None:
        if not self.last_active_at:
            self.last_active_at = self.started_at

    def is_active(self) -> bool:
        return self.ended_at is None

    def resolved_config(self, cosmetic_view: dict[str, Any]) -> dict[str, Any]:
        """Merge this session's frozen material config with live cosmetic fields.

        Caller passes the tracker's current cosmetic view; we overlay it on
        top of the frozen material slice. This is the actual config the
        session sees — frozen where it matters, live where it doesn't.
        """
        merged = dict(self.material)
        for key in COSMETIC_KEYS:
            if key in cosmetic_view:
                merged[key] = cosmetic_view[key]
        return merged


@dataclass
class ConfigVersionReport:
    version: str
    active_sessions: list[SessionConfig]
    new_sessions_after_reload: list[SessionConfig]


class ConfigVersionTracker:
    """App-owned config-version tracker with material/cosmetic classification.

    LangGraph sessions are identified by thread_id. We map thread_id to the
    config version that was active when the session started.

    Contract:
    * `set_config(new_config)` — classify diff vs current global config.
      Material → bump version (new sessions get it; live sessions don't).
      Cosmetic → update global cosmetic fields in place (live + new see it
      immediately). Returns the classification so callers can log/audit.
    * `set_version(label)` — backward-compat: bump version unconditionally
      with an empty material slice (used by the legacy A4 demo path).
    * `register_session(sid)` — snapshot the *material* config + current
      version label into a SessionConfig.
    * `touch_session(sid)` — refresh `last_active_at`; called by the harness
      layer on every successful turn so an active session is never reaped.
    * `end_session(sid)` — explicit release (user closes chat, etc.).
    * `cleanup_expired(now=None)` — reap sessions whose `last_active_at` is
      older than `session_ttl_seconds`. Returns the reaped session_ids.
    * `start_sweeper(...)` / `stop_sweeper()` — opt-in background thread
      that periodically calls `cleanup_expired`. Disabled by default.
    """

    def __init__(
        self,
        *,
        initial_config: dict[str, Any] | None = None,
        classifier: ChangeClassifier | None = None,
        initial_version: str = "v1",
        session_ttl_seconds: float | None = None,
        on_session_expired: Callable[[SessionConfig], None] | None = None,
    ) -> None:
        self._current_version = initial_version
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionConfig] = {}
        self._classifier = classifier or ChangeClassifier()
        # Deep copy so external mutation can't corrupt our state.
        cfg = copy.deepcopy(initial_config or {})
        self._global_config: dict[str, Any] = cfg
        # Cached slices, recomputed on every set_config.
        self._material_view = self._slice_material(self._global_config)
        self._cosmetic_view = self._slice_cosmetic(self._global_config)
        # TTL: None means never expire (backward-compat default).
        self._session_ttl_seconds = session_ttl_seconds
        self._on_session_expired = on_session_expired
        # Background sweeper state.
        self._sweeper_thread: threading.Thread | None = None
        self._sweeper_stop: threading.Event | None = None

    # ------------------------------------------------------------------ slices

    def _slice_material(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {k: copy.deepcopy(v) for k, v in cfg.items() if k in MATERIAL_KEYS}

    def _slice_cosmetic(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {k: copy.deepcopy(v) for k, v in cfg.items() if k in COSMETIC_KEYS}

    # ------------------------------------------------------------------ writes

    def set_config(self, new_config: dict[str, Any]) -> ChangeClassification:
        """Apply a config update with material/cosmetic classification.

        Material change → bump `_current_version` and refresh the frozen
        material view; cosmetic change → only refresh the cosmetic view;
        no_change → no-op. Returns the classification.
        """
        with self._lock:
            new_config = copy.deepcopy(new_config)
            classification = self._classifier.classify(self._global_config, new_config)
            if classification.kind == "no_change":
                return classification
            # Always update the full global config so subsequent diffs are
            # computed against the latest state.
            self._global_config = new_config
            if classification.kind == "material":
                # Bump version. New registrations pin this version; existing
                # sessions keep their frozen material slice untouched.
                self._current_version = self._bump_version(self._current_version)
                self._material_view = self._slice_material(new_config)
            # Cosmetic: refresh the live overlay; SessionConfig.resolved_config
            # picks it up next read. No version bump.
            self._cosmetic_view = self._slice_cosmetic(new_config)
            return classification

    def set_version(self, version: str) -> None:
        """Legacy unconditional version bump (backward compat for old demos).

        Equivalent to declaring a material change without specifying the
        diff — the version label moves but no config slice changes. New
        code should prefer `set_config()`.
        """
        with self._lock:
            self._current_version = version

    @staticmethod
    def _bump_version(current: str) -> str:
        """v<N> → v<N+1>; non-v-label → append numeric suffix."""
        prefix = "v"
        if current.startswith(prefix) and current[1:].isdigit():
            return f"{prefix}{int(current[1:]) + 1}"
        return f"{current}_bumped"

    # ------------------------------------------------------------------ reads

    def get_version(self) -> str:
        with self._lock:
            return self._current_version

    def get_global_config(self) -> dict[str, Any]:
        """Return a deep copy of the current global config (audit / display)."""
        with self._lock:
            return copy.deepcopy(self._global_config)

    def get_material_view(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._material_view)

    def get_cosmetic_view(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._cosmetic_view)

    # ------------------------------------------------------------ registration

    def register_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SessionConfig:
        """Snapshot material config + current version label for a new session."""
        with self._lock:
            cfg = SessionConfig(
                session_id=session_id,
                config_version=self._current_version,
                started_at=_now_iso(),
                material=copy.deepcopy(self._material_view),
                metadata=metadata or {},
            )
            self._sessions[session_id] = cfg
            return cfg

    def touch_session(self, session_id: str) -> bool:
        """Refresh `last_active_at` for a session.

        Should be called by the harness layer after every successful turn
        on the session. Returns True if the session existed (and was live);
        False if it was already gone (ended or expired). Touching an ended
        session does not revive it.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.is_active():
                return False
            session.last_active_at = _now_iso()
            return True

    def end_session(self, session_id: str) -> bool:
        """Explicitly release a session (user closed chat, logout, etc.).

        Marks `ended_at`; the SessionConfig is retained in `_sessions` so
        audit trails remain readable via `get_session()`, but
        `is_active()` returns False and TTL sweep will skip it. Returns
        True if a live session was ended.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.is_active():
                return False
            session.ended_at = _now_iso()
            return True

    # ------------------------------------------------------------------- TTL

    def cleanup_expired(self, *, now: str | None = None) -> list[str]:
        """Reap sessions whose inactivity exceeds `session_ttl_seconds`.

        Returns the list of reaped session_ids (in deterministic insertion
        order). Sessions already ended are left alone (audit retention).
        No-op if TTL is None. Fires `on_session_expired` per reaped session
        *outside* the lock so callbacks can't deadlock on tracker APIs.
        """
        if self._session_ttl_seconds is None:
            return []
        now_ts = _parse_iso(now) if now else _now_utc()
        threshold = now_ts - datetime.timedelta(seconds=self._session_ttl_seconds)
        reaped: list[str] = []
        reaped_sessions: list[SessionConfig] = []
        with self._lock:
            for sid, session in self._sessions.items():
                if not session.is_active():
                    continue
                last_ts = _parse_iso(session.last_active_at)
                if last_ts < threshold:
                    session.ended_at = _now_iso_from(now_ts)
                    reaped.append(sid)
                    reaped_sessions.append(session)
        # Fire callbacks outside the lock to avoid reentrancy deadlock.
        if self._on_session_expired is not None:
            for session in reaped_sessions:
                try:
                    self._on_session_expired(session)
                except Exception:
                    # Callback failure must not break the sweep.
                    pass
        return reaped

    def start_sweeper(
        self,
        *,
        interval_seconds: float = 60.0,
    ) -> None:
        """Start a daemon thread that periodically calls `cleanup_expired`.

        Idempotent: calling twice without `stop_sweeper()` is a no-op.
        Production deployments should call this once at app startup.
        """
        if self._session_ttl_seconds is None:
            raise RuntimeError(
                "Cannot start sweeper without a TTL; pass session_ttl_seconds "
                "to ConfigVersionTracker."
            )
        with self._lock:
            if self._sweeper_thread is not None and self._sweeper_thread.is_alive():
                return
            self._sweeper_stop = threading.Event()
            interval = max(interval_seconds, 0.1)

            def _loop(stop: threading.Event, interval_sec: float) -> None:
                while not stop.wait(interval_sec):
                    try:
                        self.cleanup_expired()
                    except Exception:
                        # Sweeper must be resilient; failures get retried.
                        pass

            thread = threading.Thread(
                target=_loop,
                args=(self._sweeper_stop, interval),
                name="ConfigVersionTracker-sweeper",
                daemon=True,
            )
            thread.start()
            self._sweeper_thread = thread

    def stop_sweeper(self, *, timeout: float | None = 1.0) -> bool:
        """Signal the sweeper to stop and join. Returns True if it joined."""
        with self._lock:
            thread = self._sweeper_thread
            stop = self._sweeper_stop
        if thread is None:
            return True
        if stop is not None:
            stop.set()
        thread.join(timeout=timeout)
        joined = not thread.is_alive()
        with self._lock:
            if self._sweeper_thread is thread:
                self._sweeper_thread = None
                self._sweeper_stop = None
        return joined

    def get_session(self, session_id: str) -> SessionConfig | None:
        with self._lock:
            return self._sessions.get(session_id)

    def active_sessions(self) -> list[SessionConfig]:
        """Return all live sessions (excludes ended/expired)."""
        with self._lock:
            return [s for s in self._sessions.values() if s.is_active()]

    def all_sessions(self) -> list[SessionConfig]:
        """Return all known sessions, including ended/expired (audit view)."""
        with self._lock:
            return list(self._sessions.values())

    def resolved_config_for(self, session_id: str) -> dict[str, Any] | None:
        """The effective config a session sees right now.

        Frozen material slice (from registration time) overlaid with the
        current live cosmetic view. None if the session isn't registered.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.resolved_config(self._cosmetic_view)


# ------------------------------------------------------------- time helpers

def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _now_iso_from(ts: datetime.datetime) -> str:
    return ts.isoformat()


def _parse_iso(s: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp; naive timestamps assumed UTC."""
    ts = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    return ts
