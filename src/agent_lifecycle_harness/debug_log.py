"""Debug logging for agent-lifecycle-harness.

Writes structured text logs to `debug.log` in the repo root so failures
are traceable after the fact.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any


_LOG_PATH = Path(__file__).resolve().parents[2] / "debug.log"


def _write(event: str, payload: dict[str, Any]) -> None:
    line = (
        f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} "
        f"[{event}] {payload}"
    )
    with _LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def log_llm_call(
    provider: str,
    model: str,
    ok: bool,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
    messages_preview: str = "",
) -> None:
    _write("llm_call", {
        "provider": provider,
        "model": model,
        "ok": ok,
        "error_code": error_code,
        "error_message": error_message,
        "messages_preview": messages_preview[:200],
    })


def log_vendor_fallback(
    from_provider: str,
    to_provider: str,
    reason: str,
) -> None:
    _write("vendor_fallback", {
        "from_provider": from_provider,
        "to_provider": to_provider,
        "reason": reason,
    })


def log_balance_error(
    provider: str,
    model: str,
    error_message: str,
) -> None:
    _write("balance_error", {
        "provider": provider,
        "model": model,
        "error_message": error_message,
    })
