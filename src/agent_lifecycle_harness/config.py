"""Configuration loader.

Loads `config.yaml` (real keys) or `config.ci.yaml` (mock mode).
Mode is determined by the `mode:` field in the loaded yaml, NOT by env var.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_CI_PATH = Path(__file__).resolve().parents[2] / "config.ci.yaml"
_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "config.example.yaml"


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a YAML mapping: {path}")
    return data


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the active configuration dict.

    Priority: explicit `path` > `config.yaml` at repo root > `config.ci.yaml` > `config.example.yaml`.
    Mode is read from the loaded config's `mode:` field.
    """
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(_DEFAULT_PATH)
    candidates.append(_CI_PATH)
    candidates.append(_EXAMPLE_PATH)

    last_err = None
    for candidate in candidates:
        try:
            return _load_yaml(candidate)
        except ConfigError as exc:
            last_err = exc

    raise ConfigError(
        "No config found. Copy config.example.yaml to config.yaml "
        "and fill in real keys, or use --config."
    ) from last_err


def is_mock_mode(config: dict[str, Any]) -> bool:
    """True when the harness should use mock providers."""
    return config.get("mode", "real") == "mock"


def provider_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return provider block by name, or raise ConfigError."""
    providers = config.get("providers", {})
    if name not in providers:
        raise ConfigError(f"Provider '{name}' not found in config.")
    return providers[name]


def default_provider_name(config: dict[str, Any]) -> str:
    return config.get("default_provider", "")


def judge_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("judge", {})
