"""Pytest configuration for agent-lifecycle-harness.

Sets the default config path to config.ci.yaml (mock mode) for all tests.
No env-var mode switching is used.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _default_config_path():
    """Ensure tests use config.ci.yaml by default."""
    import agent_lifecycle_harness.config as config_module
    config_module._DEFAULT_PATH = config_module._CI_PATH
