"""CLI entry point.

Usage:
    python -m agent_lifecycle_harness.run --all
    python -m agent_lifecycle_harness.run --demo A1
    python -m agent_lifecycle_harness.run --config config.ci.yaml --all
    AGENT_HARNESS_CI_MOCK=1 python -m agent_lifecycle_harness.run --all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agent_lifecycle_harness.config import load_config, is_mock_mode, provider_config
from agent_lifecycle_harness.llm import MultiVendorLLMClient, build_client
from agent_lifecycle_harness.demos import (
    a1_isolation,
    a2_compaction,
    a2a3_interop,
    a3_tombstone,
    a4_hotreload,
    a5_degradation,
    a6_migration,
    a7_cross_framework,
)
from agent_lifecycle_harness.debug_log import log_vendor_fallback

DEMOS = {
    "A1": a1_isolation.demo_A1_isolation,
    "A2": a2_compaction.demo_A2_compaction,
    "A2_A3": a2a3_interop.demo_A2_A3_interop,
    "A3": a3_tombstone.demo_A3_tombstone,
    "A4": a4_hotreload.demo_A4_hotreload,
    "A5": a5_degradation.demo_A5_degradation,
    "A6": a6_migration.demo_A6_migration,
    "A7": a7_cross_framework.demo_A7_cross_framework,
}


def _build_harness(config: dict[str, Any], db_suffix: str = "") -> a1_isolation.LifecycleHarness:
    provider_names = list(config.get("providers", {}).keys())
    if not provider_names:
        raise RuntimeError("No providers configured.")
    tier = "medium"

    # SUT: try all providers in config order. Demo code never sees individual vendors.
    llm = MultiVendorLLMClient(provider_names, tier, config)

    # Judge: try configured judge provider first, fallback to SUT chain if needed.
    judge_cfg = config.get("judge", {})
    judge_provider = judge_cfg.get("provider", provider_names[0])
    judge = MultiVendorLLMClient([judge_provider], tier, config)

    # Cross-vendor guard: if SUT and judge end up on the same vendor, warn.
    sut_names = {name for name, _ in llm._clients}
    judge_names = {name for name, _ in judge._clients}
    shared = sut_names & judge_names
    if shared:
        print(f"[WARN] SUT and judge share vendor(s): {shared}. Cross-vendor guard relaxed.")

    suffix = f"-{db_suffix}" if db_suffix else ""
    db_path = Path("runs") / f"{provider_names[0]}-{tier}{suffix}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return a1_isolation.LifecycleHarness(
        db_path=db_path, llm=llm, judge=judge, framework="langgraph"
    )


def _print_result(result: a1_isolation.DemoResult) -> bool:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name}")
    for a in result.assertions:
        mark = "  ok" if a.passed else "  FAIL"
        print(f"  {mark}: {a.name} — {a.evidence}")
    return result.passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="agent-lifecycle-harness runner")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--demo", choices=list(DEMOS.keys()), help="Run a single demo")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args(argv)

    if not args.all and not args.demo:
        parser.print_help()
        return 1

    # If CI-MOCK env var is set and no explicit --config, default to config.ci.yaml
    if args.config is None and os.environ.get("AGENT_HARNESS_CI_MOCK") == "1":
        args.config = "config.ci.yaml"

    config = load_config(args.config)
    mode = "REAL-LLM" if not is_mock_mode(config) else "CI-MOCK"
    print(f"[mode] {mode}")

    overall = True
    if args.all:
        for key, fn in DEMOS.items():
            print(f"\n--- {key} ---")
            demo_harness = None
            if not is_mock_mode(config) and key != "A5":
                demo_harness = _build_harness(config, db_suffix=key)
            result = fn(harness=demo_harness)
            ok = _print_result(result)
            overall = overall and ok
    elif args.demo:
        print(f"\n--- {args.demo} ---")
        demo_harness = None
        if not is_mock_mode(config) and args.demo != "A5":
            demo_harness = _build_harness(config, db_suffix=args.demo)
        result = DEMOS[args.demo](harness=demo_harness)
        ok = _print_result(result)
        overall = overall and ok

    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
