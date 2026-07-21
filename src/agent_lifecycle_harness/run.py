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
from agent_lifecycle_harness.llm import MultiVendorLLMClient, build_chat_model, build_client
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

    # A2 needs a langmem-compatible ChatModel for summarization + exact
    # token counting. Build it from the first real provider. In mock mode
    # this stays None and the harness falls back to NoCompactionStrategy.
    chat_model = None
    if not is_mock_mode(config):
        chat_model = build_chat_model(
            provider_names[0], tier, config=config, temperature=0.0,
        )

    # A4 model_registry: A4's load-bearing assertion proves that the
    # resolved ``model`` field actually selects the LLM client that answers
    # (not just a tracker label). The assertion needs two clients whose
    # output is distinguishable by signature. Real LLMs don't self-report
    # their model id in replies, so we register MockLLMClients keyed by
    # the same model names A4's config swap uses ("sut-v1" / "sut-v2").
    # This is an input-side / wiring proof (same shape as A3's), not a
    # real-LLM-behavior test: the question is "did the model pin reach the
    # client-selection branch?" — and a mock signature answers exactly that
    # under both mock and real modes. Only A4 gets this registry; other
    # demos pass no model and fall through to the fixed default client.
    from agent_lifecycle_harness.llm import MockLLMClient
    model_registry = None
    if db_suffix == "A4":
        model_registry = {
            "sut-v1": MockLLMClient(prefix="sut-v1"),
            "sut-v2": MockLLMClient(prefix="sut-v2"),
        }

    return a1_isolation.LifecycleHarness(
        db_path=db_path, llm=llm, judge=judge, framework="langgraph",
        summarization_chat_model=chat_model,
        model_registry=model_registry,
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
    import time as _time
    import datetime as _dt
    def _ts() -> str:
        return _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{_ts()}] [mode] {mode}  (LLM calls logged to debug.log)", flush=True)

    overall = True
    if args.all:
        for key, fn in DEMOS.items():
            t0 = _time.time()
            print(f"\n[{_ts()}] --- {key} --- starting", flush=True)
            demo_harness = None
            if not is_mock_mode(config) and key != "A5":
                demo_harness = _build_harness(config, db_suffix=key)
            result = fn(harness=demo_harness)
            ok = _print_result(result)
            overall = overall and ok
            print(f"[{_ts()}] --- {key} --- done in {_time.time()-t0:.1f}s ({'PASS' if ok else 'FAIL'})", flush=True)
    elif args.demo:
        t0 = _time.time()
        print(f"\n[{_ts()}] --- {args.demo} --- starting", flush=True)
        demo_harness = None
        if not is_mock_mode(config) and args.demo != "A5":
            demo_harness = _build_harness(config, db_suffix=args.demo)
        result = DEMOS[args.demo](harness=demo_harness)
        ok = _print_result(result)
        overall = overall and ok
        print(f"[{_ts()}] --- {args.demo} --- done in {_time.time()-t0:.1f}s ({'PASS' if ok else 'FAIL'})", flush=True)

    print(f"\n[{_ts()}] Overall: {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
