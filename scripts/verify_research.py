#!/usr/bin/env python
"""Adversarial verification of an untrusted ML / conditional-edge research adapter.

Exit codes: 0=VERIFIED, 1=FAILED, 2=INCOMPLETE / UNVERIFIED (never a pass).
There is intentionally no option that evaluates or reports lockbox performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifier.research_orchestrator import MANDATORY_FAMILIES, ResearchVerificationOptions, verify_research


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Research (ML / conditional-edge) leakage verification")
    parser.add_argument("--adapter", required=True, help="research adapter exposing events/features/targets[/fit_predict_fold]")
    parser.add_argument("--source", default=None, help="candidate source file/directory for hashing and static scan; defaults to the adapter")
    parser.add_argument("--data", default=None, help="canonical CSV or Parquet bars")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--target", default=None, help="target name (wide column or long-format target_name)")
    parser.add_argument(
        "--target-horizon", default=None,
        help="externally declared MAXIMUM label horizon: '60bars' (first 60 bars opening at/after the event) or a "
             "wall-clock duration such as '60min'. Required for label-coverage proof; absent => UNVERIFIED.",
    )
    parser.add_argument("--lockbox-start", default=None, help="YYYY-MM-DD (UTC); rows at/after are withheld from modelling")
    parser.add_argument("--mode", choices=["fast", "standard", "strong"], default="strong")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--contract-col", default=None)
    parser.add_argument(
        "--bar-interval", required=True,
        help="bar duration, e.g. 1min or 1s. Bars are OPEN-stamped; a bar stamped s is known at s + bar-interval. "
             "Also used for missing-interval counting in the data audit.",
    )
    parser.add_argument("--min-train-events", type=int, default=20)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--report-prefix", default=None)
    args = parser.parse_args()
    adapter = Path(args.adapter).resolve()
    prefix = Path(args.report_prefix).resolve() if args.report_prefix else root / "reports" / f"research_{adapter.stem}"
    report = verify_research(ResearchVerificationOptions(
        adapter=adapter,
        data=Path(args.data).resolve() if args.data else None,
        source=Path(args.source).resolve() if args.source else None,
        timestamp_col=args.timestamp_col, target=args.target, lockbox_start=args.lockbox_start,
        mode=args.mode, seed=args.seed, contract_col=args.contract_col,
        bar_interval=args.bar_interval, target_horizon=args.target_horizon, min_train_events=args.min_train_events,
        run_tests=not args.skip_tests, report_prefix=prefix,
    ))
    status = report.artifacts.get("family_status", {})
    for family in sorted(status):
        marker = "mandatory" if family in MANDATORY_FAMILIES else "heuristic"
        print(f"[{status[family]:10}] {family} ({marker})")
    for finding in report.findings:
        if finding.status == "FAIL":
            print(f"  FAIL {finding.family}/{finding.check}: {finding.message[:300]}")
    lockbox = report.artifacts.get("lockbox")
    if lockbox:
        print(f"LOCKBOX: {lockbox}")
    print(f"VERDICT: {report.verdict}")
    print(f"JSON: {prefix}.json")
    print(f"MARKDOWN: {prefix}.md")
    raise SystemExit(report.exit_code)


if __name__ == "__main__":
    main()
