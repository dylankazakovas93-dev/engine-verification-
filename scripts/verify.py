#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifier.orchestrator import VerificationOptions, verify


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="One-command adversarial trading-engine verification")
    parser.add_argument("--strategy", required=True, help="strategy directory name under strategies/")
    parser.add_argument("--candidate", default=None, help="candidate adapter path; defaults to strategy adapter.py")
    parser.add_argument("--source", default=None, help="candidate source file/directory for hashing and static scan")
    parser.add_argument("--data", default=None, help="canonical strategy CSV or Parquet")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--contract-col", default=None)
    parser.add_argument("--selection-map", default=None)
    parser.add_argument("--expected-contract-col", default="expected_contract_id")
    parser.add_argument("--high-res-data", default=None)
    parser.add_argument("--high-res-timestamp-col", default="ts_event")
    parser.add_argument("--high-res-contract-col", default="instrument_id")
    parser.add_argument("--high-res-symbol-col", default="symbol")
    parser.add_argument("--audit-mode", choices=["fast", "standard", "strong"], default=None)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--report-prefix", default=None)
    args = parser.parse_args()
    strategy_dir = root / "strategies" / args.strategy
    candidate = Path(args.candidate).resolve() if args.candidate else strategy_dir / "adapter.py"
    report = verify(VerificationOptions(
        strategy_dir=strategy_dir, candidate_adapter=candidate,
        candidate_source=Path(args.source).resolve() if args.source else None,
        data=Path(args.data).resolve() if args.data else None,
        timestamp_col=args.timestamp_col, contract_col=args.contract_col,
        selection_map=Path(args.selection_map).resolve() if args.selection_map else None,
        expected_contract_col=args.expected_contract_col,
        high_res_data=Path(args.high_res_data).resolve() if args.high_res_data else None,
        high_res_timestamp_col=args.high_res_timestamp_col,
        high_res_contract_col=args.high_res_contract_col,
        high_res_symbol_col=args.high_res_symbol_col,
        audit_mode=args.audit_mode, run_tests=not args.skip_tests,
        report_prefix=Path(args.report_prefix).resolve() if args.report_prefix else None,
    ))
    print(f"VERDICT: {report.verdict}")
    print(f"JSON: {(Path(str(args.report_prefix)).resolve() if args.report_prefix else root / 'reports' / args.strategy).with_suffix('.json')}")
    print(f"MARKDOWN: {(Path(str(args.report_prefix)).resolve() if args.report_prefix else root / 'reports' / args.strategy).with_suffix('.md')}")
    raise SystemExit(report.exit_code)


if __name__ == "__main__":
    main()
