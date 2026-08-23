#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from verifier.data import audit_dataset
from verifier.report import print_report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--timestamp-col",default="timestamp")
    ap.add_argument("--contract-col",default=None)
    ap.add_argument("--symbol-col",default=None)
    ap.add_argument("--expected-interval",default=None)
    ap.add_argument("--allow-nonpositive-prices",action="store_true")
    ap.add_argument("--manifest-out",default=None)
    a=ap.parse_args()
    manifest, findings = audit_dataset(
        a.data, timestamp_col=a.timestamp_col, expected_interval=a.expected_interval,
        contract_col=a.contract_col, symbol_col=a.symbol_col,
        allow_nonpositive_prices=a.allow_nonpositive_prices,
    )
    if a.manifest_out:
        manifest.write(a.manifest_out)
    raise SystemExit(print_report(findings))
if __name__=="__main__": main()
