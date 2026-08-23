#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, pandas as pd
from verifier.contracts import audit_selection_map
from verifier.report import print_report

def main():
    ap=argparse.ArgumentParser(description="Verify selected contract IDs against an upstream expected-selection map.")
    ap.add_argument("--selected",required=True)
    ap.add_argument("--expected",required=True)
    ap.add_argument("--timestamp-col",default="timestamp")
    ap.add_argument("--contract-col",default="contract_id")
    ap.add_argument("--expected-contract-col",default="expected_contract_id")
    a=ap.parse_args()
    s=pd.read_csv(a.selected); e=pd.read_csv(a.expected)
    f=audit_selection_map(s,e,timestamp_col=a.timestamp_col,contract_col=a.contract_col,expected_contract_col=a.expected_contract_col)
    raise SystemExit(print_report(f))
if __name__=="__main__": main()
