#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, pandas as pd
from verifier.schema import audit_bars
from verifier.report import print_report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--timestamp-col",default="timestamp")
    ap.add_argument("--contract-col",default=None)
    a=ap.parse_args()
    df=pd.read_csv(a.data)
    raise SystemExit(print_report(audit_bars(df,a.timestamp_col,a.contract_col)))
if __name__=="__main__": main()
