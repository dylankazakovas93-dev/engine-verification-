#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, pandas as pd
from verifier.adapter import load_adapter
from verifier.schema import audit_bars, Finding
from verifier.ledger import audit_ledger
from verifier.causality import audit_causality
from verifier.static_scan import scan_source
from verifier.report import print_report

def load_bars(path,timestamp_col):
    df=pd.read_csv(path)
    parsed=pd.to_datetime(df[timestamp_col],errors="raise")
    if not isinstance(parsed.dtype,pd.DatetimeTZDtype):
        raise ValueError("timestamps must carry an explicit timezone")
    idx=pd.DatetimeIndex(parsed.dt.tz_convert("UTC"))
    return df.drop(columns=[timestamp_col]).set_axis(idx,axis=0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--adapter",required=True)
    ap.add_argument("--source",default=None)
    ap.add_argument("--timestamp-col",default="timestamp")
    ap.add_argument("--contract-col",default=None)
    ap.add_argument("--tick-size",type=float,default=None)
    ap.add_argument("--allow-same-timestamp-reentry",action="store_true")
    a=ap.parse_args()
    raw=pd.read_csv(a.data)
    findings=audit_bars(raw,a.timestamp_col,a.contract_col)
    if any(x.status=="FAIL" for x in findings):
        raise SystemExit(print_report(findings))
    bars=load_bars(a.data,a.timestamp_col)
    mod=load_adapter(a.adapter)
    trades=mod.run(bars)
    if not isinstance(trades,pd.DataFrame):
        findings.append(Finding("adapter","FAIL","adapter run() did not return DataFrame"))
    else:
        findings += audit_ledger(trades,strict_same_timestamp_reentry=not a.allow_same_timestamp_reentry,tick_size=a.tick_size)
        findings += audit_causality(mod.run,bars)
    if a.source:
        findings += scan_source(a.source)
    raise SystemExit(print_report(findings))
if __name__=="__main__": main()
