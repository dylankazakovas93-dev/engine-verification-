#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, pandas as pd
from verifier.adapter import load_adapter
from verifier.schema import Finding
from verifier.data import audit_dataset
from verifier.ledger import audit_ledger
from verifier.causality import audit_adapter_causality
from verifier.orchestrator import load_candidate_bars
from verifier.static_scan import scan_source
from verifier.report import print_report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--adapter",required=True)
    ap.add_argument("--source",default=None)
    ap.add_argument("--timestamp-col",default="timestamp")
    ap.add_argument("--contract-col",default=None)
    ap.add_argument("--tick-size",type=float,default=None)
    ap.add_argument("--allow-same-timestamp-reentry",action="store_true")
    ap.add_argument("--audit-mode",choices=["fast","standard","strong"],default="strong")
    a=ap.parse_args()
    _manifest, findings=audit_dataset(a.data,timestamp_col=a.timestamp_col,contract_col=a.contract_col)
    if any(x.status=="FAIL" for x in findings):
        raise SystemExit(print_report(findings))
    bars=load_candidate_bars(Path(a.data),a.timestamp_col)
    mod=load_adapter(a.adapter)
    trades=mod.run(bars)
    if not isinstance(trades,pd.DataFrame):
        findings.append(Finding("adapter","FAIL","adapter run() did not return DataFrame"))
    else:
        findings += audit_ledger(trades,strict_same_timestamp_reentry=not a.allow_same_timestamp_reentry,tick_size=a.tick_size)
        causal, coverage = audit_adapter_causality(mod,bars,mode=a.audit_mode)
        findings += causal
        print("CAUSALITY COVERAGE:",json.dumps(coverage.to_dict(),indent=2))
    if a.source:
        findings += scan_source(a.source)
    raise SystemExit(print_report(findings))
if __name__=="__main__": main()
