from __future__ import annotations
from collections import Counter
from .schema import Finding

def print_report(findings: list[Finding]) -> int:
    counts=Counter(x.status for x in findings)
    print("\n=== BACKTEST VERIFICATION REPORT ===")
    for x in findings:
        mark=x.status
        print(f"[{mark:10}] {x.check}: {x.message}")
        if x.rows: print(f"             rows: {x.rows}")
    print("\nSummary:", " ".join(f"{k}={v}" for k,v in sorted(counts.items())))
    return 1 if counts.get("FAIL",0) else 0
