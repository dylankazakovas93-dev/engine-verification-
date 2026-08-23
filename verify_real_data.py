from __future__ import annotations

import argparse
import json

from nq_frozen.io import load_csv
from nq_frozen.reconcile import reconcile
from nq_frozen.core import run_backtest, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--timestamp-col", default="timestamp")
    args = p.parse_args()
    bars = load_csv(args.csv, args.timestamp_col)
    result = reconcile(bars)
    trades, _ = run_backtest(bars)
    print(json.dumps({"reconciliation": result, "summary": summarize(trades)}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
