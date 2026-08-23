from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import run_backtest, summarize
from .io import load_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen NQ strategy backtest")
    p.add_argument("csv")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--trades-out", default="trades.csv")
    args = p.parse_args()

    bars = load_csv(args.csv, args.timestamp_col)
    trades, _ = run_backtest(bars)
    out = Path(args.trades_out)
    trades.to_csv(out, index=False)
    print(json.dumps(summarize(trades), indent=2, allow_nan=True))
    print(f"trade ledger: {out}")


if __name__ == "__main__":
    main()
