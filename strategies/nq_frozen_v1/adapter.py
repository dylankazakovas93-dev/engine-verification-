from __future__ import annotations
import pandas as pd
from nq_frozen.core import run_backtest

def run(bars: pd.DataFrame) -> pd.DataFrame:
    trades, _features = run_backtest(bars)
    out=trades.copy()
    if "direction" not in out:
        out["direction"]="long"
    return out
