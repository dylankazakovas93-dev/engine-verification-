from __future__ import annotations
import pandas as pd
from nq_frozen.core import build_candidates, build_features, run_backtest
from nq_frozen.reference import run_reference

FEATURE_COLUMNS = [
    "fracdiff", "z", "ema60", "mean_range60", "keltner",
    "raw_long_signal", "signal_passes_keltner",
]

def run(bars: pd.DataFrame) -> pd.DataFrame:
    trades, _features = run_backtest(bars)
    out=trades.copy()
    if "direction" not in out:
        out["direction"]="long"
    return out


def features(bars: pd.DataFrame) -> pd.DataFrame:
    return build_features(bars)[FEATURE_COLUMNS]


def signals(bars: pd.DataFrame) -> pd.DataFrame:
    derived = build_features(bars)
    rows = derived.loc[derived["raw_long_signal"], ["z", "keltner"]].copy()
    rows.insert(0, "signal_time", rows.index)
    return rows.reset_index(drop=True)


def eligible_signals(bars: pd.DataFrame) -> pd.DataFrame:
    derived = build_features(bars)
    rows = derived.loc[derived["signal_passes_keltner"], ["z", "keltner"]].copy()
    rows.insert(0, "signal_time", rows.index)
    return rows.reset_index(drop=True)


def proposed_entries(bars: pd.DataFrame) -> pd.DataFrame:
    derived = build_features(bars)
    return pd.DataFrame([
        {"signal_time": c.signal_time, "entry_time": c.entry_time, "deadline": c.deadline}
        for c in build_candidates(derived)
    ], columns=["signal_time", "entry_time", "deadline"])


def accepted_entries(bars: pd.DataFrame) -> pd.DataFrame:
    trades = run(bars)
    columns = [c for c in ["signal_time", "entry_time", "entry_price", "direction"] if c in trades]
    return trades[columns].copy()


def reference(bars: pd.DataFrame) -> pd.DataFrame:
    trades, _features = run_reference(bars)
    if "direction" not in trades:
        trades = trades.copy()
        trades["direction"] = "long"
    return trades


def audit_stages(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    trades, derived = run_backtest(bars)
    if "direction" not in trades:
        trades = trades.copy()
        trades["direction"] = "long"
    raw = derived.loc[derived["raw_long_signal"], ["z", "keltner"]].copy()
    raw.insert(0, "signal_time", raw.index)
    eligible = derived.loc[derived["signal_passes_keltner"], ["z", "keltner"]].copy()
    eligible.insert(0, "signal_time", eligible.index)
    proposed = pd.DataFrame([
        {"signal_time": c.signal_time, "entry_time": c.entry_time, "deadline": c.deadline}
        for c in build_candidates(derived)
    ], columns=["signal_time", "entry_time", "deadline"])
    accepted_columns = [c for c in ["signal_time", "entry_time", "entry_price", "direction"] if c in trades]
    return {
        "features": derived[FEATURE_COLUMNS],
        "signals": raw.reset_index(drop=True),
        "eligible_signals": eligible.reset_index(drop=True),
        "proposed_entries": proposed,
        "accepted_entries": trades[accepted_columns].copy(),
        "trades": trades,
    }
