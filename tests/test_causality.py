from __future__ import annotations

import numpy as np
import pandas as pd

from nq_frozen.core import build_features


def synthetic(n=500):
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1min", tz="UTC")
    close = 15000 + np.cumsum(np.sin(np.arange(n) / 17.0) * 0.5 + 0.1)
    return pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.ones(n),
    }, index=idx)


def test_future_append_cannot_change_past_features():
    bars = synthetic(500)
    prefix = bars.iloc[:350].copy()
    a = build_features(prefix)
    b = build_features(bars).iloc[:350]
    for col in ["fracdiff", "z", "ema60", "mean_range60", "keltner", "raw_long_signal", "signal_passes_keltner"]:
        if a[col].dtype == bool:
            assert a[col].equals(b[col])
        else:
            assert np.allclose(a[col].to_numpy(), b[col].to_numpy(), equal_nan=True, rtol=0, atol=1e-12)


def test_mutating_future_cannot_change_past_features():
    bars = synthetic(500)
    cutoff = 350
    a = build_features(bars).iloc[:cutoff]
    mutated = bars.copy()
    mutated.iloc[cutoff:, mutated.columns.get_loc("close")] *= 3
    mutated.iloc[cutoff:, mutated.columns.get_loc("open")] = mutated.iloc[cutoff:]["close"]
    mutated.iloc[cutoff:, mutated.columns.get_loc("high")] = mutated.iloc[cutoff:]["close"] + 10
    mutated.iloc[cutoff:, mutated.columns.get_loc("low")] = mutated.iloc[cutoff:]["close"] - 10
    b = build_features(mutated).iloc[:cutoff]
    for col in ["fracdiff", "z", "ema60", "mean_range60", "keltner", "raw_long_signal", "signal_passes_keltner"]:
        if a[col].dtype == bool:
            assert a[col].equals(b[col])
        else:
            assert np.allclose(a[col].to_numpy(), b[col].to_numpy(), equal_nan=True, rtol=0, atol=1e-12)
