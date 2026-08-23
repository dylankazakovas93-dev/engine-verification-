from __future__ import annotations

import numpy as np
import pandas as pd

from nq_frozen.core import build_features
from nq_frozen.reconcile import reconcile
from nq_frozen.reference import reference_features


def realistic_synthetic(n=1200):
    # One deterministic positive shock at 16:00 UTC (11:00 ET / 10:00 CT in January)
    # produces a legal signal and at least one accepted trade.
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1min", tz="UTC")
    x = np.arange(n)
    close = 15000.0 + 0.5 * np.sin(x / 13.0)
    close[960:] += 25.0
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 5.0
    low = np.minimum(open_, close) - 5.0
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":np.ones(n)}, index=idx)


def test_reference_features_match_production():
    bars = realistic_synthetic()
    a = build_features(bars)
    b = reference_features(bars)
    for col in ["fracdiff", "ema60", "mean_range60", "keltner"]:
        assert np.allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), rtol=0, atol=1e-10, equal_nan=True)
    assert np.allclose(a["z"].to_numpy(dtype=float), b["z"].to_numpy(dtype=float), rtol=0, atol=1e-6, equal_nan=True)
    assert a["raw_long_signal"].equals(b["raw_long_signal"])
    assert a["signal_passes_keltner"].equals(b["signal_passes_keltner"])


def test_full_reconciliation_on_synthetic_dataset_with_real_trade():
    bars = realistic_synthetic()
    result = reconcile(bars)
    assert result["matched"] is True
    assert result["production_trades"] == result["reference_trades"]
    assert result["production_trades"] >= 1
    assert result["production_hash"] == result["reference_hash"]
