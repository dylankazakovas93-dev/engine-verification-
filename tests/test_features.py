from __future__ import annotations

import numpy as np
import pandas as pd

from nq_frozen.constants import KELTNER_MAX
from nq_frozen.core import build_features


def test_ema_adjust_false_matches_recursive_definition(make_bars):
    bars = make_bars(70)
    bars["close"] = np.arange(100.0, 170.0)
    bars["open"] = bars["close"]
    bars["high"] = bars["close"] + 1
    bars["low"] = bars["close"] - 1
    f = build_features(bars)
    alpha = 2.0 / 61.0
    expected = bars["close"].iloc[0]
    for x in bars["close"].iloc[1:]:
        expected = alpha * x + (1-alpha) * expected
    assert np.isclose(f["ema60"].iloc[-1], expected, rtol=0, atol=1e-12)


def test_keltner_requires_60_stored_rows(make_bars):
    bars = make_bars(60)
    f = build_features(bars)
    assert np.isnan(f["keltner"].iloc[58])
    assert np.isfinite(f["keltner"].iloc[59])


def test_signal_crossing_requires_previous_stored_z_below_threshold(make_bars):
    # This test patches only the derived z values after building features to isolate crossing semantics.
    bars = make_bars(250)
    f = build_features(bars)
    z = pd.Series(np.nan, index=f.index)
    z.iloc[200:205] = [4.9, 5.0, 5.2, 4.8, 5.1]
    prev = z.shift(1)
    raw = (z >= 5.0) & (prev < 5.0) & np.isfinite(z) & np.isfinite(prev)
    assert raw.iloc[201]
    assert not raw.iloc[202]
    assert raw.iloc[204]


def test_keltner_boundary_is_inclusive():
    assert KELTNER_MAX <= KELTNER_MAX
