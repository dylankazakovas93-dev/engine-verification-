from __future__ import annotations

import numpy as np
import pandas as pd

from nq_frozen.core import causal_fracdiff, fracdiff_weights


def test_weight_recurrence_and_cutoff():
    w = fracdiff_weights(d=0.45, max_weights=100, cutoff=0.001)
    assert w[0] == 1.0
    for k in range(1, len(w)):
        expected = -w[k - 1] * (0.45 - k + 1.0) / k
        assert np.isclose(w[k], expected, rtol=0, atol=1e-15)
        assert abs(w[k]) >= 0.001
    if len(w) < 100:
        nxt = -w[-1] * (0.45 - len(w) + 1.0) / len(w)
        assert abs(nxt) < 0.001


def test_weight_cap():
    w = fracdiff_weights(d=0.45, max_weights=3, cutoff=0.0)
    assert len(w) == 3


def test_fracdiff_orientation_current_gets_w0():
    x = pd.Series([1.0, 2.0, 4.0], index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
    w = np.array([1.0, -0.5, 0.25])
    fd = causal_fracdiff(x, w)
    assert np.isnan(fd.iloc[0])
    assert np.isnan(fd.iloc[1])
    assert fd.iloc[2] == 4.0 - 0.5 * 2.0 + 0.25 * 1.0
