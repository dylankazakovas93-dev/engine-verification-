from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def make_bars():
    def _make(rows, start="2024-01-02 14:00:00+00:00", freq="1min"):
        if isinstance(rows, int):
            idx = pd.date_range(start, periods=rows, freq=freq, tz="UTC" if "+" not in start and "Z" not in start else None)
            data = {
                "open": [100.0] * rows,
                "high": [101.0] * rows,
                "low": [99.0] * rows,
                "close": [100.0] * rows,
                "volume": [1.0] * rows,
            }
            return pd.DataFrame(data, index=idx)
        idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        data = {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] if len(r) > 5 else 1.0 for r in rows],
        }
        return pd.DataFrame(data, index=idx)
    return _make
