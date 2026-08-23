from __future__ import annotations

import pandas as pd
import pytest

from nq_frozen.core import validate_bars


def test_missing_minutes_allowed(make_bars):
    bars = make_bars([
        ("2024-01-02 10:00+00:00", 100, 101, 99, 100),
        ("2024-01-02 10:03+00:00", 100, 101, 99, 100),
    ])
    assert len(validate_bars(bars)) == 2


def test_duplicates_rejected(make_bars):
    bars = make_bars([
        ("2024-01-02 10:00+00:00", 100, 101, 99, 100),
        ("2024-01-02 10:00+00:00", 100, 101, 99, 100),
    ])
    with pytest.raises(ValueError, match="duplicate"):
        validate_bars(bars)


def test_unsorted_rejected(make_bars):
    bars = make_bars([
        ("2024-01-02 10:01+00:00", 100, 101, 99, 100),
        ("2024-01-02 10:00+00:00", 100, 101, 99, 100),
    ])
    with pytest.raises(ValueError, match="chronological"):
        validate_bars(bars)


def test_naive_timestamps_rejected():
    idx = pd.date_range("2024-01-01", periods=2, freq="1min")
    bars = pd.DataFrame({"open":[1,1],"high":[1,1],"low":[1,1],"close":[1,1],"volume":[1,1]}, index=idx)
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_bars(bars)


def test_bad_ohlc_rejected(make_bars):
    bars = make_bars([("2024-01-02 10:00+00:00", 102, 101, 99, 100)])
    with pytest.raises(ValueError, match="open outside"):
        validate_bars(bars)
