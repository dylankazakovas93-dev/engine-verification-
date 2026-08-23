from __future__ import annotations

import numpy as np
import pandas as pd

from nq_frozen.core import atr_for_entry, build_4h_true_range


def test_fixed_utc_4h_boundaries_and_partial_bucket_valid(make_bars):
    bars = make_bars([
        ("2024-01-02 08:10+00:00", 100, 102, 99, 101),
        ("2024-01-02 11:59+00:00", 101, 104, 100, 103),
        ("2024-01-02 12:30+00:00", 110, 112, 108, 111),
    ])
    t = build_4h_true_range(bars)
    assert len(t) == 2
    assert t.iloc[0]["source_start"] == pd.Timestamp("2024-01-02 08:00+00:00")
    assert t.iloc[0]["source_end"] == pd.Timestamp("2024-01-02 12:00+00:00")
    assert t.iloc[0]["tr"] == 5.0
    # Prior nonempty close=103, so second TR=max(4, 9, 5)=9.
    assert t.iloc[1]["tr"] == 9.0


def test_atr_not_available_before_bucket_end_but_available_exactly_at_end(make_bars):
    bars = make_bars([
        ("2024-01-02 08:00+00:00", 100, 105, 99, 104),
        ("2024-01-02 11:59+00:00", 104, 106, 103, 105),
    ])
    t = build_4h_true_range(bars)
    assert atr_for_entry(t, pd.Timestamp("2024-01-02 11:59+00:00")) is None
    row = atr_for_entry(t, pd.Timestamp("2024-01-02 12:00+00:00"))
    assert row is not None
    assert row["source_end"] == pd.Timestamp("2024-01-02 12:00+00:00")


def test_empty_bucket_carries_latest_nonempty_available_tr(make_bars):
    bars = make_bars([
        ("2024-01-02 08:00+00:00", 100, 105, 99, 104),
        ("2024-01-02 11:59+00:00", 104, 106, 103, 105),
        # 12:00-15:59 entirely absent
        ("2024-01-02 16:01+00:00", 105, 106, 104, 105),
    ])
    t = build_4h_true_range(bars)
    row = atr_for_entry(t, pd.Timestamp("2024-01-02 16:00+00:00"))
    assert row is not None
    assert row["source_start"] == pd.Timestamp("2024-01-02 08:00+00:00")


def test_next_nonempty_bucket_uses_previous_nonempty_close_across_empty_bucket(make_bars):
    bars = make_bars([
        ("2024-01-02 08:00+00:00", 100, 101, 99, 100),
        ("2024-01-02 16:00+00:00", 110, 112, 109, 111),
    ])
    t = build_4h_true_range(bars)
    assert t.iloc[1]["previous_close"] == 100
    assert t.iloc[1]["tr"] == 12
