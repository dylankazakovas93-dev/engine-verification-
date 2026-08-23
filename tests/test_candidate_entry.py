from __future__ import annotations

import numpy as np
import pandas as pd

from nq_frozen.core import build_candidates


def base_features(index):
    n = len(index)
    return pd.DataFrame({
        "signal_passes_keltner": [False] * n,
        "z": [0.0] * n,
        "keltner": [0.0] * n,
    }, index=index)


def test_next_stored_row_used_even_with_missing_minutes():
    # 10:00 and 10:03 ET, outside London and both before 15 ET.
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 10:00", tz="America/New_York").tz_convert("UTC"),
        pd.Timestamp("2024-01-02 10:03", tz="America/New_York").tz_convert("UTC"),
    ])
    f = base_features(idx)
    f.iloc[0, f.columns.get_loc("signal_passes_keltner")] = True
    f.iloc[0, f.columns.get_loc("z")] = 5.1
    f.iloc[0, f.columns.get_loc("keltner")] = 1.0
    # previous_z lookup is only valid for real signals after warmup, so add prior stored row.
    prior = idx[0] - pd.Timedelta(minutes=1)
    f2 = pd.concat([base_features(pd.DatetimeIndex([prior])), f])
    f2.iloc[1, f2.columns.get_loc("signal_passes_keltner")] = True
    f2.iloc[0, f2.columns.get_loc("z")] = 4.9
    f2.iloc[1, f2.columns.get_loc("z")] = 5.1
    f2.iloc[1, f2.columns.get_loc("keltner")] = 1.0
    c = build_candidates(f2)
    assert len(c) == 1
    assert c[0].entry_time == idx[1]


def test_signal_not_carried_across_1500_cutoff():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 14:58", tz="America/New_York").tz_convert("UTC"),
        pd.Timestamp("2024-01-02 14:59", tz="America/New_York").tz_convert("UTC"),
        pd.Timestamp("2024-01-02 18:00", tz="America/New_York").tz_convert("UTC"),
    ])
    f = base_features(idx)
    f.iloc[0, f.columns.get_loc("z")] = 4.9
    f.iloc[1, f.columns.get_loc("z")] = 5.1
    f.iloc[1, f.columns.get_loc("keltner")] = 1.0
    f.iloc[1, f.columns.get_loc("signal_passes_keltner")] = True
    assert build_candidates(f) == []
