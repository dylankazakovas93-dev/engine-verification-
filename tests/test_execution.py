from __future__ import annotations

import pandas as pd

from nq_frozen.core import simulate_trade
from nq_frozen.models import Candidate


def candidate(entry_time: str, deadline: str, entry_idx: int = 0) -> Candidate:
    et = pd.Timestamp(entry_time)
    return Candidate(
        signal_idx=max(0, entry_idx - 1),
        entry_idx=entry_idx,
        signal_time=et - pd.Timedelta(minutes=1),
        entry_time=et,
        signal_z=5.1,
        previous_z=4.9,
        signal_keltner=1.0,
        deadline=pd.Timestamp(deadline),
    )


def atr_row(start="2024-01-02 12:00+00:00", tr=10.0):
    s = pd.Timestamp(start)
    return pd.Series({"source_start": s, "source_end": s + pd.Timedelta(hours=4), "tr": tr})


def test_opening_stop_gap_exits_at_open_and_ignores_later_high(make_bars):
    bars = make_bars([
        ("2024-01-02 17:00+00:00", 100, 101, 99, 100),
        ("2024-01-02 17:01+00:00", 87, 110, 85, 100),
        ("2024-01-02 20:00+00:00", 100, 100, 100, 100),
    ])
    c = candidate("2024-01-02 17:00+00:00", "2024-01-02 20:00+00:00")
    t = simulate_trade(bars, c, atr_row(tr=10.0))
    # Entry bar itself has low 99 <= stop 90? no. second bar gaps below 90.
    assert t.exit_reason == "SL"
    assert t.exit_time == pd.Timestamp("2024-01-02 17:01+00:00")
    assert t.exit_price == 87
    assert t.gross_R == -1.3


def test_opening_target_gap_gets_no_improvement(make_bars):
    bars = make_bars([
        ("2024-01-02 17:00+00:00", 100, 101, 99, 100),
        ("2024-01-02 17:01+00:00", 110, 115, 109, 112),
        ("2024-01-02 20:00+00:00", 100, 100, 100, 100),
    ])
    c = candidate("2024-01-02 17:00+00:00", "2024-01-02 20:00+00:00")
    t = simulate_trade(bars, c, atr_row(tr=10.0))
    assert t.exit_reason == "TP"
    assert t.exit_price == 105.0
    assert t.gross_R == 0.5


def test_same_bar_both_touched_stop_wins(make_bars):
    bars = make_bars([
        ("2024-01-02 17:00+00:00", 100, 106, 89, 101),
        ("2024-01-02 20:00+00:00", 101, 101, 101, 101),
    ])
    c = candidate("2024-01-02 17:00+00:00", "2024-01-02 20:00+00:00")
    t = simulate_trade(bars, c, atr_row(tr=10.0))
    assert t.exit_reason == "SL"
    assert t.exit_price == 90.0


def test_exact_deadline_flattens_at_open_before_high_low(make_bars):
    bars = make_bars([
        ("2024-01-02 19:59+00:00", 100, 101, 99, 100),
        ("2024-01-02 20:00+00:00", 102, 200, 50, 150),
    ])
    c = candidate("2024-01-02 19:59+00:00", "2024-01-02 20:00+00:00")
    t = simulate_trade(bars, c, atr_row(tr=20.0))
    assert t.exit_reason == "FLAT_1500"
    assert t.exit_time == pd.Timestamp("2024-01-02 20:00+00:00")
    assert t.exit_price == 102


def test_missing_deadline_uses_full_last_predeadline_range_then_close(make_bars):
    bars = make_bars([
        ("2024-01-02 19:58+00:00", 100, 101, 99, 100),
        ("2024-01-02 19:59+00:00", 100, 104, 96, 103),
        ("2024-01-02 20:02+00:00", 500, 600, 1, 500),
    ])
    c = candidate("2024-01-02 19:58+00:00", "2024-01-02 20:00+00:00")
    t = simulate_trade(bars, c, atr_row(tr=20.0))
    assert t.exit_reason == "FLAT_1500"
    assert t.exit_time == pd.Timestamp("2024-01-02 19:59+00:00")
    assert t.exit_price == 103


def test_missing_deadline_last_bar_can_hit_target_before_forced_close(make_bars):
    bars = make_bars([
        ("2024-01-02 19:58+00:00", 100, 101, 99, 100),
        ("2024-01-02 19:59+00:00", 100, 111, 96, 103),
        ("2024-01-02 20:02+00:00", 100, 100, 100, 100),
    ])
    c = candidate("2024-01-02 19:58+00:00", "2024-01-02 20:00+00:00")
    t = simulate_trade(bars, c, atr_row(tr=20.0))
    assert t.exit_reason == "TP"
    assert t.exit_time == pd.Timestamp("2024-01-02 19:59+00:00")
    assert t.exit_price == 110.0
