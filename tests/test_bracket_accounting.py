from __future__ import annotations

import numpy as np
import pandas as pd

from nq_frozen.core import _accounting, bracket, ceil_to_tick, sample_label, summarize


def test_ceil_to_tick_exact_and_fractional():
    assert ceil_to_tick(99.75) == 99.75
    assert ceil_to_tick(99.76) == 100.0
    assert ceil_to_tick(99.01) == 99.25


def test_bracket_rounding_upward():
    # stop 98.9 -> 99.0, target 100.55 -> 100.75
    b = bracket(100.0, 1.1)
    assert b == (98.9, 99.0, 100.55, 100.75, 1.0)


def test_bracket_skips_zero_risk():
    assert bracket(100.0, 0.1) is None  # stop 99.9 ceils to 100.0


def test_tp_accounting_nominal_half_r_even_if_target_rounding_farther():
    t = pd.Timestamp("2024-01-02 14:00+00:00")
    gross, cost_pts, cost_r, net, year, sample = _accounting(t, 100, 101, "TP", 2, 98)
    assert gross == 0.5
    assert cost_pts == 1.0
    assert cost_r == 0.5
    assert net == 0.0


def test_stop_gap_can_be_worse_than_minus_one_r():
    t = pd.Timestamp("2022-01-02 14:00+00:00")
    gross, *_ = _accounting(t, 100, 87, "SL", 10, 90)
    assert gross == -1.3


def test_normal_stop_is_exact_minus_one_r():
    t = pd.Timestamp("2022-01-02 14:00+00:00")
    gross, *_ = _accounting(t, 100, 90, "SL", 10, 90)
    assert gross == -1.0


def test_forced_flat_uses_actual_r():
    t = pd.Timestamp("2022-01-02 14:00+00:00")
    gross, *_ = _accounting(t, 100, 103, "FLAT_1500", 10, 90)
    assert gross == 0.3


def test_cost_boundary_uses_et_entry_year():
    # 2023-01-01 00:30 UTC is still 2022-12-31 in New York.
    t = pd.Timestamp("2023-01-01 00:30+00:00")
    _, cost_pts, *_ = _accounting(t, 100, 105, "TP", 10, 90)
    assert cost_pts == 0.0
    t2 = pd.Timestamp("2023-01-01 05:30+00:00")
    _, cost_pts2, *_ = _accounting(t2, 100, 105, "TP", 10, 90)
    assert cost_pts2 == 1.0


def test_sample_labels():
    assert sample_label(2024) == "IS"
    assert sample_label(2025) == "OOS"
    assert sample_label(2009) == "UNLABELED"


def test_summary_pf_and_win_rate():
    trades = pd.DataFrame({"net_R": [0.5, -1.0, 0.25, 0.0]})
    s = summarize(trades)
    assert s["wins"] == 2
    assert s["loss_or_zero"] == 2
    assert s["win_rate"] == 50.0
    assert s["profit_factor"] == 0.75
