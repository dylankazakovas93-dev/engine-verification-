from __future__ import annotations

import pandas as pd

from verifier.causality import _canonical_before


def test_missing_boundary_forced_flat_is_not_knowable_before_its_deadline():
    trade = pd.DataFrame({
        "entry_time": [pd.Timestamp("2024-01-02 17:00+00:00")],
        "exit_time": [pd.Timestamp("2024-01-02 17:01+00:00")],
        "exit_reason": ["FLAT_1500"],
        "deadline": [pd.Timestamp("2024-01-02 20:00+00:00")],
    })
    cutoff = pd.Timestamp("2024-01-02 17:02+00:00")
    assert _canonical_before(trade, "trades", cutoff).empty


def test_price_exit_is_knowable_at_exit_without_waiting_for_deadline():
    trade = pd.DataFrame({
        "entry_time": [pd.Timestamp("2024-01-02 17:00+00:00")],
        "exit_time": [pd.Timestamp("2024-01-02 17:01+00:00")],
        "exit_reason": ["TP"],
        "deadline": [pd.Timestamp("2024-01-02 20:00+00:00")],
    })
    cutoff = pd.Timestamp("2024-01-02 17:02+00:00")
    assert len(_canonical_before(trade, "trades", cutoff)) == 1
