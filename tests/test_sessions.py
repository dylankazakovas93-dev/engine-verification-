from __future__ import annotations

import pandas as pd

from nq_frozen.core import classify_ct, eligible_signal_entry_pair, permitted_et, trading_deadline


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/Chicago").tz_convert("UTC")


def et(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


def test_ct_boundaries():
    assert classify_ct(utc("2024-01-02 01:59")) == "globex_asia"
    assert classify_ct(utc("2024-01-02 02:00")) == "london"
    assert classify_ct(utc("2024-01-02 08:29")) == "london"
    assert classify_ct(utc("2024-01-02 08:30")) == "ny_am"
    assert classify_ct(utc("2024-01-02 11:59")) == "ny_am"
    assert classify_ct(utc("2024-01-02 12:00")) == "ny_pm"
    assert classify_ct(utc("2024-01-02 14:59")) == "ny_pm"
    assert classify_ct(utc("2024-01-02 15:00")) == "ny_late"
    assert classify_ct(utc("2024-01-02 15:59")) == "ny_late"
    assert classify_ct(utc("2024-01-02 16:00")) == "maintenance"
    assert classify_ct(utc("2024-01-02 16:59")) == "maintenance"
    assert classify_ct(utc("2024-01-02 17:00")) == "globex_asia"


def test_et_permitted_boundaries():
    assert permitted_et(et("2024-01-02 14:59"))
    assert not permitted_et(et("2024-01-02 15:00"))
    assert not permitted_et(et("2024-01-02 17:59"))
    assert permitted_et(et("2024-01-02 18:00"))


def test_deadline_same_day_and_next_day():
    assert trading_deadline(et("2024-01-02 14:59")) == et("2024-01-02 15:00")
    assert trading_deadline(et("2024-01-02 18:00")) == et("2024-01-03 15:00")


def test_same_session_required():
    ok, d = eligible_signal_entry_pair(et("2024-01-02 14:58"), et("2024-01-02 14:59"))
    assert ok and d == et("2024-01-02 15:00")
    ok, d = eligible_signal_entry_pair(et("2024-01-02 14:59"), et("2024-01-02 18:00"))
    assert not ok and d is None


def test_london_excluded_even_if_et_permitted():
    signal = utc("2024-01-02 02:00")
    entry = utc("2024-01-02 02:01")
    ok, _ = eligible_signal_entry_pair(signal, entry)
    assert not ok


def test_dst_deadline_uses_local_calendar_not_24h_arithmetic():
    # Friday before US spring DST shift: an eligible Sunday evening entry must still land at Monday 15:00 ET.
    ts = et("2024-03-10 18:00")
    assert trading_deadline(ts) == et("2024-03-11 15:00")
