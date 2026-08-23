from __future__ import annotations

import pandas as pd

from verifier.high_resolution import audit_high_resolution_execution, resolve_ambiguous_bar
from verifier.orchestrator import load_high_resolution_trade_windows


def second_frame(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"]).set_index("timestamp")


def test_one_second_sequence_resolves_stop_before_target():
    seconds = second_frame([
        (pd.Timestamp("2026-01-01T10:00:00Z"), 100, 102, 99, 101),
        (pd.Timestamp("2026-01-01T10:00:01Z"), 100, 101, 89, 90),
        (pd.Timestamp("2026-01-01T10:00:02Z"), 100, 111, 99, 110),
    ])
    classification, event, timestamp, _ = resolve_ambiguous_bar(seconds, stop_price=90, target_price=110)
    assert classification == "RESOLVED_BY_1S"
    assert event == "SL"
    assert timestamp == pd.Timestamp("2026-01-01T10:00:01Z")


def test_same_one_second_both_remains_ambiguous():
    seconds = second_frame([(pd.Timestamp("2026-01-01T10:00:00Z"), 100, 111, 89, 100)])
    classification, event, _, _ = resolve_ambiguous_bar(seconds, stop_price=90, target_price=110)
    assert classification == "STILL_AMBIGUOUS_WITHIN_1S"
    assert event == "BOTH"


def test_coarse_audit_only_targets_both_touched_bars():
    minute = second_frame([(pd.Timestamp("2026-01-01T10:00:00Z"), 100, 111, 89, 100)])
    seconds = second_frame([
        (pd.Timestamp("2026-01-01T10:00:00Z"), 100, 101, 89, 90),
        (pd.Timestamp("2026-01-01T10:00:01Z"), 100, 111, 99, 110),
    ])
    trades = pd.DataFrame({"exit_time": [pd.Timestamp("2026-01-01T10:00:00Z")], "stop_price": [90], "target_price": [110]})
    findings, resolutions = audit_high_resolution_execution(trades, minute, seconds)
    assert findings[0].status == "PASS"
    assert resolutions[0].classification == "RESOLVED_BY_1S"


def test_parquet_loader_reads_only_matching_trade_window_and_contract(tmp_path):
    path = tmp_path / "seconds.parquet"
    pd.DataFrame({
        "ts_event": pd.to_datetime([
            "2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z",
            "2026-01-01T10:00:00Z", "2026-01-01T10:02:00Z",
        ]),
        "instrument_id": [7, 7, 8, 7], "open": [100] * 4, "high": [101] * 4,
        "low": [99] * 4, "close": [100] * 4, "volume": [1] * 4,
    }).to_parquet(path, index=False)
    trades = pd.DataFrame({"exit_time": [pd.Timestamp("2026-01-01T10:00:00Z")], "contract_id": [7]})
    loaded = load_high_resolution_trade_windows(
        path, trades, timestamp_col="ts_event", high_res_contract_col="instrument_id",
        candidate_contract_col="contract_id", interval="1min",
    )
    assert len(loaded) == 2
    assert set(loaded["contract_id"]) == {7}
