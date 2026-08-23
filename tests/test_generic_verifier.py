from __future__ import annotations
import pandas as pd
import numpy as np

from verifier.schema import audit_bars
from verifier.ledger import audit_ledger
from verifier.static_scan import scan_source
from verifier.causality import audit_causality


def _status(findings, name):
    return [f.status for f in findings if f.check == name]


def test_generic_audit_catches_overlap():
    t = pd.DataFrame({
        "entry_time": ["2026-01-01T10:00:00Z", "2026-01-01T10:04:00Z"],
        "exit_time":  ["2026-01-01T10:05:00Z", "2026-01-01T10:06:00Z"],
        "entry_price": [100, 101],
        "exit_price": [102, 103],
    })
    f = audit_ledger(t)
    assert _status(f, "one_global_position") == ["FAIL"]


def test_generic_audit_catches_same_timestamp_reentry_when_strict():
    t = pd.DataFrame({
        "entry_time": ["2026-01-01T10:00:00Z", "2026-01-01T10:05:00Z"],
        "exit_time":  ["2026-01-01T10:05:00Z", "2026-01-01T10:06:00Z"],
        "entry_price": [100, 101],
        "exit_price": [102, 103],
    })
    f = audit_ledger(t, strict_same_timestamp_reentry=True)
    assert _status(f, "one_global_position") == ["FAIL"]


def test_generic_audit_catches_bad_r_math():
    t = pd.DataFrame({
        "entry_time": ["2026-01-01T10:00:00Z"],
        "exit_time":  ["2026-01-01T10:05:00Z"],
        "entry_price": [100],
        "exit_price": [102],
        "gross_R": [.5],
        "cost_R": [.1],
        "net_R": [.5],  # should be .4
    })
    f = audit_ledger(t)
    assert _status(f, "r_accounting") == ["FAIL"]


def test_generic_data_audit_marks_roll_unverified_without_contract_id():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
        "open": [100,101,102], "high":[101,102,103],
        "low":[99,100,101], "close":[100,101,102], "volume":[1,1,1],
    })
    f = audit_bars(df)
    assert _status(f, "contract_provenance") == ["UNVERIFIED"]


def test_static_scan_flags_negative_shift(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("x = close.shift(-1)\n")
    f = scan_source(p)
    assert any(x.check == "static_negative_shift" for x in f)


def test_causality_harness_catches_engine_using_final_future_close():
    idx = pd.date_range("2026-01-01", periods=400, freq="min", tz="UTC")
    bars = pd.DataFrame({
        "open": np.arange(400, dtype=float)+100,
        "high": np.arange(400, dtype=float)+101,
        "low": np.arange(400, dtype=float)+99,
        "close": np.arange(400, dtype=float)+100.5,
        "volume": 1.0,
    }, index=idx)

    def leaking_engine(x):
        # A deliberately bad engine: past exit price depends on the final future close.
        if len(x) < 20:
            return pd.DataFrame(columns=["entry_time","exit_time","entry_price","exit_price"])
        return pd.DataFrame({
            "entry_time": [x.index[5]],
            "exit_time": [x.index[10]],
            "entry_price": [float(x.iloc[5].open)],
            "exit_price": [float(x.iloc[-1].close)],
        })
    f = audit_causality(leaking_engine, bars)
    assert any(x.status == "FAIL" for x in f)
