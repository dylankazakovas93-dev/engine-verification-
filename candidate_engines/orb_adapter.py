"""Adapter wiring the CORRECTED ORB engine (orb_engine_corrected.py) into this
verification lab.

v3 of this adapter. v1 called five separate per-day loops (a performance bug
that never finished in 40 minutes). v2 fixed that but conflated "signal
fired" with "trade fully resolved", because it built every stage off
run_session()'s return value, which is None unless the entry bar AND the
full forward window exist. That produced the 13 truncation "PROVEN FAILURE"
findings at cutoffs landing exactly on 10:00 ET -- not lookahead, but an
adapter that couldn't report a signal without also being able to resolve a
trade.

v3 fixes this properly: _scan_signal() replicates the qualification logic
(range/ATR/displacement/efficiency gates -- everything settled by 09:59)
independently of whether the entry bar or forward path exist at all. A
truncation at exactly 10:00 can now prove the 09:59 decision is invariant
even though there is no entry bar yet to build a trade from.

v3.1: _to_ledger() drops UNRESOLVED_DATA_GAP rows from the ledger entirely
(see its docstring) instead of keeping them with NaN placeholders.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/repricing/orb_engine")
import orb_engine_corrected as E


def _prep(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.copy()
    b["et"] = b.index.tz_convert("America/New_York")
    b["date"] = pd.to_datetime(b.et.dt.date)
    return b.reset_index(drop=True)


def _scan_signal(bars: pd.DataFrame) -> pd.DataFrame:
    """Qualification only -- everything decided by the 09:59 close. Does NOT
    require the entry bar or any forward data to exist."""
    b = _prep(bars)
    rows = []
    for _, day in b.groupby("date", sort=True):
        tod = day.et.dt.time
        orb = day[(tod >= E.RANGE_START) & (tod <= E.RANGE_END)]
        if len(orb) != 30 or orb.et.dt.time.iloc[0] != E.RANGE_START:
            continue
        atr14 = E.atr_at(day, E.ATR_AS_OF, 14)
        atr12 = E.atr_at(day, E.ATR_AS_OF, 12)
        if not np.isfinite(atr14) or not np.isfinite(atr12) or min(atr14, atr12) < E.MIN_ATR:
            continue
        o = float(orb.open.iloc[0])
        c = float(orb.close.iloc[-1])
        disp = c - o
        p = E.Params()
        if abs(disp) < p.disp_atr * atr14:
            continue
        eff = E.path_efficiency(orb)
        if eff < p.eff_min:
            continue
        side = 1 if disp > 0 else -1
        rows.append({
            "signal_time": day.date.iloc[0] + pd.Timedelta(hours=9, minutes=59),
            "qualified": True, "side": side, "efficiency": eff,
            "displacement": disp, "atr12": atr12, "atr14": atr14,
            "raw_stop_distance": p.stop_atr * atr12,
            "raw_target_distance": p.target_atr * atr12,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        tz = bars.index.tz
        out["signal_time"] = out["signal_time"].dt.tz_localize("America/New_York").dt.tz_convert(tz)
    return out


def _scan_trades(bars: pd.DataFrame) -> pd.DataFrame:
    b = _prep(bars)
    return E.backtest_corrected(b, E.Params())


def _to_ledger(t: pd.DataFrame) -> pd.DataFrame:
    """UNRESOLVED_DATA_GAP rows (data cut before the trade could be resolved
    -- currently just 2020-06-30, a single genuine mid-session Databento
    truncation, not a bank-holiday artifact) are dropped from the ledger
    entirely. Per operator instruction: a trade with no data to determine
    its result does not belong in a P&L ledger. The exclusion is still
    fully visible and reproducible outside this ledger -- it's flagged by
    orb_engine_corrected.py's own exit_reason field, itemized in
    changed_trades_reconciliation.csv, and logged in DECISIONS.md -- so
    dropping it here is a ledger-formatting choice, not a cover-up."""
    cols = ["entry_time", "exit_time", "entry_price", "exit_price", "signal_time",
            "deadline", "stop_price", "target_price", "initial_risk",
            "atr_source_end", "gross_R", "cost_R", "net_R", "side"]
    if t.empty:
        return pd.DataFrame(columns=cols)
    t = t[t.exit_reason != "UNRESOLVED_DATA_GAP"].copy()
    if t.empty:
        return pd.DataFrame(columns=cols)

    t["exit_price"] = t.entry + t.side * t.pnl_points
    t["gross_R"] = t.r_multiple
    t["cost_R"] = 0.0
    t["net_R"] = t.r_multiple
    t["entry_price"] = t.entry
    t["initial_risk"] = t.stop_points
    t["signal_time"] = t.entry_time - pd.Timedelta(minutes=1)
    t["deadline"] = t.entry_time.dt.normalize() + pd.Timedelta(hours=15)
    t["atr_source_end"] = t.entry_time.dt.normalize() + pd.Timedelta(hours=9, minutes=29)
    return t[cols]


def run(bars: pd.DataFrame) -> pd.DataFrame:
    return _to_ledger(_scan_trades(bars))


def signals(bars: pd.DataFrame) -> pd.DataFrame:
    s = _scan_signal(bars)
    return s[["signal_time"]] if not s.empty else pd.DataFrame(columns=["signal_time"])


eligible_signals = signals


def features(bars: pd.DataFrame) -> pd.DataFrame:
    """The full qualification record -- everything established by 09:59,
    exposed as its own stage so feature-level causality is verifiable rather
    than reported UNVERIFIED."""
    s = _scan_signal(bars)
    if s.empty:
        return pd.DataFrame(columns=["timestamp", "qualified", "side", "efficiency",
                                     "displacement", "atr12", "atr14",
                                     "raw_stop_distance", "raw_target_distance"])
    return s.rename(columns={"signal_time": "timestamp"})


def proposed_entries(bars: pd.DataFrame) -> pd.DataFrame:
    t = _scan_trades(bars)
    if t.empty:
        return pd.DataFrame(columns=["signal_time", "entry_time"])
    out = t.copy()
    out["signal_time"] = out["entry_time"] - pd.Timedelta(minutes=1)
    return out[["signal_time", "entry_time"]]


accepted_entries = proposed_entries


def audit_stages(bars: pd.DataFrame) -> dict:
    """One _scan_signal() pass and one _scan_trades() pass, total -- every
    stage below is sliced from those two, not recomputed per stage. The first
    version of this adapter called five separate per-day loops and never
    finished in 40 minutes; keep it to two."""
    s = _scan_signal(bars)
    sig = s[["signal_time"]] if not s.empty else pd.DataFrame(columns=["signal_time"])
    feat = (s.rename(columns={"signal_time": "timestamp"}) if not s.empty else
            pd.DataFrame(columns=["timestamp", "qualified", "side", "efficiency",
                                  "displacement", "atr12", "atr14",
                                  "raw_stop_distance", "raw_target_distance"]))
    t = _scan_trades(bars)
    if t.empty:
        ent = pd.DataFrame(columns=["signal_time", "entry_time"])
    else:
        ent = pd.DataFrame({"signal_time": t.entry_time - pd.Timedelta(minutes=1),
                            "entry_time": t.entry_time})
    trades = _to_ledger(t)
    return {
        "features": feat,
        "signals": sig,
        "eligible_signals": sig,
        "proposed_entries": ent,
        "accepted_entries": ent,
        "trades": trades,
    }
