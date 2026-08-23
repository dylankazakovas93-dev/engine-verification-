from __future__ import annotations
import numpy as np
import pandas as pd
from .schema import Finding

def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def audit_ledger(
    trades: pd.DataFrame,
    *,
    strict_same_timestamp_reentry: bool = True,
    tick_size: float | None = None,
    long_only: bool | None = None,
    required_deadline: bool = False,
) -> list[Finding]:
    f: list[Finding]=[]
    req=["entry_time","exit_time","entry_price","exit_price"]
    if required_deadline:
        req.append("deadline")
    if long_only is True:
        req.append("direction")
    miss=[c for c in req if c not in trades.columns]
    if miss:
        return [Finding("ledger_schema","FAIL",f"missing canonical columns: {miss}",family="ledger")]
    f.append(Finding("ledger_schema","PASS","canonical entry/exit fields present"))
    if trades.empty:
        f.append(Finding("trade_count","INFO","0 accepted trades"))
        for finding in f:
            finding.family = finding.family or "ledger"
        return f

    t=trades.copy()
    t["_entry"]=_to_utc(t["entry_time"])
    t["_exit"]=_to_utc(t["exit_time"])
    bad_time=t["_entry"].isna()|t["_exit"].isna()
    f.append(Finding("ledger_timestamps","FAIL" if bad_time.any() else "PASS",f"{int(bad_time.sum())} invalid trade timestamps"))
    order=t["_exit"] < t["_entry"]
    f.append(Finding("exit_after_entry","FAIL" if order.any() else "PASS",f"{int(order.sum())} exits precede entries",
                     np.flatnonzero(order.to_numpy())[:20].tolist() if order.any() else None))

    s=t.sort_values(["_entry","_exit"]).reset_index()
    prev_exit=s["_exit"].shift(1)
    overlap=(s["_entry"] <= prev_exit) if strict_same_timestamp_reentry else (s["_entry"] < prev_exit)
    overlap=overlap.fillna(False)
    f.append(Finding("one_global_position","FAIL" if overlap.any() else "PASS",
                     f"{int(overlap.sum())} overlapping/same-open positions detected",
                     s.loc[overlap,"index"].astype(int).head(20).tolist() if overlap.any() else None))

    ep=pd.to_numeric(t["entry_price"],errors="coerce")
    xp=pd.to_numeric(t["exit_price"],errors="coerce")
    bad_price=(~np.isfinite(ep))|(~np.isfinite(xp))|(ep<=0)|(xp<=0)
    f.append(Finding("trade_prices","FAIL" if bad_price.any() else "PASS",f"{int(bad_price.sum())} invalid entry/exit prices"))

    if "signal_time" in t:
        sig=_to_utc(t["signal_time"])
        bad=t["_entry"] <= sig
        f.append(Finding("entry_after_signal","FAIL" if bad.any() else "PASS",
                         f"{int(bad.sum())} entries are not strictly after signal timestamps",
                         np.flatnonzero(bad.to_numpy())[:20].tolist() if bad.any() else None))
    if "deadline" in t:
        dl=_to_utc(t["deadline"])
        bad=t["_exit"] > dl
        f.append(Finding("deadline","FAIL" if bad.any() else "PASS",
                         f"{int(bad.sum())} trades exit after their deadline",
                         np.flatnonzero(bad.to_numpy())[:20].tolist() if bad.any() else None))
    if "atr_source_end" in t:
        ae=_to_utc(t["atr_source_end"])
        bad=ae > t["_entry"]
        f.append(Finding("atr_causality","FAIL" if bad.any() else "PASS",
                         f"{int(bad.sum())} trades use ATR/source data ending after entry",
                         np.flatnonzero(bad.to_numpy())[:20].tolist() if bad.any() else None))
    if "initial_risk" in t:
        r=pd.to_numeric(t["initial_risk"],errors="coerce")
        bad=(~np.isfinite(r))|(r<=0)
        f.append(Finding("initial_risk","FAIL" if bad.any() else "PASS",f"{int(bad.sum())} trades have nonpositive/nonfinite initial risk"))
    if {"gross_R","cost_R","net_R"}.issubset(t.columns):
        g=pd.to_numeric(t["gross_R"],errors="coerce")
        c=pd.to_numeric(t["cost_R"],errors="coerce")
        n=pd.to_numeric(t["net_R"],errors="coerce")
        bad=~np.isclose((g-c).to_numpy(float),n.to_numpy(float),rtol=0,atol=1e-9,equal_nan=False)
        f.append(Finding("r_accounting","FAIL" if bad.any() else "PASS",
                         f"{int(bad.sum())} rows violate net_R = gross_R - cost_R",
                         np.flatnonzero(bad)[:20].tolist() if bad.any() else None))
    if tick_size:
        for c in ["stop_price","target_price"]:
            if c in t:
                x=pd.to_numeric(t[c],errors="coerce").to_numpy(float)/tick_size
                bad=~np.isclose(x,np.round(x),rtol=0,atol=1e-9)
                f.append(Finding(f"{c}_tick_alignment","FAIL" if bad.any() else "PASS",
                                 f"{int(bad.sum())} {c} values are off tick"))
    if "direction" in t and long_only is not False:
        bad=~t["direction"].astype(str).str.lower().isin(["long","1","buy"])
        f.append(Finding("long_only","FAIL" if bad.any() else "PASS",f"{int(bad.sum())} non-long trades"))
    for finding in f:
        finding.family = finding.family or "ledger"
    return f
