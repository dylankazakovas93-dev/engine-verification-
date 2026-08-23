from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

@dataclass
class Finding:
    check: str
    status: str
    message: str
    rows: list[int] | None = None

    def to_dict(self):
        return asdict(self)

def audit_bars(
    bars: pd.DataFrame,
    timestamp_col: str = "timestamp",
    contract_col: str | None = None,
) -> list[Finding]:
    f: list[Finding] = []
    df = bars.copy()
    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=False)
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index, index=df.index)
    else:
        return [Finding("timestamps", "FAIL", f"no {timestamp_col!r} column and index is not DatetimeIndex")]

    bad_ts = pd.isna(ts)
    f.append(Finding("timestamps_parse", "FAIL" if bad_ts.any() else "PASS",
                     f"{int(bad_ts.sum())} unparseable timestamps",
                     np.flatnonzero(np.asarray(bad_ts))[:20].tolist() if bad_ts.any() else None))

    try:
        idx = pd.DatetimeIndex(ts)
        aware = idx.tz is not None
    except Exception:
        aware = False
        idx = None
    f.append(Finding("timestamps_timezone", "PASS" if aware else "FAIL",
                     "timestamps are timezone-aware" if aware else "timestamps are timezone-naive or mixed"))

    if idx is not None and len(idx):
        f.append(Finding("timestamps_monotonic", "PASS" if idx.is_monotonic_increasing else "FAIL",
                         "chronological" if idx.is_monotonic_increasing else "rows are not chronological"))
        f.append(Finding("timestamps_unique", "PASS" if not idx.has_duplicates else "FAIL",
                         "no duplicate timestamps" if not idx.has_duplicates else f"{int(idx.duplicated().sum())} duplicate timestamps"))

    required = ["open","high","low","close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        f.append(Finding("ohlc_schema","FAIL",f"missing columns: {missing}"))
        return f
    f.append(Finding("ohlc_schema","PASS","required OHLC columns present"))

    vals = df[required].apply(pd.to_numeric, errors="coerce")
    bad_num = ~np.isfinite(vals.to_numpy(dtype=float))
    f.append(Finding("ohlc_finite","FAIL" if bad_num.any() else "PASS",
                     f"{int(bad_num.sum())} nonfinite/non-numeric OHLC cells"))
    if not bad_num.any():
        geom = (
            (vals["high"] < vals["low"]) |
            (vals["open"] < vals["low"]) | (vals["open"] > vals["high"]) |
            (vals["close"] < vals["low"]) | (vals["close"] > vals["high"])
        )
        f.append(Finding("ohlc_geometry","FAIL" if geom.any() else "PASS",
                         f"{int(geom.sum())} rows violate OHLC geometry",
                         np.flatnonzero(geom.to_numpy())[:20].tolist() if geom.any() else None))
        nonpos=(vals<=0).any(axis=1)
        f.append(Finding("ohlc_positive","FAIL" if nonpos.any() else "PASS",
                         f"{int(nonpos.sum())} rows contain nonpositive prices",
                         np.flatnonzero(nonpos.to_numpy())[:20].tolist() if nonpos.any() else None))

    if "volume" in df.columns:
        v=pd.to_numeric(df["volume"],errors="coerce")
        bad=(~np.isfinite(v.to_numpy(dtype=float))) | (v.to_numpy(dtype=float)<0)
        f.append(Finding("volume","FAIL" if bad.any() else "PASS",
                         f"{int(bad.sum())} invalid volume rows"))
    else:
        f.append(Finding("volume","WARN","volume column absent"))

    if contract_col:
        if contract_col not in df.columns:
            f.append(Finding("contract_provenance","UNVERIFIED",f"{contract_col!r} not present; rollover cannot be verified from continuous OHLC alone"))
        else:
            nulls=df[contract_col].isna()
            f.append(Finding("contract_id_missing","FAIL" if nulls.any() else "PASS",
                             f"{int(nulls.sum())} rows missing contract_id"))
            if idx is not None:
                tmp=pd.DataFrame({"ts":idx, "contract":df[contract_col].astype(str).to_numpy()})
                dup_contract=tmp.groupby("ts")["contract"].nunique().gt(1)
                f.append(Finding("contract_multiple_per_timestamp","FAIL" if dup_contract.any() else "PASS",
                                 f"{int(dup_contract.sum())} timestamps contain multiple selected contract IDs"))
                switches=(tmp["contract"] != tmp["contract"].shift(1))
                f.append(Finding("contract_switches","INFO",f"{max(int(switches.sum())-1,0)} contract switches observed"))
    else:
        f.append(Finding("contract_provenance","UNVERIFIED",
                         "no contract column supplied; continuous-contract roll correctness is not inferable from OHLCV alone"))
    return f
