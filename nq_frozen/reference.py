from __future__ import annotations

"""Deliberately simple, loop-first reference implementation.

This module does not import production-engine functions. It exists to provide an
independent implementation of the frozen rules for reconciliation.
"""

from dataclasses import asdict
from datetime import datetime, time, timedelta
from decimal import Decimal, ROUND_CEILING
from math import exp, isfinite, log, sqrt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .constants import (
    COST_CHANGE_YEAR,
    D,
    IS_YEARS,
    KELTNER_EMA_SPAN,
    KELTNER_MAX,
    KELTNER_RANGE_WINDOW,
    MAX_WEIGHTS,
    OOS_YEARS,
    POST_2023_COST_POINTS,
    TARGET_R,
    TICK_SIZE,
    WEIGHT_CUTOFF,
    Z_THRESHOLD,
    Z_WINDOW,
)
from .models import Trade

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")


def _validate(bars: pd.DataFrame) -> None:
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("reference engine requires timezone-aware DatetimeIndex")
    if str(bars.index.tz) != "UTC":
        raise ValueError("reference engine requires UTC index")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("reference engine requires unique chronological rows")
    for c in ("open", "high", "low", "close", "volume"):
        if c not in bars.columns:
            raise ValueError(f"missing {c}")


def _weights() -> list[float]:
    w = [1.0]
    for k in range(1, MAX_WEIGHTS):
        nxt = -w[-1] * (D - k + 1.0) / k
        if abs(nxt) < WEIGHT_CUTOFF:
            break
        w.append(nxt)
    return w


def reference_features(bars: pd.DataFrame) -> pd.DataFrame:
    _validate(bars)
    n = len(bars)
    closes = [float(x) for x in bars["close"]]
    highs = [float(x) for x in bars["high"]]
    lows = [float(x) for x in bars["low"]]
    logs = [log(x) for x in closes]
    w = _weights()
    m = len(w)

    fd = [float("nan")] * n
    for i in range(m - 1, n):
        total = 0.0
        ok = True
        for k, wk in enumerate(w):
            x = logs[i - k]
            if not isfinite(x):
                ok = False
                break
            total += wk * x
        if ok:
            fd[i] = total

    z = [float("nan")] * n
    valid_fd: list[float] = []
    for i, x in enumerate(fd):
        if isfinite(x):
            valid_fd.append(x)
            if len(valid_fd) >= Z_WINDOW:
                win = valid_fd[-Z_WINDOW:]
                mean = sum(win) / Z_WINDOW
                var = sum((v - mean) ** 2 for v in win) / Z_WINDOW
                sd = sqrt(var)
                if sd > 0:
                    z[i] = (x - mean) / sd

    alpha = 2.0 / (KELTNER_EMA_SPAN + 1.0)
    ema = [float("nan")] * n
    if n:
        ema[0] = closes[0]
        for i in range(1, n):
            ema[i] = alpha * closes[i] + (1.0 - alpha) * ema[i - 1]

    mean_range = [float("nan")] * n
    keltner = [float("nan")] * n
    ranges = [highs[i] - lows[i] for i in range(n)]
    for i in range(KELTNER_RANGE_WINDOW - 1, n):
        mr = sum(ranges[i - KELTNER_RANGE_WINDOW + 1 : i + 1]) / KELTNER_RANGE_WINDOW
        mean_range[i] = mr
        if mr != 0 and isfinite(ema[i]):
            keltner[i] = (closes[i] - ema[i]) / mr

    raw = [False] * n
    passed = [False] * n
    for i in range(1, n):
        if isfinite(z[i]) and isfinite(z[i - 1]) and z[i] >= Z_THRESHOLD and z[i - 1] < Z_THRESHOLD:
            raw[i] = True
            if isfinite(keltner[i]) and keltner[i] <= KELTNER_MAX:
                passed[i] = True

    return pd.DataFrame(
        {
            "fracdiff": fd,
            "z": z,
            "ema60": ema,
            "mean_range60": mean_range,
            "keltner": keltner,
            "raw_long_signal": raw,
            "signal_passes_keltner": passed,
        },
        index=bars.index,
    )


def _ct_class(ts: pd.Timestamp) -> str:
    x = ts.tz_convert(CT)
    m = x.hour * 60 + x.minute
    if m >= 17 * 60 or m < 2 * 60:
        return "globex_asia"
    if m < 8 * 60 + 30:
        return "london"
    if m < 12 * 60:
        return "ny_am"
    if m < 15 * 60:
        return "ny_pm"
    if m < 16 * 60:
        return "ny_late"
    return "maintenance"


def _permitted_et(ts: pd.Timestamp) -> bool:
    x = ts.tz_convert(ET)
    m = x.hour * 60 + x.minute
    return m >= 18 * 60 or m < 15 * 60


def _deadline(ts: pd.Timestamp) -> pd.Timestamp | None:
    if not _permitted_et(ts):
        return None
    x = ts.tz_convert(ET)
    d = x.date() + (timedelta(days=1) if x.hour >= 18 else timedelta(0))
    local = pd.Timestamp(datetime.combine(d, time(15), tzinfo=ET))
    return local.tz_convert("UTC")


def _eligible_pair(signal: pd.Timestamp, entry: pd.Timestamp) -> tuple[bool, pd.Timestamp | None]:
    if _ct_class(signal) == "london" or _ct_class(entry) == "london":
        return False, None
    if not _permitted_et(signal) or not _permitted_et(entry):
        return False, None
    ds, de = _deadline(signal), _deadline(entry)
    if ds is None or ds != de:
        return False, None
    return True, ds


def _four_hour_table(bars: pd.DataFrame) -> list[dict]:
    buckets: list[dict] = []
    current_start = None
    current = None
    for ts, row in bars.iterrows():
        b = ts.floor("4h")
        if current_start is None or b != current_start:
            if current is not None:
                buckets.append(current)
            current_start = b
            current = {
                "source_start": b,
                "source_end": b + pd.Timedelta(hours=4),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        else:
            current["high"] = max(current["high"], float(row["high"]))
            current["low"] = min(current["low"], float(row["low"]))
            current["close"] = float(row["close"])
    if current is not None:
        buckets.append(current)

    prev_close = None
    for b in buckets:
        hl = b["high"] - b["low"]
        if prev_close is None:
            tr = hl
        else:
            tr = max(hl, abs(b["high"] - prev_close), abs(b["low"] - prev_close))
        b["previous_close"] = prev_close
        b["tr"] = tr
        b["available_at"] = b["source_end"]
        prev_close = b["close"]
    return buckets


def _atr_at(table: list[dict], entry: pd.Timestamp) -> dict | None:
    chosen = None
    for r in table:
        if r["available_at"] <= entry:
            chosen = r
        else:
            break
    if chosen is None or not isfinite(chosen["tr"]) or chosen["tr"] <= 0:
        return None
    return chosen


def _ceil_tick(x: float) -> float:
    p = Decimal(str(x))
    t = Decimal(str(TICK_SIZE))
    return float((p / t).to_integral_value(rounding=ROUND_CEILING) * t)


def _label(year: int) -> str:
    if year in IS_YEARS:
        return "IS"
    if year in OOS_YEARS:
        return "OOS"
    return "UNLABELED"


def run_reference(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate(bars)
    f = reference_features(bars)
    atr_table = _four_hour_table(bars)
    trades: list[Trade] = []
    previous_exit = None

    for signal_i in range(len(bars)):
        if not bool(f.iloc[signal_i]["signal_passes_keltner"]):
            continue
        entry_i = signal_i + 1
        if entry_i >= len(bars):
            continue
        signal_ts = bars.index[signal_i]
        entry_ts = bars.index[entry_i]
        ok, deadline = _eligible_pair(signal_ts, entry_ts)
        if not ok or deadline is None:
            continue
        if previous_exit is not None and entry_ts <= previous_exit:
            continue

        atr_row = _atr_at(atr_table, entry_ts)
        if atr_row is None:
            continue
        entry = float(bars.iloc[entry_i]["open"])
        atr = float(atr_row["tr"])
        stop_unrounded = entry - atr
        target_unrounded = entry + TARGET_R * atr
        stop = _ceil_tick(stop_unrounded)
        target = _ceil_tick(target_unrounded)
        risk = entry - stop
        if not isfinite(risk) or risk <= 0:
            continue

        exact_deadline = deadline in bars.index
        last_pre_i = None
        for j in range(entry_i, len(bars)):
            if bars.index[j] < deadline:
                last_pre_i = j
            else:
                break
        if last_pre_i is None:
            continue

        exit_ts = exit_px = reason = None
        final_i = bars.index.get_loc(deadline) if exact_deadline else last_pre_i
        for j in range(entry_i, int(final_i) + 1):
            ts = bars.index[j]
            row = bars.iloc[j]
            if ts == deadline:
                exit_ts, exit_px, reason = ts, float(row["open"]), "FLAT_1500"
                break
            o, lo, hi = float(row["open"]), float(row["low"]), float(row["high"])
            if o <= stop:
                exit_ts, exit_px, reason = ts, o, "SL"
                break
            if o >= target:
                exit_ts, exit_px, reason = ts, target, "TP"
                break
            if lo <= stop:
                exit_ts, exit_px, reason = ts, stop, "SL"
                break
            if hi >= target:
                exit_ts, exit_px, reason = ts, target, "TP"
                break
            if not exact_deadline and j == last_pre_i:
                exit_ts, exit_px, reason = ts, float(row["close"]), "FLAT_1500"
                break
        if reason is None:
            raise AssertionError("reference trade failed to resolve")

        if reason == "TP":
            gross = TARGET_R
        elif reason == "SL":
            gross = (exit_px - entry) / risk if exit_px < stop else -1.0
        else:
            gross = (exit_px - entry) / risk
        year = int(entry_ts.tz_convert(ET).year)
        cost_points = 0.0 if year < COST_CHANGE_YEAR else POST_2023_COST_POINTS
        cost_r = cost_points / risk
        net = gross - cost_r
        t = Trade(
            signal_time=signal_ts,
            entry_time=entry_ts,
            entry_price=entry,
            signal_z=float(f.iloc[signal_i]["z"]),
            previous_z=float(f.iloc[signal_i - 1]["z"]),
            signal_keltner=float(f.iloc[signal_i]["keltner"]),
            atr_source_start=atr_row["source_start"],
            atr_source_end=atr_row["source_end"],
            atr_value=atr,
            stop_unrounded=stop_unrounded,
            stop_price=stop,
            target_unrounded=target_unrounded,
            target_price=target,
            initial_risk=risk,
            deadline=deadline,
            exit_time=exit_ts,
            exit_price=float(exit_px),
            exit_reason=reason,
            gross_R=float(gross),
            cost_points=float(cost_points),
            cost_R=float(cost_r),
            net_R=float(net),
            entry_year=year,
            sample=_label(year),
        )
        trades.append(t)
        previous_exit = exit_ts

    if not trades:
        cols = list(Trade.__dataclass_fields__.keys())
        return pd.DataFrame(columns=cols), f
    return pd.DataFrame([asdict(t) for t in trades]), f
