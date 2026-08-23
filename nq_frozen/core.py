from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, time, timedelta
from decimal import Decimal, ROUND_CEILING
from math import isfinite
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
from .models import Candidate, Trade

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
_REQUIRED = ("open", "high", "low", "close", "volume")


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate strategy input without repairing it."""
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars index must be a pandas DatetimeIndex")
    if bars.index.tz is None:
        raise ValueError("bars index must be timezone-aware UTC")
    if str(bars.index.tz) not in {"UTC", "utc", "+00:00"}:
        # Accept equivalent UTC offsets only after explicit normalization by caller.
        raise ValueError("bars index must be expressed internally in UTC")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must already be chronological; engine will not sort them")
    if bars.index.has_duplicates:
        raise ValueError("duplicate timestamps are prohibited")
    missing = [c for c in _REQUIRED if c not in bars.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if len(bars) == 0:
        return bars.copy()

    out = bars.loc[:, _REQUIRED].copy()
    for c in _REQUIRED:
        out[c] = pd.to_numeric(out[c], errors="raise")
        if not np.isfinite(out[c].to_numpy(dtype=float)).all():
            raise ValueError(f"column {c} contains nonfinite values")

    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (out["volume"] < 0).any():
        raise ValueError("volume cannot be negative")
    if (out["high"] < out["low"]).any():
        raise ValueError("high < low found")
    if ((out["open"] < out["low"]) | (out["open"] > out["high"])).any():
        raise ValueError("open outside [low, high] found")
    if ((out["close"] < out["low"]) | (out["close"] > out["high"])).any():
        raise ValueError("close outside [low, high] found")
    return out


def fracdiff_weights(
    d: float = D,
    max_weights: int = MAX_WEIGHTS,
    cutoff: float = WEIGHT_CUTOFF,
) -> np.ndarray:
    if max_weights <= 0:
        raise ValueError("max_weights must be positive")
    w = [1.0]
    for k in range(1, max_weights):
        nxt = -w[-1] * (d - k + 1.0) / k
        if abs(nxt) < cutoff:
            break
        w.append(float(nxt))
    return np.asarray(w, dtype=float)


def causal_fracdiff(log_close: pd.Series, weights: np.ndarray | None = None) -> pd.Series:
    """w[0] * x[t] + w[1] * x[t-1] + ... over stored rows."""
    w = fracdiff_weights() if weights is None else np.asarray(weights, dtype=float)
    x = log_close.to_numpy(dtype=float)
    result = np.full(len(x), np.nan, dtype=float)
    m = len(w)
    for i in range(m - 1, len(x)):
        window = x[i - m + 1 : i + 1]
        if np.isfinite(window).all():
            result[i] = float(np.dot(w, window[::-1]))
    return pd.Series(result, index=log_close.index, name="fracdiff")


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    bars = validate_bars(bars)
    out = bars.copy()
    log_close = np.log(out["close"].astype(float))
    out["fracdiff"] = causal_fracdiff(log_close)

    fd = out["fracdiff"]
    mean = fd.rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()
    std = fd.rolling(Z_WINDOW, min_periods=Z_WINDOW).std(ddof=0)
    out["z"] = (fd - mean) / std.replace(0.0, np.nan)

    ema = out["close"].ewm(span=KELTNER_EMA_SPAN, adjust=False).mean()
    mean_range = (out["high"] - out["low"]).rolling(
        KELTNER_RANGE_WINDOW, min_periods=KELTNER_RANGE_WINDOW
    ).mean()
    out["ema60"] = ema
    out["mean_range60"] = mean_range
    out["keltner"] = (out["close"] - ema) / mean_range.replace(0.0, np.nan)

    prev_z = out["z"].shift(1)
    out["raw_long_signal"] = (
        (out["z"] >= Z_THRESHOLD)
        & (prev_z < Z_THRESHOLD)
        & np.isfinite(out["z"])
        & np.isfinite(prev_z)
    )
    out["signal_passes_keltner"] = (
        out["raw_long_signal"]
        & np.isfinite(out["keltner"])
        & (out["keltner"] <= KELTNER_MAX)
    )
    return out


def classify_ct(ts_utc: pd.Timestamp) -> str:
    local = ts_utc.tz_convert(CT)
    minute = local.hour * 60 + local.minute
    if minute >= 17 * 60 or minute < 2 * 60:
        return "globex_asia"
    if 2 * 60 <= minute < 8 * 60 + 30:
        return "london"
    if 8 * 60 + 30 <= minute < 12 * 60:
        return "ny_am"
    if 12 * 60 <= minute < 15 * 60:
        return "ny_pm"
    if 15 * 60 <= minute < 16 * 60:
        return "ny_late"
    return "maintenance"


def permitted_et(ts_utc: pd.Timestamp) -> bool:
    local = ts_utc.tz_convert(ET)
    minute = local.hour * 60 + local.minute
    return minute >= 18 * 60 or minute < 15 * 60


def trading_deadline(ts_utc: pd.Timestamp) -> pd.Timestamp | None:
    """Return the next applicable 15:00 ET deadline for an eligible timestamp."""
    if not permitted_et(ts_utc):
        return None
    local = ts_utc.tz_convert(ET)
    d = local.date()
    if local.hour >= 18:
        d = d + timedelta(days=1)
    deadline_local = pd.Timestamp(datetime.combine(d, time(15, 0), tzinfo=ET))
    return deadline_local.tz_convert("UTC")


def eligible_signal_entry_pair(signal_time: pd.Timestamp, entry_time: pd.Timestamp) -> tuple[bool, pd.Timestamp | None]:
    if classify_ct(signal_time) == "london" or classify_ct(entry_time) == "london":
        return False, None
    if not permitted_et(signal_time) or not permitted_et(entry_time):
        return False, None
    ds = trading_deadline(signal_time)
    de = trading_deadline(entry_time)
    if ds is None or de is None or ds != de:
        return False, None
    return True, ds


def build_candidates(features: pd.DataFrame) -> list[Candidate]:
    candidates: list[Candidate] = []
    n = len(features)
    for i in np.flatnonzero(features["signal_passes_keltner"].to_numpy(dtype=bool)):
        entry_i = int(i) + 1
        if entry_i >= n:
            continue
        signal_time = features.index[int(i)]
        entry_time = features.index[entry_i]
        eligible, deadline = eligible_signal_entry_pair(signal_time, entry_time)
        if not eligible or deadline is None:
            continue
        candidates.append(
            Candidate(
                signal_idx=int(i),
                entry_idx=entry_i,
                signal_time=signal_time,
                entry_time=entry_time,
                signal_z=float(features.iloc[int(i)]["z"]),
                previous_z=float(features.iloc[int(i) - 1]["z"]),
                signal_keltner=float(features.iloc[int(i)]["keltner"]),
                deadline=deadline,
            )
        )
    return candidates


def build_4h_true_range(bars: pd.DataFrame) -> pd.DataFrame:
    """Return one row per nonempty UTC-aligned 4h bucket with causal availability time."""
    bars = validate_bars(bars)
    if bars.empty:
        return pd.DataFrame(
            columns=["source_start", "source_end", "high", "low", "close", "previous_close", "tr", "available_at"]
        )
    bucket_start = bars.index.floor("4h")
    work = bars.assign(_bucket=bucket_start)
    grouped = work.groupby("_bucket", sort=True, observed=True)
    agg = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        rows=("close", "size"),
    )
    agg.index.name = "source_start"
    agg["source_start"] = agg.index
    agg["source_end"] = agg["source_start"] + pd.Timedelta(hours=4)
    agg["previous_close"] = agg["close"].shift(1)
    a = agg["high"] - agg["low"]
    b = (agg["high"] - agg["previous_close"]).abs()
    c = (agg["low"] - agg["previous_close"]).abs()
    agg["tr"] = pd.concat([a, b, c], axis=1).max(axis=1, skipna=True)
    agg["available_at"] = agg["source_end"]
    return agg.reset_index(drop=True)[
        ["source_start", "source_end", "high", "low", "close", "previous_close", "tr", "available_at"]
    ]


def atr_for_entry(atr_table: pd.DataFrame, entry_time: pd.Timestamp) -> pd.Series | None:
    if atr_table.empty:
        return None
    avail = atr_table["available_at"] <= entry_time
    if not bool(avail.any()):
        return None
    row = atr_table.loc[avail].iloc[-1]
    tr = float(row["tr"])
    if not isfinite(tr) or tr <= 0.0:
        return None
    return row


def ceil_to_tick(price: float, tick: float = TICK_SIZE) -> float:
    p = Decimal(str(price))
    t = Decimal(str(tick))
    ticks = (p / t).to_integral_value(rounding=ROUND_CEILING)
    return float(ticks * t)


def bracket(entry_price: float, atr: float) -> tuple[float, float, float, float, float] | None:
    stop_unrounded = float(entry_price - atr)
    target_unrounded = float(entry_price + TARGET_R * atr)
    stop = ceil_to_tick(stop_unrounded)
    target = ceil_to_tick(target_unrounded)
    risk = float(entry_price - stop)
    if not isfinite(risk) or risk <= 0.0:
        return None
    return stop_unrounded, stop, target_unrounded, target, risk


def _entry_year_et(ts: pd.Timestamp) -> int:
    return int(ts.tz_convert(ET).year)


def sample_label(year: int) -> str:
    if year in IS_YEARS:
        return "IS"
    if year in OOS_YEARS:
        return "OOS"
    return "UNLABELED"


def _accounting(
    entry_time: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    initial_risk: float,
    stop_price: float,
) -> tuple[float, float, float, float, int, str]:
    year = _entry_year_et(entry_time)
    if exit_reason == "TP":
        gross_r = TARGET_R
    elif exit_reason == "SL":
        if exit_price < stop_price:
            gross_r = (exit_price - entry_price) / initial_risk
        else:
            gross_r = -1.0
    elif exit_reason == "FLAT_1500":
        gross_r = (exit_price - entry_price) / initial_risk
    else:
        raise ValueError(f"unknown exit reason {exit_reason}")
    cost_points = 0.0 if year < COST_CHANGE_YEAR else POST_2023_COST_POINTS
    cost_r = cost_points / initial_risk
    net_r = gross_r - cost_r
    return float(gross_r), float(cost_points), float(cost_r), float(net_r), year, sample_label(year)


def simulate_trade(
    bars: pd.DataFrame,
    candidate: Candidate,
    atr_row: pd.Series,
) -> Trade | None:
    entry = float(bars.iloc[candidate.entry_idx]["open"])
    atr = float(atr_row["tr"])
    bg = bracket(entry, atr)
    if bg is None:
        return None
    stop_unrounded, stop, target_unrounded, target, risk = bg

    idx = bars.index
    deadline = candidate.deadline
    # Identify whether an exact deadline row exists and the final printed row before deadline.
    exact_pos = idx.get_indexer([deadline])[0]
    pre_deadline_positions = np.flatnonzero((idx >= candidate.entry_time) & (idx < deadline))
    if len(pre_deadline_positions) == 0:
        return None  # impossible for a valid entry, but fail closed.
    last_pre_pos = int(pre_deadline_positions[-1])

    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None

    end_pos = int(exact_pos) if exact_pos >= 0 else last_pre_pos
    for pos in range(candidate.entry_idx, end_pos + 1):
        ts = idx[pos]
        if ts > deadline:
            break
        row = bars.iloc[pos]

        if ts == deadline:
            exit_time = ts
            exit_price = float(row["open"])
            exit_reason = "FLAT_1500"
            break

        o = float(row["open"])
        lo = float(row["low"])
        hi = float(row["high"])

        if o <= stop:
            exit_time = ts
            exit_price = o
            exit_reason = "SL"
            break
        if o >= target:
            exit_time = ts
            exit_price = target
            exit_reason = "TP"
            break
        if lo <= stop:
            exit_time = ts
            exit_price = stop
            exit_reason = "SL"
            break
        if hi >= target:
            exit_time = ts
            exit_price = target
            exit_reason = "TP"
            break

        if exact_pos < 0 and pos == last_pre_pos:
            exit_time = ts
            exit_price = float(row["close"])
            exit_reason = "FLAT_1500"
            break

    if exit_time is None or exit_price is None or exit_reason is None:
        raise AssertionError("accepted trade did not resolve by its own deadline")

    gross_r, cost_points, cost_r, net_r, year, sample = _accounting(
        candidate.entry_time, entry, exit_price, exit_reason, risk, stop
    )
    return Trade(
        signal_time=candidate.signal_time,
        entry_time=candidate.entry_time,
        entry_price=entry,
        signal_z=candidate.signal_z,
        previous_z=candidate.previous_z,
        signal_keltner=candidate.signal_keltner,
        atr_source_start=atr_row["source_start"],
        atr_source_end=atr_row["source_end"],
        atr_value=atr,
        stop_unrounded=stop_unrounded,
        stop_price=stop,
        target_unrounded=target_unrounded,
        target_price=target,
        initial_risk=risk,
        deadline=deadline,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        gross_R=gross_r,
        cost_points=cost_points,
        cost_R=cost_r,
        net_R=net_r,
        entry_year=year,
        sample=sample,
    )


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
    return pd.DataFrame([asdict(t) for t in trades])


def validate_trade_ledger(trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    if not trades["entry_time"].is_monotonic_increasing:
        raise AssertionError("entry times are not monotonic")
    if not (trades["exit_time"] >= trades["entry_time"]).all():
        raise AssertionError("exit before entry")
    if not trades["exit_reason"].isin(["TP", "SL", "FLAT_1500"]).all():
        raise AssertionError("invalid exit reason")
    if not (trades["initial_risk"] > 0).all():
        raise AssertionError("nonpositive initial risk")
    if not (trades["atr_value"] > 0).all():
        raise AssertionError("nonpositive ATR")
    if not (trades["atr_source_end"] <= trades["entry_time"]).all():
        raise AssertionError("future ATR source used")
    if not (trades["exit_time"] <= trades["deadline"]).all():
        raise AssertionError("trade consumed data after deadline")
    # A previous trade must be closed strictly before the next accepted entry under the
    # frozen priority rule. Equal timestamps would mean the next signal should have been rejected.
    if len(trades) > 1:
        prev_exit = trades["exit_time"].iloc[:-1].reset_index(drop=True)
        next_entry = trades["entry_time"].iloc[1:].reset_index(drop=True)
        if not (prev_exit < next_entry).all():
            raise AssertionError("accepted trades overlap or share an exit/entry timestamp")


def run_backtest(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = validate_bars(bars)
    features = build_features(bars)
    candidates = build_candidates(features)
    atr_table = build_4h_true_range(bars)

    accepted: list[Trade] = []
    open_until: pd.Timestamp | None = None
    for c in candidates:
        # Frozen priority: if previous trade is still open immediately before this row,
        # reject, including equality with a previous exit timestamp.
        if open_until is not None and c.entry_time <= open_until:
            continue
        atr_row = atr_for_entry(atr_table, c.entry_time)
        if atr_row is None:
            continue
        t = simulate_trade(bars, c, atr_row)
        if t is None:
            continue
        accepted.append(t)
        open_until = t.exit_time

    trades = trades_to_frame(accepted)
    validate_trade_ledger(trades)
    return trades, features


def summarize(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {"trades": 0, "wins": 0, "loss_or_zero": 0, "win_rate": np.nan, "profit_factor": np.nan, "net_R": 0.0}
    net = trades["net_R"].astype(float)
    pos = float(net[net > 0].sum())
    neg = float(net[net < 0].sum())
    wins = int((net > 0).sum())
    if neg < 0:
        pf = pos / abs(neg)
    elif pos > 0:
        pf = np.inf
    else:
        pf = np.nan
    return {
        "trades": int(len(net)),
        "wins": wins,
        "loss_or_zero": int((net <= 0).sum()),
        "win_rate": 100.0 * wins / len(net),
        "profit_factor": float(pf),
        "net_R": float(net.sum()),
    }
