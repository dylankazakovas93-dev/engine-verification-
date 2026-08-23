from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .schema import Finding

Resolution = Literal["RESOLVED_BY_1S", "STILL_AMBIGUOUS_WITHIN_1S", "DATA_UNAVAILABLE"]


@dataclass(frozen=True)
class ExecutionResolution:
    trade_row: int
    coarse_bar_time: str
    classification: Resolution
    first_event: str | None
    event_time: str | None
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_ambiguous_bar(
    seconds: pd.DataFrame,
    *,
    stop_price: float,
    target_price: float,
) -> tuple[Resolution, str | None, pd.Timestamp | None, str]:
    """Resolve only across seconds; never invent high/low ordering inside one second."""
    if seconds.empty:
        return "DATA_UNAVAILABLE", None, None, "no one-second rows in the coarse execution interval"
    required = {"open", "high", "low", "close"}
    if not required.issubset(seconds.columns):
        return "DATA_UNAVAILABLE", None, None, f"one-second data missing {sorted(required - set(seconds.columns))}"
    for ts, row in seconds.sort_index(kind="stable").iterrows():
        opening = float(row["open"])
        if opening <= stop_price:
            return "RESOLVED_BY_1S", "SL_OPEN", ts, "stop opening gap occurs first"
        if opening >= target_price:
            return "RESOLVED_BY_1S", "TP_OPEN", ts, "target opening gap occurs first (target price receives no improvement)"
        stop_hit = float(row["low"]) <= stop_price
        target_hit = float(row["high"]) >= target_price
        if stop_hit and target_hit:
            return "STILL_AMBIGUOUS_WITHIN_1S", "BOTH", ts, "stop and target occur inside the same one-second candle"
        if stop_hit:
            return "RESOLVED_BY_1S", "SL", ts, "stop occurs in an earlier one-second candle"
        if target_hit:
            return "RESOLVED_BY_1S", "TP", ts, "target occurs in an earlier one-second candle"
    return "DATA_UNAVAILABLE", None, None, "one-second rows do not reach either bracket level"


def audit_high_resolution_execution(
    trades: pd.DataFrame,
    coarse_bars: pd.DataFrame,
    second_bars: pd.DataFrame,
    *,
    coarse_interval: str | pd.Timedelta = "1min",
    contract_col: str | None = None,
) -> tuple[list[Finding], list[ExecutionResolution]]:
    required_trade = {"exit_time", "stop_price", "target_price"}
    if not required_trade.issubset(trades.columns):
        return [Finding("high_resolution_execution", "UNVERIFIED", f"trade ledger missing {sorted(required_trade - set(trades.columns))}", family="high_resolution")], []
    if not isinstance(coarse_bars.index, pd.DatetimeIndex) or not isinstance(second_bars.index, pd.DatetimeIndex):
        return [Finding("high_resolution_execution", "FAIL", "coarse and one-second data require DatetimeIndex", family="high_resolution")], []
    interval = pd.Timedelta(coarse_interval)
    resolutions: list[ExecutionResolution] = []
    for row_number, trade in trades.reset_index(drop=True).iterrows():
        exit_time = pd.Timestamp(trade["exit_time"])
        if exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize("UTC")
        else:
            exit_time = exit_time.tz_convert("UTC")
        if exit_time not in coarse_bars.index:
            continue
        coarse = coarse_bars.loc[exit_time]
        if isinstance(coarse, pd.DataFrame):
            coarse = coarse.iloc[0]
        stop, target = float(trade["stop_price"]), float(trade["target_price"])
        opening = float(coarse["open"])
        ambiguous = opening > stop and opening < target and float(coarse["low"]) <= stop and float(coarse["high"]) >= target
        if not ambiguous:
            continue
        subset = second_bars[(second_bars.index >= exit_time) & (second_bars.index < exit_time + interval)]
        if contract_col:
            if contract_col not in trade.index or contract_col not in subset.columns:
                classification, event, event_time, message = "DATA_UNAVAILABLE", None, None, "contract identity unavailable for comparable one-second filtering"
            else:
                subset = subset[subset[contract_col].astype(str) == str(trade[contract_col])]
                classification, event, event_time, message = resolve_ambiguous_bar(subset, stop_price=stop, target_price=target)
        else:
            classification, event, event_time, message = resolve_ambiguous_bar(subset, stop_price=stop, target_price=target)
        resolutions.append(ExecutionResolution(
            int(row_number), exit_time.isoformat(), classification, event,
            event_time.isoformat() if event_time is not None else None, message,
        ))
    if not resolutions:
        return [Finding("high_resolution_execution", "PASS", "no coarse execution bars touched both stop and target", family="high_resolution")], []
    unavailable = sum(r.classification == "DATA_UNAVAILABLE" for r in resolutions)
    ambiguous = sum(r.classification == "STILL_AMBIGUOUS_WITHIN_1S" for r in resolutions)
    resolved = sum(r.classification == "RESOLVED_BY_1S" for r in resolutions)
    status = "UNVERIFIED" if unavailable else "PASS"
    return [Finding(
        "high_resolution_execution", status,
        f"ambiguous coarse bars={len(resolutions)}; resolved_by_1s={resolved}; still_ambiguous_within_1s={ambiguous}; data_unavailable={unavailable}",
        family="high_resolution", evidence={"resolutions": [r.to_dict() for r in resolutions[:100]]},
    )], resolutions


def compare_one_second_aggregation(
    second_bars: pd.DataFrame,
    minute_bars: pd.DataFrame,
    *,
    contract_col: str | None = None,
    atol: float = 1e-9,
) -> list[Finding]:
    if contract_col is None:
        return [Finding(
            "high_resolution_aggregation", "UNVERIFIED",
            "aggregation equality not attempted without comparable contract identity/methodology",
            family="high_resolution",
        )]
    if contract_col not in second_bars or contract_col not in minute_bars:
        return [Finding("high_resolution_aggregation", "UNVERIFIED", "contract column absent from one or both datasets", family="high_resolution")]
    work = second_bars.copy()
    work["_minute"] = work.index.floor("1min")
    agg = work.groupby(["_minute", contract_col], sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).reset_index().rename(columns={"_minute": "timestamp"})
    expected = minute_bars.copy().reset_index(names="timestamp")
    merged = agg.merge(expected, on=["timestamp", contract_col], suffixes=("_1s", "_1m"))
    if merged.empty:
        return [Finding("high_resolution_aggregation", "UNVERIFIED", "no genuinely comparable timestamp/contract rows", family="high_resolution")]
    mismatch = np.zeros(len(merged), dtype=bool)
    for column in ("open", "high", "low", "close", "volume"):
        mismatch |= ~np.isclose(merged[f"{column}_1s"], merged[f"{column}_1m"], rtol=0, atol=atol, equal_nan=True)
    return [Finding(
        "high_resolution_aggregation", "FAIL" if mismatch.any() else "PASS",
        f"{int(mismatch.sum())}/{len(merged)} comparable minute/contract rows differ",
        rows=np.flatnonzero(mismatch)[:20].tolist() if mismatch.any() else None,
        family="high_resolution",
    )]
