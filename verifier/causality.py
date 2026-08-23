from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .schema import Finding

STAGE_ORDER = (
    "features", "signals", "eligible_signals", "proposed_entries",
    "accepted_entries", "trades",
)
STAGE_TIME = {
    "features": "timestamp",
    "signals": "signal_time",
    "eligible_signals": "signal_time",
    "proposed_entries": "entry_time",
    "accepted_entries": "entry_time",
    "trades": "exit_time",
}
MODE_LIMITS = {"fast": 6, "standard": 16, "strong": 28}


@dataclass(frozen=True)
class CausalityCoverage:
    mode: str
    seed: int
    cutoffs: tuple[str, ...]
    stages: tuple[str, ...]
    truncation_comparisons: int
    mutation_comparisons: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "cutoff_count": len(self.cutoffs),
            "cutoffs": list(self.cutoffs),
            "stages": list(self.stages),
            "truncation_comparisons": self.truncation_comparisons,
            "mutation_comparisons": self.mutation_comparisons,
        }


def mutate_future(bars: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Drastically mutate future rows while retaining structurally valid OHLC."""
    x = bars.copy()
    mask = x.index >= cutoff
    count = int(mask.sum())
    if count == 0:
        return x
    scale = np.linspace(1.7, 2.3, count)
    original_open = x.loc[mask, "open"].to_numpy(float) if "open" in x else np.ones(count)
    original_close = x.loc[mask, "close"].to_numpy(float) if "close" in x else original_open
    mutated_open = original_open * scale
    mutated_close = original_close * scale[::-1]
    if "open" in x:
        x.loc[mask, "open"] = mutated_open
    if "close" in x:
        x.loc[mask, "close"] = mutated_close
    if {"open", "close", "high", "low"}.issubset(x.columns):
        x.loc[mask, "high"] = np.maximum(mutated_open, mutated_close) * 1.03
        x.loc[mask, "low"] = np.minimum(mutated_open, mutated_close) * 0.97
    return x


def _as_frame(value: Any, stage: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"adapter stage {stage} must return a pandas DataFrame or Series")
    frame = value.copy()
    if isinstance(frame.index, pd.DatetimeIndex) and "timestamp" not in frame.columns:
        frame.insert(0, "timestamp", frame.index)
    return frame.reset_index(drop=True)


def collect_stages(adapter: Any, bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Collect every canonical decision stage exposed by an adapter."""
    aggregate = getattr(adapter, "audit_stages", None)
    if callable(aggregate):
        raw = aggregate(bars)
        if not isinstance(raw, dict):
            raise TypeError("adapter audit_stages(bars) must return a dict of DataFrames")
        stages = {name: _as_frame(value, name) for name, value in raw.items() if name in STAGE_ORDER}
        if "trades" not in stages:
            stages["trades"] = _as_frame(adapter.run(bars), "trades")
        return stages
    stages: dict[str, pd.DataFrame] = {"trades": _as_frame(adapter.run(bars), "trades")}
    for name in STAGE_ORDER[:-1]:
        fn = getattr(adapter, name, None)
        if callable(fn):
            stages[name] = _as_frame(fn(bars), name)
    return stages


def _time_column(frame: pd.DataFrame, stage: str) -> str | None:
    preferred = STAGE_TIME[stage]
    if preferred in frame.columns:
        return preferred
    for name in ("decision_time", "timestamp", "signal_time", "entry_time", "exit_time"):
        if name in frame.columns:
            return name
    return None


def _canonical_before(frame: pd.DataFrame, stage: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy().reset_index(drop=True)
    x = frame.copy()
    time_col = _time_column(x, stage)
    if time_col is None:
        raise ValueError(f"stage {stage} has no canonical decision timestamp")
    x[time_col] = pd.to_datetime(x[time_col], utc=True, errors="coerce")
    known_at = x[time_col].copy()
    # A missing-boundary forced flat may be priced on the final pre-deadline row, but
    # that row is not legally identifiable as final until the declared boundary passes.
    if stage == "trades" and {"exit_reason", "deadline"}.issubset(x.columns):
        deadline = pd.to_datetime(x["deadline"], utc=True, errors="coerce")
        forced = x["exit_reason"].astype(str).str.contains(r"FLAT|FORCED|TIME", case=False, regex=True)
        known_at.loc[forced] = pd.concat([known_at.loc[forced], deadline.loc[forced]], axis=1).max(axis=1)
    return x[known_at < cutoff].reset_index(drop=True)


def _equal(a: pd.DataFrame, b: pd.DataFrame, atol: float = 1e-9) -> tuple[bool, str]:
    if list(a.columns) != list(b.columns):
        return False, f"columns differ: full={list(a.columns)} comparison={list(b.columns)}"
    if len(a) != len(b):
        return False, f"row count differs: full={len(a)} comparison={len(b)}"
    for column in a.columns:
        left, right = a[column], b[column]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            av = pd.to_numeric(left, errors="coerce").to_numpy(float)
            bv = pd.to_numeric(right, errors="coerce").to_numpy(float)
            equal = np.isclose(av, bv, rtol=0, atol=atol, equal_nan=True)
            if not bool(equal.all()):
                return False, f"numeric column {column!r} differs at rows {np.flatnonzero(~equal)[:10].tolist()}"
        else:
            equal = left.astype(str).to_numpy() == right.astype(str).to_numpy()
            if not bool(equal.all()):
                return False, f"column {column!r} differs at rows {np.flatnonzero(~equal)[:10].tolist()}"
    return True, "unchanged"


def generate_cutoffs(
    bars: pd.DataFrame,
    full_stages: dict[str, pd.DataFrame],
    *, mode: str = "strong", seed: int = 1729,
    explicit: list[pd.Timestamp] | None = None,
) -> list[pd.Timestamp]:
    if explicit is not None:
        return sorted({pd.Timestamp(x).tz_convert("UTC") for x in explicit})
    if mode not in MODE_LIMITS:
        raise ValueError(f"unknown causality mode {mode!r}; choose {sorted(MODE_LIMITS)}")
    n = len(bars)
    if n < 3:
        return []
    positions: set[int] = set(int(p) for p in np.linspace(max(1, n // 20), n - 1, 11, dtype=int))
    positions.update(p for p in (2, 10, 25, 50, 60, 100, 160, 200, 240) if p < n)
    rng = np.random.default_rng(seed)
    low = min(max(2, n // 25), n - 1)
    if low < n - 1:
        positions.update(int(p) for p in rng.integers(low, n, size=min(10, n - low)))
    normalized = bars.index.normalize().asi8
    transition = np.flatnonzero(
        (normalized[1:] != normalized[:-1]) | np.isin(bars.index[1:].hour, [0, 4, 8, 12, 15, 16, 18, 20])
    ) + 1
    positions.update(int(x) for x in transition[:12] if x < n)
    index_ns = bars.index.asi8
    for stage, frame in full_stages.items():
        time_col = _time_column(frame, stage)
        if frame.empty or time_col is None:
            continue
        times = pd.to_datetime(frame[time_col], utc=True, errors="coerce").dropna()
        if len(times) > 8:
            times = times.iloc[np.linspace(0, len(times) - 1, 8, dtype=int)]
        for ts in times:
            pos = int(np.searchsorted(index_ns, ts.value, side="right"))
            if 0 < pos < n:
                positions.add(pos)
    ordered = sorted(positions)
    limit = MODE_LIMITS[mode]
    if len(ordered) > limit:
        ordered = [ordered[i] for i in np.linspace(0, len(ordered) - 1, limit, dtype=int)]
    return [bars.index[p] for p in sorted(set(ordered))]


def audit_adapter_causality(
    adapter: Any, bars: pd.DataFrame, *, mode: str = "strong", seed: int = 1729,
    cutoffs: list[pd.Timestamp] | None = None, atol: float = 1e-9,
) -> tuple[list[Finding], CausalityCoverage]:
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        empty = CausalityCoverage(mode, seed, (), (), 0, 0)
        return [Finding("causality", "FAIL", "bars must have a timezone-aware DatetimeIndex", family="causality")], empty
    full = collect_stages(adapter, bars)
    chosen = generate_cutoffs(bars, full, mode=mode, seed=seed, explicit=cutoffs)
    exposed = tuple(name for name in STAGE_ORDER if name in full)
    findings: list[Finding] = []
    if not chosen:
        coverage = CausalityCoverage(mode, seed, (), exposed, 0, 0)
        return [Finding("causality", "UNVERIFIED", "no valid causality cutoffs", family="causality")], coverage
    for cutoff in chosen:
        prefix_stages = collect_stages(adapter, bars[bars.index < cutoff])
        mutated_stages = collect_stages(adapter, mutate_future(bars, cutoff))
        for stage in exposed:
            baseline = _canonical_before(full[stage], stage, cutoff)
            prefix = _canonical_before(prefix_stages.get(stage, pd.DataFrame()), stage, cutoff)
            mutated = _canonical_before(mutated_stages.get(stage, pd.DataFrame()), stage, cutoff)
            same_prefix, why_prefix = _equal(baseline, prefix, atol)
            same_mutated, why_mutated = _equal(baseline, mutated, atol)
            family = "feature_causality" if stage == "features" else "causality"
            findings.append(Finding(
                f"truncation_{stage}", "PASS" if same_prefix else "FAIL",
                f"cutoff={cutoff.isoformat()}: {why_prefix}", family=family,
                classification="PROVEN FAILURE" if not same_prefix else None,
            ))
            findings.append(Finding(
                f"future_mutation_{stage}", "PASS" if same_mutated else "FAIL",
                f"cutoff={cutoff.isoformat()}: {why_mutated}", family=family,
                classification="PROVEN FAILURE" if not same_mutated else None,
            ))
    if "features" not in exposed:
        findings.append(Finding(
            "feature_causality", "UNVERIFIED", "FEATURE-LEVEL CAUSALITY UNVERIFIED: adapter exposes no features stage",
            family="feature_causality",
        ))
    comparisons = len(chosen) * len(exposed)
    coverage = CausalityCoverage(mode, seed, tuple(x.isoformat() for x in chosen), exposed, comparisons, comparisons)
    return findings, coverage


def audit_causality(
    run_fn: Callable[[pd.DataFrame], pd.DataFrame], bars: pd.DataFrame,
    cutoffs: list[pd.Timestamp] | None = None,
) -> list[Finding]:
    """Backward-compatible ledger-only wrapper used by existing integrations."""
    class LegacyAdapter:
        run = staticmethod(run_fn)

    if len(bars) < 300 and cutoffs is None:
        return [Finding("causality", "WARN", "dataset too short for robust truncation audit", family="causality")]
    findings, _ = audit_adapter_causality(LegacyAdapter, bars, mode="fast", cutoffs=cutoffs)
    for finding in findings:
        if finding.check == "truncation_trades":
            finding.check = "truncation_invariance"
        elif finding.check == "future_mutation_trades":
            finding.check = "future_mutation"
    return findings
