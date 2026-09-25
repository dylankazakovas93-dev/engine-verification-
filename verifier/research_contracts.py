"""Structural contract for untrusted ML / conditional-edge research adapters.

This module is deliberately independent of the frozen NQ strategy verifier. It reuses the
repository's :class:`~verifier.schema.Finding` type and verdict semantics, but it never imports
or modifies frozen strategy code.

Research adapter contract (see ``docs/RESEARCH_VERIFICATION.md``)::

    events(bars) -> DataFrame[event_id, event_time, direction, ...]
    features(bars, events) -> DataFrame[event_id, feature_asof_time, <features>...]
    targets(bars, events) -> DataFrame[event_id, target_start, target_end, <labels>...]
    fit_predict_fold(features, targets, train_ids, validation_ids, target_name)
        -> DataFrame[event_id, prediction]          # optional; absent => ml_leakage UNVERIFIED

Clock convention: a bar row stamped ``t`` is treated as known at ``t`` (identical to the
existing truncation/mutation harness, which keeps rows ``index < cutoff``).
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import itertools
from pathlib import Path
import re
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

from .schema import Finding

FAMILY = "research_contract"
PROVEN = "PROVEN FAILURE"
HIGH = "HIGH-RISK / MANUAL REVIEW"

EVENT_REQUIRED = ("event_id", "event_time", "direction")
FEATURE_KEYS = ("event_id", "feature_asof_time")
TARGET_KEYS = ("event_id", "target_start", "target_end")
TARGET_NAME_COL = "target_name"
TARGET_VALUE_COL = "target_value"
REQUIRED_FUNCTIONS = ("events", "features", "targets")
MODEL_FUNCTION = "fit_predict_fold"
# An adapter may set this module attribute to True to document that features() intentionally
# omits some events (for example, warm-up events). Omitted events are then reported as INFO and
# excluded from modelling instead of failing the contract.
ALLOW_MISSING_FEATURES_ATTR = "FEATURES_ALLOW_MISSING_EVENTS"

# ---------------------------------------------------------------------------
# Feature/target firewall: centralized reserved-name logic.
# ---------------------------------------------------------------------------
# A column name is split into lowercase tokens at non-alphanumerics, camelCase, and
# letter/digit boundaries ("fwdReturn60m" -> fwd, return, 60, m). Matching is on WHOLE
# tokens, so "maestro", "name", "futures_basis", "targeted_vol" and "relabelled" are not
# reserved, while "mae_20", "Future_Close", "label" and "fwd_ret" are.
RESERVED_TOKENS = frozenset({
    "target", "targets", "label", "labels", "future", "forward", "fwd", "mfe", "mae",
})
# Multi-token phrases whose individual tokens are innocent on their own.
RESERVED_PHRASES = (
    ("barrier", "result"), ("barrier", "hit"), ("barrier", "outcome"),
    ("next", "return"), ("next", "ret"),
)


def tokenize_column_name(name: Any) -> tuple[str, ...]:
    text = str(name)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", text)
    return tuple(token for token in re.split(r"[^0-9A-Za-z]+", text.lower()) if token)


def reserved_feature_reason(name: Any) -> str | None:
    """Return why ``name`` is reserved for labels/future information, or ``None``."""
    tokens = tokenize_column_name(name)
    for token in tokens:
        if token in RESERVED_TOKENS:
            return f"token {token!r} is reserved for future/label information"
    for phrase in RESERVED_PHRASES:
        width = len(phrase)
        for start in range(len(tokens) - width + 1):
            if tokens[start:start + width] == phrase:
                return f"phrase {'_'.join(phrase)!r} is reserved for future/label information"
    return None


# ---------------------------------------------------------------------------
# Adapter loading and pipeline execution
# ---------------------------------------------------------------------------
_LOAD_COUNTER = itertools.count()


def load_research_adapter(path: str | Path) -> Any:
    """Load a fresh module instance of an untrusted research adapter.

    Every call executes the source again under a unique module name so module-level caches from
    an earlier invocation cannot make a truncated/poisoned run silently reuse full-sample output.
    """
    p = Path(path).resolve()
    name = f"research_candidate_adapter_{next(_LOAD_COUNTER)}"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load research adapter {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def adapter_capabilities(adapter: Any) -> dict[str, bool]:
    return {name: callable(getattr(adapter, name, None)) for name in (*REQUIRED_FUNCTIONS, MODEL_FUNCTION)}


@dataclass
class ResearchTables:
    events: pd.DataFrame | None
    features: pd.DataFrame | None
    targets: pd.DataFrame | None


def _frame_or_error(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"research adapter {name}() must return a pandas DataFrame, got {type(value).__name__}")
    return value.copy().reset_index(drop=True)


def run_pipeline(adapter: Any, bars: pd.DataFrame) -> ResearchTables:
    """Run events -> features -> targets on one bar frame. Missing functions yield ``None``.

    The verifier passes its own copies so a candidate that mutates its input cannot alter the
    verifier's reference bars.
    """
    caps = adapter_capabilities(adapter)
    if not caps["events"]:
        return ResearchTables(None, None, None)
    events = _frame_or_error(adapter.events(bars.copy()), "events")
    features = _frame_or_error(adapter.features(bars.copy(), events.copy()), "features") if caps["features"] else None
    targets = _frame_or_error(adapter.targets(bars.copy(), events.copy()), "targets") if caps["targets"] else None
    return ResearchTables(events, features, targets)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _f(check: str, status: str, message: str, *, classification: str | None = None,
       evidence: dict | None = None, rows: list[int] | None = None) -> Finding:
    return Finding(check, status, message, rows=rows, family=FAMILY, classification=classification, evidence=evidence)


def _is_tz_aware(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.DatetimeTZDtype)


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _sample_ids(values: Any, limit: int = 10) -> list[str]:
    return [str(v) for v in list(values)[:limit]]


def _missing_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return [c for c in required if c not in frame.columns]


def _time_checks(frame: pd.DataFrame, column: str, prefix: str) -> tuple[list[Finding], bool]:
    """Timezone-awareness and null checks for one timestamp column; returns (findings, usable)."""
    findings: list[Finding] = []
    series = frame[column]
    aware = _is_tz_aware(series)
    findings.append(_f(
        f"{prefix}_{column}_timezone", "PASS" if aware else "FAIL",
        f"{column} is timezone-aware ({series.dtype})" if aware else f"{column} must have a timezone-aware datetime dtype; got {series.dtype}",
        classification=None if aware else PROVEN,
    ))
    parsed = to_utc(series)
    nulls = parsed.isna()
    findings.append(_f(
        f"{prefix}_{column}_valid", "FAIL" if nulls.any() else "PASS",
        f"{int(nulls.sum())} null/unparseable {column} values",
        rows=np.flatnonzero(nulls.to_numpy())[:20].tolist() if nulls.any() else None,
        classification=PROVEN if nulls.any() else None,
    ))
    return findings, aware and not nulls.any()


def event_time_lookup(events: pd.DataFrame) -> pd.Series:
    """Map event_id -> UTC event_time for a structurally valid events table."""
    return pd.Series(to_utc(events["event_time"]).array, index=pd.Index(events["event_id"].to_numpy(), dtype=object))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
def audit_events(events: pd.DataFrame, bars_index: pd.DatetimeIndex | None = None) -> tuple[list[Finding], bool]:
    """Return findings and whether downstream stages may rely on the events table."""
    findings: list[Finding] = []
    missing = _missing_columns(events, EVENT_REQUIRED)
    if missing:
        return [_f("events_schema", "FAIL", f"events missing required columns {missing}", classification=PROVEN)], False
    findings.append(_f("events_schema", "PASS", f"events expose {list(EVENT_REQUIRED)}; {len(events)} rows"))
    if events.empty:
        findings.append(_f("events_nonempty", "UNVERIFIED", "events() returned zero rows; nothing can be verified"))
        return findings, False

    ids = events["event_id"]
    null_ids = ids.isna()
    findings.append(_f(
        "events_id_present", "FAIL" if null_ids.any() else "PASS", f"{int(null_ids.sum())} null event_id values",
        rows=np.flatnonzero(null_ids.to_numpy())[:20].tolist() if null_ids.any() else None,
        classification=PROVEN if null_ids.any() else None,
    ))
    duplicated_ids = ids.duplicated(keep=False)
    findings.append(_f(
        "events_id_unique", "FAIL" if duplicated_ids.any() else "PASS",
        f"{int(ids.duplicated().sum())} duplicate event_id values" + (f"; e.g. {_sample_ids(ids[duplicated_ids].unique())}" if duplicated_ids.any() else ""),
        rows=np.flatnonzero(duplicated_ids.to_numpy())[:20].tolist() if duplicated_ids.any() else None,
        classification=PROVEN if duplicated_ids.any() else None,
    ))
    try:
        duplicate_rows = events.astype(str).duplicated(keep=False)
    except Exception:  # pragma: no cover - exotic unhashable payloads
        duplicate_rows = pd.Series(False, index=events.index)
    findings.append(_f(
        "events_no_duplicate_rows", "FAIL" if duplicate_rows.any() else "PASS",
        f"{int(duplicate_rows.sum())} rows are exact duplicates of another event row",
        rows=np.flatnonzero(duplicate_rows.to_numpy())[:20].tolist() if duplicate_rows.any() else None,
        classification=PROVEN if duplicate_rows.any() else None,
    ))
    time_findings, time_ok = _time_checks(events, "event_time", "events")
    findings += time_findings
    null_direction = events["direction"].isna()
    findings.append(_f(
        "events_direction_present", "FAIL" if null_direction.any() else "PASS",
        f"{int(null_direction.sum())} null direction values; observed values {sorted(map(str, events['direction'].dropna().unique()))[:10]}",
        classification=PROVEN if null_direction.any() else None,
    ))
    usable = time_ok and not null_ids.any() and not duplicated_ids.any()
    if time_ok:
        times = to_utc(events["event_time"])
        ordered = bool(times.is_monotonic_increasing)
        backwards = np.flatnonzero((times.diff() < pd.Timedelta(0)).to_numpy())
        findings.append(_f(
            "events_chronological_order", "PASS" if ordered else "FAIL",
            "events are returned in nondecreasing event_time order" if ordered
            else f"events are not in chronological order; first backwards rows {backwards[:10].tolist()}",
            rows=backwards[:20].tolist() if not ordered else None, classification=None if ordered else PROVEN,
        ))
        if bars_index is not None and len(bars_index):
            first, last = bars_index[0], bars_index[-1]
            outside = (times < first) | (times > last)
            findings.append(_f(
                "events_within_data", "FAIL" if outside.any() else "PASS",
                f"{int(outside.sum())} events are timestamped outside the supplied bars [{first.isoformat()}, {last.isoformat()}]"
                + (f"; e.g. {_sample_ids(events.loc[outside, 'event_id'])}" if outside.any() else ""),
                rows=np.flatnonzero(outside.to_numpy())[:20].tolist() if outside.any() else None,
                classification=PROVEN if outside.any() else None,
            ))
            usable &= not outside.any()
    return findings, usable


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in FEATURE_KEYS]


def audit_features(
    features: pd.DataFrame, events: pd.DataFrame, *, allow_missing_events: bool = False,
) -> tuple[list[Finding], bool]:
    findings: list[Finding] = []
    missing = _missing_columns(features, FEATURE_KEYS)
    if missing:
        return [_f("features_schema", "FAIL", f"features missing required columns {missing}", classification=PROVEN)], False
    columns = feature_columns(features)
    findings.append(_f(
        "features_schema", "PASS" if columns else "FAIL",
        f"{len(columns)} candidate feature columns; {len(features)} rows" if columns else "features() returned no explanatory feature columns",
        classification=None if columns else PROVEN,
    ))
    ids = features["event_id"]
    dup = ids.duplicated(keep=False)
    findings.append(_f(
        "features_one_row_per_event", "FAIL" if dup.any() else "PASS",
        f"{int(ids.duplicated().sum())} duplicate feature event_id rows" + (f"; e.g. {_sample_ids(ids[dup].unique())}" if dup.any() else ""),
        rows=np.flatnonzero(dup.to_numpy())[:20].tolist() if dup.any() else None, classification=PROVEN if dup.any() else None,
    ))
    event_ids = set(events["event_id"].tolist())
    unknown = ~ids.isin(event_ids)
    findings.append(_f(
        "features_known_event_ids", "FAIL" if unknown.any() else "PASS",
        f"{int(unknown.sum())} feature rows reference event_ids absent from events" + (f"; e.g. {_sample_ids(ids[unknown])}" if unknown.any() else ""),
        rows=np.flatnonzero(unknown.to_numpy())[:20].tolist() if unknown.any() else None, classification=PROVEN if unknown.any() else None,
    ))
    absent = sorted(event_ids - set(ids.tolist()), key=str)
    if absent and allow_missing_events:
        findings.append(_f(
            "features_cover_events", "INFO",
            f"{len(absent)} events have no feature row; adapter documents {ALLOW_MISSING_FEATURES_ATTR}=True, so they are excluded from modelling; e.g. {_sample_ids(absent)}",
        ))
    else:
        findings.append(_f(
            "features_cover_events", "FAIL" if absent else "PASS",
            f"{len(absent)} events have no feature row" + (f"; e.g. {_sample_ids(absent)}" if absent else ""),
            classification=PROVEN if absent else None,
        ))
    time_findings, time_ok = _time_checks(features, "feature_asof_time", "features")
    findings += time_findings
    usable = bool(columns) and time_ok and not dup.any() and not unknown.any() and (allow_missing_events or not absent)
    if time_ok and not unknown.any():
        event_time = event_time_lookup(events)
        asof = to_utc(features["feature_asof_time"])
        matched_event_time = pd.Series(event_time.reindex(pd.Index(ids.to_numpy(), dtype=object)).array, index=features.index)
        late = asof > matched_event_time
        findings.append(_f(
            "features_asof_not_after_event", "FAIL" if late.any() else "PASS",
            f"{int(late.sum())} feature rows have feature_asof_time > event_time" + (
                f"; e.g. {[(str(i), a.isoformat(), e.isoformat()) for i, a, e in zip(ids[late][:5], asof[late][:5], matched_event_time[late][:5])]}" if late.any() else ""
            ),
            rows=np.flatnonzero(late.to_numpy())[:20].tolist() if late.any() else None, classification=PROVEN if late.any() else None,
        ))
        usable &= not late.any()
    return findings, usable


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def target_layout(targets: pd.DataFrame) -> str:
    """``long`` when rows are keyed by (event_id, target_name) with a target_value column."""
    return "long" if TARGET_NAME_COL in targets.columns else "wide"


def target_key_columns(targets: pd.DataFrame) -> list[str]:
    keys = list(TARGET_KEYS)
    if target_layout(targets) == "long":
        keys.append(TARGET_NAME_COL)
    return keys


def target_value_columns(targets: pd.DataFrame) -> list[str]:
    keys = set(target_key_columns(targets))
    return [c for c in targets.columns if c not in keys]


def available_target_names(targets: pd.DataFrame) -> list[str]:
    if target_layout(targets) == "long":
        return sorted(map(str, targets[TARGET_NAME_COL].dropna().unique()))
    return [str(c) for c in target_value_columns(targets)]


def select_target(targets: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Return [event_id, target_start, target_end, value] for one named target (not dropped NaN)."""
    if target_layout(targets) == "long":
        if TARGET_VALUE_COL not in targets.columns:
            raise ValueError(f"long-format targets require a {TARGET_VALUE_COL!r} column")
        rows = targets[targets[TARGET_NAME_COL].astype(str) == str(target_name)]
        value = rows[TARGET_VALUE_COL]
    else:
        if target_name not in targets.columns or target_name in TARGET_KEYS:
            raise ValueError(f"target {target_name!r} not produced; available {available_target_names(targets)}")
        rows = targets
        value = rows[target_name]
    if rows.empty:
        raise ValueError(f"target {target_name!r} not produced; available {available_target_names(targets)}")
    return pd.DataFrame({
        "event_id": rows["event_id"].reset_index(drop=True),
        "target_start": to_utc(rows["target_start"]).reset_index(drop=True),
        "target_end": to_utc(rows["target_end"]).reset_index(drop=True),
        "value": pd.to_numeric(value, errors="coerce").astype(float).reset_index(drop=True),
    })


def audit_targets(
    targets: pd.DataFrame, events: pd.DataFrame, bars_index: pd.DatetimeIndex | None = None,
) -> tuple[list[Finding], bool]:
    findings: list[Finding] = []
    missing = _missing_columns(targets, TARGET_KEYS)
    if missing:
        return [_f("targets_schema", "FAIL", f"targets missing required columns {missing}", classification=PROVEN)], False
    layout = target_layout(targets)
    if layout == "long" and TARGET_VALUE_COL not in targets.columns:
        return [_f("targets_schema", "FAIL", f"long-format targets (with {TARGET_NAME_COL!r}) require {TARGET_VALUE_COL!r}", classification=PROVEN)], False
    values = target_value_columns(targets)
    findings.append(_f(
        "targets_schema", "PASS" if values else "FAIL",
        f"{layout} layout; label columns {values}; {len(targets)} rows" if values else "targets() returned no label columns",
        classification=None if values else PROVEN,
    ))
    ids = targets["event_id"]
    unknown = ~ids.isin(set(events["event_id"].tolist()))
    findings.append(_f(
        "targets_known_event_ids", "FAIL" if unknown.any() else "PASS",
        f"{int(unknown.sum())} target rows reference event_ids absent from events" + (f"; e.g. {_sample_ids(ids[unknown])}" if unknown.any() else ""),
        rows=np.flatnonzero(unknown.to_numpy())[:20].tolist() if unknown.any() else None, classification=PROVEN if unknown.any() else None,
    ))
    key = ["event_id", TARGET_NAME_COL] if layout == "long" else ["event_id"]
    dup = targets.duplicated(subset=key, keep=False)
    findings.append(_f(
        "targets_unique_key", "FAIL" if dup.any() else "PASS",
        f"{int(targets.duplicated(subset=key).sum())} duplicate {tuple(key)} target rows"
        + ("" if layout == "long" else f"; multiple rows per event require an explicit {TARGET_NAME_COL!r} key"),
        rows=np.flatnonzero(dup.to_numpy())[:20].tolist() if dup.any() else None, classification=PROVEN if dup.any() else None,
    ))
    ok = not unknown.any() and not dup.any() and bool(values)
    for column in ("target_start", "target_end"):
        time_findings, time_ok = _time_checks(targets, column, "targets")
        findings += time_findings
        ok &= time_ok
    if ok:
        start, end = to_utc(targets["target_start"]), to_utc(targets["target_end"])
        event_time = pd.Series(event_time_lookup(events).reindex(pd.Index(ids.to_numpy(), dtype=object)).array, index=targets.index)
        early = start < event_time
        findings.append(_f(
            "targets_start_not_before_event", "FAIL" if early.any() else "PASS",
            f"{int(early.sum())} target rows have target_start < event_time" + (f"; e.g. {_sample_ids(ids[early])}" if early.any() else ""),
            rows=np.flatnonzero(early.to_numpy())[:20].tolist() if early.any() else None, classification=PROVEN if early.any() else None,
        ))
        inverted = end < start
        findings.append(_f(
            "targets_end_not_before_start", "FAIL" if inverted.any() else "PASS",
            f"{int(inverted.sum())} target rows have target_end < target_start" + (f"; e.g. {_sample_ids(ids[inverted])}" if inverted.any() else ""),
            rows=np.flatnonzero(inverted.to_numpy())[:20].tolist() if inverted.any() else None, classification=PROVEN if inverted.any() else None,
        ))
        ok &= not early.any() and not inverted.any()
        if bars_index is not None and len(bars_index):
            beyond = end > bars_index[-1]
            findings.append(_f(
                "targets_resolved_within_data", "FAIL" if beyond.any() else "PASS",
                f"{int(beyond.sum())} targets claim target_end after the final bar {bars_index[-1].isoformat()}; unresolved targets must be omitted"
                + (f"; e.g. {_sample_ids(ids[beyond])}" if beyond.any() else ""),
                rows=np.flatnonzero(beyond.to_numpy())[:20].tolist() if beyond.any() else None, classification=PROVEN if beyond.any() else None,
            ))
            ok &= not beyond.any()
    return findings, ok


def audit_target_selection(targets: pd.DataFrame, target_name: str | None) -> tuple[list[Finding], bool]:
    if target_name is None:
        return [_f("target_selection", "UNVERIFIED", "no --target supplied; walk-forward modelling and ML leakage cannot be audited")], False
    try:
        selected = select_target(targets, target_name)
    except ValueError as exc:
        return [_f("target_selection", "FAIL", str(exc), classification=PROVEN)], False
    finite = np.isfinite(selected["value"].to_numpy(float))
    status = "PASS" if finite.any() else "FAIL"
    return [_f(
        "target_selection", status,
        f"target {target_name!r}: {int(finite.sum())} finite labels, {int((~finite).sum())} nonfinite/non-numeric (excluded from modelling)",
        classification=None if finite.any() else PROVEN,
    )], bool(finite.any())


# ---------------------------------------------------------------------------
# Feature/target firewall
# ---------------------------------------------------------------------------
def audit_firewall(features: pd.DataFrame, targets: pd.DataFrame | None, *, min_rows: int = 20) -> list[Finding]:
    """Reject label/future columns in the feature frame.

    1. reserved names (token-based, see :func:`reserved_feature_reason`)  -> FAIL
    2. feature column named identically to any target label column        -> FAIL
    3. feature column exactly equal to a non-constant target label column  -> FAIL (exact copy)
    4. |corr(feature, label)| >= 0.9999 on >= ``min_rows`` rows           -> WARN (near-linear copy)
    Checks 3/4 are value-based and cannot see transformed copies; research causality is the
    stronger dynamic test for those.
    """
    findings: list[Finding] = []
    columns = feature_columns(features)
    reserved = {str(c): reason for c in columns if (reason := reserved_feature_reason(c))}
    findings.append(_f(
        "firewall_reserved_names", "FAIL" if reserved else "PASS",
        f"feature frame contains reserved label/future columns: {reserved}" if reserved else f"none of {len(columns)} feature columns match reserved label/future patterns",
        classification=PROVEN if reserved else None, evidence={"reserved": reserved} if reserved else None,
    ))
    if targets is None or not set(TARGET_KEYS).issubset(targets.columns):
        findings.append(_f("firewall_target_values", "UNVERIFIED", "targets unavailable; feature/target value firewall not evaluated"))
        return findings
    label_columns = [c for c in target_value_columns(targets) if c != TARGET_NAME_COL]
    overlap = sorted(str(c) for c in set(map(str, columns)) & set(map(str, label_columns)))
    findings.append(_f(
        "firewall_target_column_overlap", "FAIL" if overlap else "PASS",
        f"feature columns share names with target label columns: {overlap}" if overlap else "no feature column shares a name with a target label column",
        classification=PROVEN if overlap else None,
    ))
    # Build per-label series keyed by event_id.
    labels: dict[str, pd.Series] = {}
    if target_layout(targets) == "long" and TARGET_VALUE_COL in targets.columns:
        for name, group in targets.groupby(TARGET_NAME_COL, sort=True):
            labels[f"{TARGET_NAME_COL}={name}"] = pd.Series(pd.to_numeric(group[TARGET_VALUE_COL], errors="coerce").to_numpy(float), index=group["event_id"].to_numpy())
    else:
        for name in label_columns:
            numeric = pd.to_numeric(targets[name], errors="coerce")
            if numeric.notna().any():
                labels[str(name)] = pd.Series(numeric.to_numpy(float), index=targets["event_id"].to_numpy())
    exact: list[tuple[str, str]] = []
    near: list[tuple[str, str, float]] = []
    for column in columns:
        feature = pd.to_numeric(features[column], errors="coerce")
        if feature.notna().sum() < 2:
            continue
        fseries = pd.Series(feature.to_numpy(float), index=features["event_id"].to_numpy())
        for label_name, label in labels.items():
            if label.index.has_duplicates or fseries.index.has_duplicates:
                continue
            joined = pd.concat([fseries.rename("f"), label.rename("y")], axis=1, join="inner").dropna()
            if len(joined) < 2 or joined["y"].nunique() < 2:
                continue
            if np.allclose(joined["f"].to_numpy(), joined["y"].to_numpy(), rtol=0, atol=1e-12):
                exact.append((str(column), label_name))
                continue
            if len(joined) >= min_rows and joined["f"].nunique() > 1:
                corr = float(np.corrcoef(joined["f"].to_numpy(), joined["y"].to_numpy())[0, 1])
                if np.isfinite(corr) and abs(corr) >= 0.9999:
                    near.append((str(column), label_name, corr))
    findings.append(_f(
        "firewall_target_value_identity", "FAIL" if exact else "PASS",
        f"feature columns are exact copies of target labels: {exact}" if exact else f"no feature column equals any of {len(labels)} numeric target labels",
        classification=PROVEN if exact else None,
    ))
    findings.append(_f(
        "firewall_target_near_copy", "WARN" if near else "PASS",
        f"feature columns are near-linear copies of target labels (|corr|>=0.9999): {near}" if near else "no near-linear feature/label copies found",
        classification=HIGH if near else None,
    ))
    return findings


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def frames_identical(a: pd.DataFrame | None, b: pd.DataFrame | None, atol: float = 1e-12) -> tuple[bool, str]:
    from .causality import _equal  # reuse the frozen harness comparator
    if a is None or b is None:
        return (a is None and b is None), "one run returned no table"
    return _equal(a.reset_index(drop=True), b.reset_index(drop=True), atol)


def audit_determinism(first: ResearchTables, second: ResearchTables) -> list[Finding]:
    findings = []
    for name in ("events", "features", "targets"):
        a, b = getattr(first, name), getattr(second, name)
        if a is None and b is None:
            continue
        same, why = frames_identical(a, b)
        findings.append(_f(
            f"determinism_{name}", "PASS" if same else "FAIL",
            f"two identical {name}() invocations on the same bars: {why}",
            classification=None if same else PROVEN,
        ))
    return findings


def audit_research_contract(
    adapter_factory: Callable[[], Any], bars: pd.DataFrame, *, target_name: str | None,
) -> tuple[list[Finding], ResearchTables | None, dict[str, bool]]:
    """Run the full contract audit.

    Returns findings, the full-sample tables (``None`` if the adapter cannot run), and a dict of
    readiness flags: events_ok, features_ok, targets_ok, target_ok, model_available.
    """
    adapter = adapter_factory()
    caps = adapter_capabilities(adapter)
    ready = {"events_ok": False, "features_ok": False, "targets_ok": False, "target_ok": False,
             "model_available": caps[MODEL_FUNCTION]}
    findings: list[Finding] = [_f(
        "adapter_interface", "PASS" if all(caps[n] for n in REQUIRED_FUNCTIONS) else "UNVERIFIED",
        f"exposed functions: {caps}" + ("" if all(caps[n] for n in REQUIRED_FUNCTIONS) else f"; required {list(REQUIRED_FUNCTIONS)} missing -> dependent checks UNVERIFIED"),
        evidence={"capabilities": caps},
    )]
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        findings.append(_f("bars_index", "FAIL", "bars must have a timezone-aware DatetimeIndex", classification=PROVEN))
        return findings, None, ready
    try:
        tables = run_pipeline(adapter, bars)
        repeat = run_pipeline(adapter_factory(), bars)
    except Exception as exc:
        findings.append(_f("candidate_execution", "FAIL", f"{type(exc).__name__}: {exc}", classification=PROVEN))
        return findings, None, ready
    if tables.events is None:
        findings.append(_f("events_contract", "UNVERIFIED", "adapter exposes no events(bars)"))
        return findings, tables, ready
    event_findings, ready["events_ok"] = audit_events(tables.events, bars.index)
    findings += event_findings
    findings += audit_determinism(tables, repeat)
    if not ready["events_ok"]:
        findings.append(_f("downstream_contract", "UNVERIFIED", "events table unusable; feature/target contract checks not evaluated"))
        return findings, tables, ready
    allow_missing = bool(getattr(adapter, ALLOW_MISSING_FEATURES_ATTR, False))
    if tables.features is None:
        findings.append(_f("features_contract", "UNVERIFIED", "adapter exposes no features(bars, events)"))
    else:
        feature_findings, ready["features_ok"] = audit_features(tables.features, tables.events, allow_missing_events=allow_missing)
        findings += feature_findings
    if tables.targets is None:
        findings.append(_f("targets_contract", "UNVERIFIED", "adapter exposes no targets(bars, events)"))
    else:
        target_findings, ready["targets_ok"] = audit_targets(tables.targets, tables.events, bars.index)
        findings += target_findings
        if ready["targets_ok"]:
            selection, ready["target_ok"] = audit_target_selection(tables.targets, target_name)
            findings += selection
    if tables.features is not None and set(FEATURE_KEYS).issubset(tables.features.columns):
        findings += audit_firewall(tables.features, tables.targets if ready["targets_ok"] else None)
    return findings, tables, ready
