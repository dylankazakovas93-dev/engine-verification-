"""Truncation and future-mutation causality for research adapters.

Reuses the frozen harness primitives unchanged:

* :func:`verifier.causality.mutate_future` - drastic, structurally valid future OHLC mutation
* :func:`verifier.causality.generate_cutoffs` - distributed / warm-up / session-transition /
  event-adjacent / fixed-seed random cutoffs
* :func:`verifier.causality._equal` - column-by-column comparator

For each cutoff ``T`` the full pipeline (events -> features -> targets) is run on
(1) the full bars, (2) bars strictly before ``T`` and (3) bars mutated at/after ``T``.

Compared rows:

* events       - ``event_time < T``
* features     - rows whose event is knowable (``event_time < T``)
* features_asof- rows with ``feature_asof_time < T <= event_time`` present in both full and
                 mutated runs (a feature must not depend on data after its declared as-of time)
* targets      - rows fully resolved (``target_end < T``)

Targets crossing ``T`` are not compared: they are allowed to depend on future data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .causality import _equal, generate_cutoffs, mutate_future
from .research_contracts import TARGET_NAME_COL, ResearchTables, run_pipeline, to_utc
from .schema import Finding

FAMILY = "research_causality"
PROVEN = "PROVEN FAILURE"
HIGH = "HIGH-RISK / MANUAL REVIEW"
# Extra cutoffs placed on the first bar strictly after sampled event / as-of / target_end
# timestamps. These are the tightest cutoffs at which a row first becomes comparable, so a
# one-bar peek beyond the declared time is exposed deterministically.
ADJACENT_PER_KIND = {"fast": 2, "standard": 4, "strong": 6}
STAGES = ("events", "features", "features_asof", "targets")


@dataclass
class ResearchCausalityCoverage:
    mode: str
    seed: int
    cutoffs: tuple[str, ...] = ()
    comparisons: dict[str, int] = field(default_factory=dict)
    rows_compared: dict[str, int] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    execution_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "seed": self.seed, "cutoff_count": len(self.cutoffs),
            "cutoffs": list(self.cutoffs), "comparisons": dict(self.comparisons),
            "rows_compared": dict(self.rows_compared), "failures": dict(self.failures),
            "execution_errors": self.execution_errors,
        }


def _f(check: str, status: str, message: str, *, classification: str | None = None, evidence: dict | None = None) -> Finding:
    return Finding(check, status, message, family=FAMILY, classification=classification, evidence=evidence)


def _sort_by(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Deterministic canonical order that tolerates mixed-type event IDs."""
    if frame.empty:
        return frame.reset_index(drop=True)
    work = frame.copy()
    helper = []
    for i, key in enumerate(keys):
        name = f"__sort_{i}"
        work[name] = to_utc(work[key]) if key.endswith("_time") or key in ("target_start", "target_end") else work[key].astype(str)
        helper.append(name)
    return work.sort_values(helper, kind="mergesort").drop(columns=helper).reset_index(drop=True)


def _event_time_by_id(events: pd.DataFrame | None) -> pd.Series:
    if events is None or events.empty or not {"event_id", "event_time"}.issubset(events.columns):
        return pd.Series(pd.DatetimeIndex([], tz="UTC"), index=pd.Index([], dtype=object))
    frame = events.drop_duplicates("event_id", keep="first")
    return pd.Series(to_utc(frame["event_time"]).array, index=pd.Index(frame["event_id"].to_numpy(), dtype=object))


def events_before(events: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    x = events[to_utc(events["event_time"]) < cutoff]
    return _sort_by(x, ["event_time", "event_id"])


def _feature_known_time(features: pd.DataFrame, events: pd.DataFrame | None) -> pd.Series:
    """Event time for each feature row; unknown events fall back to feature_asof_time."""
    lookup = _event_time_by_id(events)
    known = pd.Series(lookup.reindex(pd.Index(features["event_id"].to_numpy(), dtype=object)).array, index=features.index)
    asof = to_utc(features["feature_asof_time"])
    return known.fillna(asof)


def features_before(features: pd.DataFrame | None, events: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame()
    x = features[_feature_known_time(features, events) < cutoff]
    return _sort_by(x, ["event_id"])


def features_asof_window(features: pd.DataFrame | None, events: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Feature rows with feature_asof_time < cutoff <= event_time (declared known, event not yet)."""
    if features is None or features.empty:
        return pd.DataFrame()
    asof = to_utc(features["feature_asof_time"])
    known = _feature_known_time(features, events)
    x = features[(asof < cutoff) & (known >= cutoff)]
    return _sort_by(x, ["event_id"])


def targets_before(targets: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame:
    if targets is None or targets.empty:
        return pd.DataFrame()
    x = targets[to_utc(targets["target_end"]) < cutoff]
    keys = ["event_id", TARGET_NAME_COL] if TARGET_NAME_COL in x.columns else ["event_id"]
    return _sort_by(x, keys)


def _adjacent_cutoffs(index: pd.DatetimeIndex, times: pd.Series, count: int) -> list[pd.Timestamp]:
    """First bar strictly after ``count`` evenly sampled timestamps.

    Uses Timestamp-based ``searchsorted`` (resolution-agnostic): with pandas >= 3 an index may be
    stored in microseconds while ``Timestamp.value`` is always nanoseconds, so raw integer
    comparisons would be silently wrong.
    """
    times = to_utc(times).dropna().sort_values(kind="mergesort").reset_index(drop=True)
    if times.empty or count <= 0:
        return []
    picks = times.iloc[np.unique(np.linspace(0, len(times) - 1, min(count, len(times)), dtype=int))]
    out = []
    for value in picks:
        pos = int(index.searchsorted(value, side="right"))
        if 0 < pos < len(index):
            out.append(index[pos])
    return out


def generate_research_cutoffs(
    bars: pd.DataFrame, tables: ResearchTables, *, mode: str = "strong", seed: int = 1729,
    explicit: list[pd.Timestamp] | None = None,
) -> list[pd.Timestamp]:
    if explicit is not None:
        return sorted({pd.Timestamp(x).tz_convert("UTC") for x in explicit})
    # Map research tables onto the frozen harness stage names so its cutoff logic is reused as-is.
    proxy: dict[str, pd.DataFrame] = {}
    if tables.events is not None and "event_time" in tables.events:
        proxy["signals"] = pd.DataFrame({"signal_time": to_utc(tables.events["event_time"])})
    if tables.features is not None and "feature_asof_time" in tables.features:
        proxy["features"] = pd.DataFrame({"timestamp": to_utc(tables.features["feature_asof_time"])})
    if tables.targets is not None and "target_end" in tables.targets:
        proxy["trades"] = pd.DataFrame({"exit_time": to_utc(tables.targets["target_end"])})
    chosen = set(generate_cutoffs(bars, proxy, mode=mode, seed=seed))
    per_kind = ADJACENT_PER_KIND[mode]
    for frame, column in ((tables.events, "event_time"), (tables.features, "feature_asof_time"), (tables.targets, "target_end")):
        if frame is not None and column in frame:
            chosen.update(_adjacent_cutoffs(bars.index, frame[column], per_kind))
    return sorted(chosen)


def _build_comparisons(
    full: ResearchTables, other: ResearchTables, kind: str, cutoff: pd.Timestamp, exposed: dict[str, bool],
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    comparisons = [("events", events_before(full.events, cutoff), events_before(other.events, cutoff))]
    if exposed["features"]:
        comparisons.append(("features", features_before(full.features, full.events, cutoff),
                            features_before(other.features, other.events, cutoff)))
    if exposed["features_asof"] and kind == "future_mutation":
        base = features_asof_window(full.features, full.events, cutoff)
        if not base.empty and other.features is not None and not other.features.empty:
            shared = base["event_id"].isin(set(other.features["event_id"].tolist()))
            base = base[shared].reset_index(drop=True)
            candidate = other.features[other.features["event_id"].isin(set(base["event_id"].tolist()))]
            comparisons.append(("features_asof", base, _sort_by(candidate, ["event_id"])))
    if exposed["targets"]:
        comparisons.append(("targets", targets_before(full.targets, cutoff), targets_before(other.targets, cutoff)))
    return comparisons


def audit_research_causality(
    adapter_factory: Callable[[], Any], bars: pd.DataFrame, full: ResearchTables, *,
    mode: str = "strong", seed: int = 1729, cutoffs: list[pd.Timestamp] | None = None, atol: float = 1e-9,
) -> tuple[list[Finding], ResearchCausalityCoverage]:
    """Compare full / truncated / future-mutated research tables at many cutoffs."""
    coverage = ResearchCausalityCoverage(mode, seed)
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        return [_f("research_causality", "FAIL", "bars must have a timezone-aware DatetimeIndex", classification=PROVEN)], coverage
    if full.events is None:
        return [_f("research_causality", "UNVERIFIED", "adapter exposes no events(); research causality cannot be evaluated")], coverage
    exposed = {"events": True, "features": full.features is not None, "features_asof": full.features is not None,
               "targets": full.targets is not None}
    chosen = generate_research_cutoffs(bars, full, mode=mode, seed=seed, explicit=cutoffs)
    coverage.cutoffs = tuple(c.isoformat() for c in chosen)
    findings: list[Finding] = []
    if not chosen:
        return [_f("research_causality", "UNVERIFIED", "no valid causality cutoffs (dataset too short)")], coverage

    counters = {(stage, kind): {"comparisons": 0, "rows": 0, "failures": 0}
                for stage in STAGES for kind in ("truncation", "future_mutation")}
    for cutoff in chosen:
        runs: dict[str, ResearchTables] = {}
        for kind, frame in (("truncation", bars[bars.index < cutoff]), ("future_mutation", mutate_future(bars, cutoff))):
            try:
                runs[kind] = run_pipeline(adapter_factory(), frame)
            except Exception as exc:
                coverage.execution_errors += 1
                findings.append(_f(
                    f"{kind}_execution", "UNVERIFIED",
                    f"cutoff={cutoff.isoformat()}: candidate raised {type(exc).__name__}: {exc}; comparison impossible at this cutoff",
                    classification=HIGH, evidence={"cutoff": cutoff.isoformat(), "kind": kind},
                ))
        for kind, other in runs.items():
            try:
                comparisons = _build_comparisons(full, other, kind, cutoff, exposed)
            except (KeyError, TypeError, ValueError) as exc:
                # A table whose schema/types change when only the future changes is itself a
                # causality violation; it must never crash the verifier or be skipped silently.
                findings.append(_f(
                    f"{kind}_schema", "FAIL",
                    f"cutoff={cutoff.isoformat()}: {kind} run returned tables that cannot be compared: {type(exc).__name__}: {exc}",
                    classification=PROVEN, evidence={"cutoff": cutoff.isoformat()},
                ))
                continue
            for stage, baseline, comparison in comparisons:
                same, why = _equal(baseline, comparison, atol) if not (baseline.empty and comparison.empty) else (True, "no rows")
                counter = counters[(stage, kind)]
                counter["comparisons"] += 1
                counter["rows"] += len(baseline)
                if not same:
                    counter["failures"] += 1
                    ids_full = set(map(str, baseline.get("event_id", pd.Series(dtype=object)).tolist()))
                    ids_other = set(map(str, comparison.get("event_id", pd.Series(dtype=object)).tolist()))
                    findings.append(_f(
                        f"{kind}_{stage}", "FAIL",
                        f"cutoff={cutoff.isoformat()}: {why}; rows full={len(baseline)} {kind}={len(comparison)}",
                        classification=PROVEN,
                        evidence={
                            "cutoff": cutoff.isoformat(),
                            "only_in_full": sorted(ids_full - ids_other)[:20],
                            "only_in_comparison": sorted(ids_other - ids_full)[:20],
                            "first_event_ids_full": sorted(ids_full)[:10],
                        },
                    ))
    for (stage, kind), counter in counters.items():
        key = f"{kind}_{stage}"
        coverage.comparisons[key] = counter["comparisons"]
        coverage.rows_compared[key] = counter["rows"]
        coverage.failures[key] = counter["failures"]
        if not exposed[stage]:
            findings.append(_f(key, "UNVERIFIED", f"adapter does not expose the table needed for {stage}; {stage} causality UNVERIFIED"))
            continue
        if stage == "features_asof" and kind == "truncation":
            continue  # declared-as-of rows whose event is not yet knowable do not exist in a prefix run
        if counter["failures"]:
            continue  # per-cutoff FAIL findings already emitted
        if stage == "features_asof":
            if counter["rows"] == 0:
                findings.append(_f(key, "INFO", "no feature row declared feature_asof_time strictly before a cutoff at/after its event_time; as-of invariance covered by the features comparison"))
            else:
                findings.append(_f(key, "PASS", f"{counter['rows']} declared-as-of feature rows unchanged across {counter['comparisons']} future-mutated cutoffs"))
            continue
        if counter["rows"] == 0:
            findings.append(_f(key, "UNVERIFIED", f"{stage}: zero rows were comparable at {len(chosen)} cutoffs; no evidence of invariance"))
            continue
        findings.append(_f(
            key, "PASS",
            f"{stage}: {counter['rows']} knowable rows identical across {counter['comparisons']} {kind} cutoffs",
        ))
    return findings, coverage
