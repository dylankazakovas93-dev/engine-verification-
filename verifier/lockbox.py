"""Final-holdout (lockbox) isolation for research verification.

Development mode physically removes every lockbox row from the tables handed to
``fit_predict_fold``. Conservative boundary rule (documented in RESEARCH_VERIFICATION.md):

* an event with ``event_time >= lockbox_start`` is withheld;
* an event with ``event_time < lockbox_start`` whose target window ends at/after
  ``lockbox_start`` is ALSO withheld, because its label is computed from lockbox-period prices.

Event/feature generation may run on the full bar history (the causality poison tests prove that
historical events/features are invariant to later data). There is deliberately no API here that
returns lockbox rows for modelling or computes lockbox performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .research_contracts import to_utc
from .schema import Finding
from .walkforward import LockboxViolation

FAMILY = "lockbox"


def _f(check: str, status: str, message: str, *, classification: str | None = None, evidence: dict | None = None) -> Finding:
    return Finding(check, status, message, family=FAMILY, classification=classification, evidence=evidence)


def parse_lockbox_start(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """``YYYY-MM-DD`` (interpreted as UTC midnight) or any timezone-aware timestamp."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


@dataclass
class LockboxSplit:
    lockbox_start: pd.Timestamp | None
    development_ids: tuple
    withheld_ids: tuple          # event_time >= lockbox_start
    boundary_ids: tuple          # event_time < lockbox_start <= max(target_end)
    features: pd.DataFrame | None = None   # model-accessible (development only)
    targets: pd.DataFrame | None = None    # model-accessible (development only)
    event_times: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns, UTC]"))

    def counts(self) -> dict[str, Any]:
        return {
            "lockbox_start": self.lockbox_start.isoformat() if self.lockbox_start is not None else None,
            "development_event_count": len(self.development_ids),
            "withheld_lockbox_event_count": len(self.withheld_ids),
            "withheld_boundary_event_count": len(self.boundary_ids),
        }


def split_lockbox(
    events: pd.DataFrame, features: pd.DataFrame | None, targets: pd.DataFrame | None,
    lockbox_start: pd.Timestamp | None, resolution_times: pd.Series | None = None,
) -> LockboxSplit:
    """``resolution_times`` (event_id -> declared-horizon resolution time, bar open + interval of the
    Nth window bar) widens the boundary rule: an event is also withheld when its DECLARED horizon
    resolves at/after the lockbox, even if the candidate's claimed target_end is earlier."""
    event_time = pd.Series(to_utc(events["event_time"]).array, index=pd.Index(events["event_id"].to_numpy(), dtype=object))
    if lockbox_start is None:
        dev = tuple(events["event_id"].tolist())
        return LockboxSplit(None, dev, (), (), features.copy() if features is not None else None,
                            targets.copy() if targets is not None else None, event_time)
    withheld = set(event_time.index[event_time >= lockbox_start].tolist())
    boundary: set = set()
    if targets is not None and not targets.empty:
        end = to_utc(targets["target_end"])
        crossing = targets.loc[end >= lockbox_start, "event_id"].tolist()
        boundary = {i for i in crossing if i not in withheld}
    if resolution_times is not None and not resolution_times.empty:
        declared = pd.to_datetime(resolution_times, utc=True)
        boundary |= {i for i in declared.index[(declared >= lockbox_start).to_numpy()] if i not in withheld}
    removed = withheld | boundary
    ordered_ids = events["event_id"].tolist()
    dev = tuple(i for i in ordered_ids if i not in removed)
    dev_set = set(dev)
    model_features = features[features["event_id"].isin(dev_set)].reset_index(drop=True) if features is not None else None
    model_targets = targets[targets["event_id"].isin(dev_set)].reset_index(drop=True) if targets is not None else None
    return LockboxSplit(
        lockbox_start, dev,
        tuple(i for i in ordered_ids if i in withheld),
        tuple(i for i in ordered_ids if i in boundary),
        model_features, model_targets, event_time,
    )


def assert_model_tables_exclude_lockbox(
    features: pd.DataFrame, targets: pd.DataFrame, event_times: pd.Series, lockbox_start: pd.Timestamp | None,
) -> None:
    """Hard guard executed before EVERY model call. Raises instead of warning."""
    if lockbox_start is None:
        return
    for name, frame in (("features", features), ("targets", targets)):
        ids = pd.Index(frame["event_id"].to_numpy(), dtype=object)
        times = pd.Series(event_times.reindex(ids).array)
        if times.isna().any():
            raise LockboxViolation(f"{name} passed to model contains event_ids with unknown event_time")
        if (times >= lockbox_start).any():
            raise LockboxViolation(f"{name} passed to model contains events at/after lockbox {lockbox_start.isoformat()}")
    if "target_end" in targets.columns and (to_utc(targets["target_end"]) >= lockbox_start).any():
        raise LockboxViolation(f"targets passed to model contain target windows ending at/after lockbox {lockbox_start.isoformat()}")


def audit_lockbox(split: LockboxSplit, accessed_ids: set | None) -> list[Finding]:
    counts = split.counts()
    if split.lockbox_start is None:
        return [_f("lockbox_declared", "UNVERIFIED",
                   "no --lockbox-start supplied; final-holdout isolation cannot be demonstrated", evidence=counts)]
    findings = [_f("lockbox_declared", "PASS", f"lockbox_start={counts['lockbox_start']} (UTC)", evidence=counts)]
    withheld_total = len(split.withheld_ids) + len(split.boundary_ids)
    if not split.development_ids:
        findings.append(_f("lockbox_counts", "UNVERIFIED", f"no development events remain before the lockbox: {counts}", evidence=counts))
    elif not split.withheld_ids:
        findings.append(_f("lockbox_counts", "UNVERIFIED",
                           f"lockbox withholds zero events at/after the boundary; the holdout is empty: {counts}", evidence=counts))
    else:
        findings.append(_f(
            "lockbox_counts", "INFO",
            f"development events={counts['development_event_count']}; withheld lockbox events={counts['withheld_lockbox_event_count']}; "
            f"withheld boundary-crossing events={counts['withheld_boundary_event_count']}",
            evidence=counts,
        ))
    forbidden = set(split.withheld_ids) | set(split.boundary_ids)
    leaked_tables = []
    for name, frame in (("features", split.features), ("targets", split.targets)):
        if frame is not None:
            bad = sorted(map(str, set(frame["event_id"].tolist()) & forbidden))
            if bad:
                leaked_tables.append({"table": name, "ids": bad[:10], "count": len(bad)})
    try:
        if split.features is not None and split.targets is not None:
            assert_model_tables_exclude_lockbox(split.features, split.targets, split.event_times, split.lockbox_start)
        guard_error = None
    except LockboxViolation as exc:
        guard_error = str(exc)
    ok = not leaked_tables and guard_error is None
    findings.append(_f(
        "lockbox_model_tables", "PASS" if ok else "FAIL",
        f"model-accessible features/targets contain none of the {withheld_total} withheld events" if ok
        else f"model-accessible tables contain lockbox rows: {leaked_tables or guard_error}",
        classification=None if ok else "PROVEN FAILURE",
    ))
    if accessed_ids is None:
        findings.append(_f("lockbox_model_access_log", "INFO", "no fit_predict_fold calls were made; access log empty"))
    else:
        touched = sorted(map(str, set(accessed_ids) & forbidden))
        findings.append(_f(
            "lockbox_model_access_log", "FAIL" if touched else "PASS",
            f"fit_predict_fold received {len(touched)} withheld lockbox event_ids: {touched[:10]}" if touched
            else f"{len(accessed_ids)} distinct event_ids were passed to fit_predict_fold; none is withheld",
            classification="PROVEN FAILURE" if touched else None,
        ))
    return findings
