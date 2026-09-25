"""Verifier-owned purged, expanding, chronological walk-forward folds.

The candidate never builds its own folds. For a validation block ``[validation_start,
validation_end)`` a development observation is a legal training row only if

    event_time < validation_start  AND  target_end < validation_start

(purging). ``event_time < validation_start`` alone is NOT sufficient: a label whose window
overlaps validation carries validation-period price information.

Blocks are UTC calendar years (v1): train through year Y-1 -> validate year Y. No shuffling,
no random assignment, no K-fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .research_contracts import to_utc
from .schema import Finding

FAMILY = "walkforward"
PROVEN = "PROVEN FAILURE"
OBSERVATION_COLUMNS = ("event_id", "event_time", "target_start", "target_end")


class LockboxViolation(RuntimeError):
    """Raised when verifier code is asked to operate on rows at/after the lockbox boundary."""


@dataclass(frozen=True)
class Fold:
    fold: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp  # exclusive
    train_ids: tuple
    validation_ids: tuple
    purged_ids: tuple  # event_time < validation_start but target_end >= validation_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "train_count": len(self.train_ids),
            "validation_count": len(self.validation_ids),
            "purged_count": len(self.purged_ids),
        }


def _f(check: str, status: str, message: str, *, classification: str | None = None, evidence: dict | None = None) -> Finding:
    return Finding(check, status, message, family=FAMILY, classification=classification, evidence=evidence)


def normalize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in OBSERVATION_COLUMNS if c not in observations.columns]
    if missing:
        raise ValueError(f"walk-forward observations missing {missing}")
    obs = observations[list(OBSERVATION_COLUMNS)].copy()
    for column in ("event_time", "target_start", "target_end"):
        obs[column] = to_utc(obs[column])
    if obs[["event_time", "target_end"]].isna().any().any():
        raise ValueError("walk-forward observations contain null event_time/target_end")
    if obs["event_id"].duplicated().any():
        raise ValueError("walk-forward observations contain duplicate event_id")
    obs["__id"] = obs["event_id"].astype(str)
    return obs.sort_values(["event_time", "__id"], kind="mergesort").drop(columns="__id").reset_index(drop=True)


def _ordered_ids(obs: pd.DataFrame, mask: pd.Series | np.ndarray) -> tuple:
    return tuple(obs.loc[mask, "event_id"].tolist())


def build_purged_walkforward_folds(
    observations: pd.DataFrame, *, lockbox_start: pd.Timestamp | None = None,
    min_train_events: int = 20, min_validation_events: int = 1,
) -> tuple[list[Fold], list[dict[str, Any]]]:
    """Expanding-window annual folds with target-window purging.

    Returns (folds, skipped_blocks). ``observations`` must already be lockbox-filtered; any row at
    or after ``lockbox_start`` raises :class:`LockboxViolation` rather than being silently used.
    """
    obs = normalize_observations(observations)
    if obs.empty:
        return [], []
    if lockbox_start is not None:
        leaked = obs[(obs["event_time"] >= lockbox_start) | (obs["target_end"] >= lockbox_start)]
        if not leaked.empty:
            raise LockboxViolation(f"{len(leaked)} observations touch the lockbox (event_time or target_end >= {lockbox_start.isoformat()})")
    first_year = int(obs["event_time"].iloc[0].year)
    last_year = int(obs["event_time"].iloc[-1].year)
    folds: list[Fold] = []
    skipped: list[dict[str, Any]] = []
    for year in range(first_year + 1, last_year + 1):
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        if lockbox_start is not None:
            if start >= lockbox_start:
                break
            end = min(end, lockbox_start)
        before = obs["event_time"] < start
        train_mask = before & (obs["target_end"] < start)
        purged_mask = before & ~train_mask
        validation_mask = (obs["event_time"] >= start) & (obs["event_time"] < end)
        n_train, n_validation = int(train_mask.sum()), int(validation_mask.sum())
        if n_train < min_train_events or n_validation < min_validation_events:
            skipped.append({"validation_start": start.isoformat(), "validation_end": end.isoformat(),
                            "train_count": n_train, "validation_count": n_validation,
                            "reason": f"requires >= {min_train_events} train and >= {min_validation_events} validation events"})
            continue
        folds.append(Fold(
            fold=len(folds), validation_start=start, validation_end=end,
            train_ids=_ordered_ids(obs, train_mask), validation_ids=_ordered_ids(obs, validation_mask),
            purged_ids=_ordered_ids(obs, purged_mask),
        ))
    return folds, skipped


def audit_folds(
    folds: list[Fold], observations: pd.DataFrame, *, lockbox_start: pd.Timestamp | None = None,
) -> list[Finding]:
    """Independently prove the fold invariants. Accepts arbitrary (possibly malicious) folds."""
    obs = normalize_observations(observations)
    if not folds:
        return [_f("walkforward_folds", "UNVERIFIED", "no walk-forward folds could be built (insufficient history across calendar years)")]
    by_id = obs.set_index(obs["event_id"].astype(object))
    known = set(by_id.index.tolist())
    findings: list[Finding] = []
    violations: dict[str, list[dict[str, Any]]] = {k: [] for k in (
        "unknown_ids", "duplicate_ids", "train_validation_overlap", "train_not_before_validation",
        "train_target_overlaps_validation", "validation_outside_block", "validation_incomplete",
        "validation_not_chronological", "train_not_expanding_window", "lockbox",
    )}
    for fold in folds:
        tag = {"fold": fold.fold, "validation_start": fold.validation_start.isoformat()}
        train, validation = list(fold.train_ids), list(fold.validation_ids)
        unknown = [str(i) for i in train + validation if i not in known]
        if unknown:
            violations["unknown_ids"].append({**tag, "ids": unknown[:10]})
            continue
        if len(set(train)) != len(train) or len(set(validation)) != len(validation):
            violations["duplicate_ids"].append(tag)
        overlap = set(train) & set(validation)
        if overlap:
            violations["train_validation_overlap"].append({**tag, "ids": sorted(map(str, overlap))[:10]})
        t = by_id.loc[train] if train else by_id.iloc[:0]
        v = by_id.loc[validation] if validation else by_id.iloc[:0]
        late = t[t["event_time"] >= fold.validation_start]
        if not late.empty:
            violations["train_not_before_validation"].append({**tag, "ids": list(map(str, late["event_id"]))[:10],
                                                              "latest_train_event": late["event_time"].max().isoformat()})
        unpurged = t[t["target_end"] >= fold.validation_start]
        if not unpurged.empty:
            violations["train_target_overlaps_validation"].append({
                **tag, "ids": list(map(str, unpurged["event_id"]))[:10],
                "max_train_target_end": unpurged["target_end"].max().isoformat(),
            })
        outside = v[(v["event_time"] < fold.validation_start) | (v["event_time"] >= fold.validation_end)]
        if not outside.empty:
            violations["validation_outside_block"].append({**tag, "ids": list(map(str, outside["event_id"]))[:10]})
        in_block = obs[(obs["event_time"] >= fold.validation_start) & (obs["event_time"] < fold.validation_end)]
        if set(in_block["event_id"].tolist()) != set(validation):
            missing = sorted(map(str, set(in_block["event_id"].tolist()) - set(validation)))
            violations["validation_incomplete"].append({**tag, "missing": missing[:10], "missing_count": len(missing)})
        if not v["event_time"].is_monotonic_increasing:
            violations["validation_not_chronological"].append(tag)
        legal = obs[(obs["event_time"] < fold.validation_start) & (obs["target_end"] < fold.validation_start)]
        if set(legal["event_id"].tolist()) != set(train):
            violations["train_not_expanding_window"].append({
                **tag, "expected_count": len(legal), "supplied_count": len(train),
            })
        if lockbox_start is not None:
            touched = pd.concat([t, v])
            bad = touched[(touched["event_time"] >= lockbox_start) | (touched["target_end"] >= lockbox_start)]
            if not bad.empty or fold.validation_end > lockbox_start:
                violations["lockbox"].append({**tag, "ids": list(map(str, bad["event_id"]))[:10]})
    messages = {
        "unknown_ids": "fold IDs reference observations absent from the model table",
        "duplicate_ids": "fold ID lists contain duplicates",
        "train_validation_overlap": "train and validation IDs overlap",
        "train_not_before_validation": "training events at/after validation_start (nonchronological / shuffled split)",
        "train_target_overlaps_validation": "training target windows end at/after validation_start (unpurged)",
        "validation_outside_block": "validation events outside their declared block",
        "validation_incomplete": "validation IDs are not exactly the chronological block (random/subsampled split)",
        "validation_not_chronological": "validation IDs are not in chronological order",
        "train_not_expanding_window": "training IDs are not exactly the purged expanding window",
        "lockbox": "folds touch rows or time at/after the lockbox boundary",
    }
    for key, items in violations.items():
        findings.append(_f(
            f"walkforward_{key}", "FAIL" if items else "PASS",
            f"{messages[key]}: {items[:3]}" if items else f"no violation across {len(folds)} folds: not ({messages[key]})",
            classification=PROVEN if items else None, evidence={"violations": items[:20]} if items else None,
        ))
    starts = [f.validation_start for f in folds]
    ends = [f.validation_end for f in folds]
    ordered = all(s < e for s, e in zip(starts, ends)) and all(ends[i] <= starts[i + 1] for i in range(len(folds) - 1))
    findings.append(_f(
        "walkforward_blocks_chronological", "PASS" if ordered else "FAIL",
        "validation blocks are strictly increasing and non-overlapping" if ordered
        else f"validation blocks overlap or are out of order: {[(s.isoformat(), e.isoformat()) for s, e in zip(starts, ends)]}",
        classification=None if ordered else PROVEN,
    ))
    seen: dict[Any, int] = {}
    repeated = []
    for fold in folds:
        for i in fold.validation_ids:
            if i in seen:
                repeated.append(str(i))
            seen[i] = fold.fold
    findings.append(_f(
        "walkforward_validation_unique_across_folds", "FAIL" if repeated else "PASS",
        f"{len(repeated)} observations validated in more than one fold; e.g. {repeated[:10]}" if repeated
        else f"each of {len(seen)} validation observations belongs to exactly one fold",
        classification=PROVEN if repeated else None,
    ))
    return findings


def audit_oof_predictions(folds: list[Fold], predictions: dict[int, pd.DataFrame | None]) -> list[Finding]:
    """Every OOF prediction corresponds to exactly one validation observation of its fold."""
    if not folds:
        return [_f("walkforward_oof_alignment", "UNVERIFIED", "no folds; OOF alignment not evaluated")]
    problems: list[dict[str, Any]] = []
    missing_folds = []
    total = 0
    for fold in folds:
        frame = predictions.get(fold.fold)
        if frame is None:
            missing_folds.append(fold.fold)
            continue
        issue = prediction_frame_problem(frame, fold.validation_ids)
        if issue:
            problems.append({"fold": fold.fold, "problem": issue})
        else:
            total += len(frame)
    if problems:
        return [_f("walkforward_oof_alignment", "FAIL", f"OOF predictions do not map one-to-one onto validation IDs: {problems[:3]}",
                   classification=PROVEN, evidence={"problems": problems[:20]})]
    if missing_folds:
        return [_f("walkforward_oof_alignment", "UNVERIFIED", f"no predictions available for folds {missing_folds}")]
    return [_f("walkforward_oof_alignment", "PASS", f"{total} OOF predictions across {len(folds)} folds, exactly one per validation observation")]


def prediction_frame_problem(frame: Any, validation_ids: tuple | list) -> str | None:
    """Return a description of why ``frame`` is not a valid prediction table, else ``None``."""
    if not isinstance(frame, pd.DataFrame):
        return f"fit_predict_fold returned {type(frame).__name__}, not a DataFrame"
    missing = [c for c in ("event_id", "prediction") if c not in frame.columns]
    if missing:
        return f"prediction frame missing columns {missing}"
    ids = frame["event_id"].tolist()
    if len(ids) != len(set(ids)):
        return f"{len(ids) - len(set(ids))} duplicate prediction event_ids"
    expected = set(validation_ids)
    extra = sorted(map(str, set(ids) - expected))
    absent = sorted(map(str, expected - set(ids)))
    if extra or absent:
        return f"prediction IDs != validation IDs (extra={extra[:5]} [{len(extra)}], missing={absent[:5]} [{len(absent)}])"
    values = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        return f"{int((~np.isfinite(values)).sum())} nonfinite/non-numeric predictions"
    return None
