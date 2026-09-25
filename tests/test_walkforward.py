from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import research_toys as T
from verifier.walkforward import (
    Fold, LockboxViolation, audit_folds, audit_oof_predictions, build_purged_walkforward_folds,
)

LOCKBOX = pd.Timestamp(T.LOCKBOX, tz="UTC")


@pytest.fixture(scope="module")
def observations():
    bars = T.make_bars()
    events = T.CleanToy.events(bars)
    targets = T.CleanToy.targets(bars, events)
    obs = targets.merge(events[["event_id", "event_time"]], on="event_id")
    obs = obs[(obs["event_time"] < LOCKBOX) & (obs["target_end"] < LOCKBOX)]
    return obs[["event_id", "event_time", "target_start", "target_end"]].reset_index(drop=True)


@pytest.fixture(scope="module")
def folds(observations):
    built, _ = build_purged_walkforward_folds(observations, lockbox_start=LOCKBOX)
    return built


def _failing(findings):
    return {f.check for f in findings if f.status == "FAIL"}


def test_builder_produces_annual_expanding_purged_folds(folds, observations):
    assert [f.validation_start.year for f in folds] == [2016, 2017, 2018, 2019]
    assert all(f.validation_end <= LOCKBOX for f in folds)
    by_id = observations.set_index("event_id")
    for previous, current in zip(folds, folds[1:]):
        assert set(previous.train_ids) <= set(current.train_ids)  # expanding
    for fold in folds:
        train = by_id.loc[list(fold.train_ids)]
        assert (train["target_end"] < fold.validation_start).all()
        purged = by_id.loc[list(fold.purged_ids)]
        # Purged rows precede validation by event_time but their labels overlap it.
        assert len(purged) > 0
        assert (purged["event_time"] < fold.validation_start).all()
        assert (purged["target_end"] >= fold.validation_start).all()


def test_verifier_proves_clean_folds(folds, observations):
    findings = audit_folds(folds, observations, lockbox_start=LOCKBOX)
    assert findings and {f.status for f in findings} == {"PASS"}


def test_unpurged_fold_fails(folds, observations):
    bad = replace(folds[1], train_ids=folds[1].train_ids + folds[1].purged_ids)
    findings = audit_folds([folds[0], bad, *folds[2:]], observations, lockbox_start=LOCKBOX)
    failing = _failing(findings)
    assert "walkforward_train_target_overlaps_validation" in failing
    assert "walkforward_train_not_before_validation" not in failing  # purged rows precede validation by event_time


def test_random_nonchronological_split_fails(folds, observations):
    rng = np.random.default_rng(0)
    ids = observations["event_id"].to_numpy()
    shuffled = rng.permutation(ids)
    random_fold = Fold(0, folds[0].validation_start, folds[0].validation_end,
                       tuple(shuffled[:200]), tuple(shuffled[200:260]), ())
    failing = _failing(audit_folds([random_fold], observations, lockbox_start=LOCKBOX))
    assert {"walkforward_train_not_before_validation", "walkforward_validation_outside_block",
            "walkforward_validation_incomplete"} <= failing


def test_training_rows_after_validation_start_fail(folds, observations):
    later = folds[2].validation_ids[:5]
    bad = replace(folds[1], train_ids=folds[1].train_ids + later)
    failing = _failing(audit_folds([bad], observations, lockbox_start=LOCKBOX))
    assert "walkforward_train_not_before_validation" in failing


def test_train_validation_overlap_and_subsampled_validation_fail(folds, observations):
    overlap = replace(folds[1], train_ids=folds[1].train_ids + folds[1].validation_ids[:3])
    assert "walkforward_train_validation_overlap" in _failing(audit_folds([overlap], observations))
    subsample = replace(folds[1], validation_ids=folds[1].validation_ids[::2])
    assert "walkforward_validation_incomplete" in _failing(audit_folds([subsample], observations))


def test_overlapping_or_reordered_blocks_fail(folds, observations):
    failing = _failing(audit_folds([folds[1], folds[0]], observations))
    assert "walkforward_blocks_chronological" in failing
    duplicate = _failing(audit_folds([folds[0], folds[0]], observations))
    assert "walkforward_validation_unique_across_folds" in duplicate


def test_purge_rule_uses_target_end_not_event_time():
    obs = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(30)] + ["cross", "val"],
        "event_time": list(pd.date_range("2016-01-04", periods=30, freq="7D", tz="UTC"))
        + [pd.Timestamp("2016-12-30", tz="UTC"), pd.Timestamp("2017-02-01", tz="UTC")],
    })
    obs["target_start"] = obs["event_time"]
    obs["target_end"] = obs["event_time"] + pd.Timedelta(days=1)
    obs.loc[obs["event_id"] == "cross", "target_end"] = pd.Timestamp("2017-01-03", tz="UTC")
    folds, _ = build_purged_walkforward_folds(obs, min_train_events=5)
    assert len(folds) == 1
    assert "cross" not in folds[0].train_ids and "cross" in folds[0].purged_ids
    # A builder that purged by event_time only would have trained on it; the auditor rejects that.
    naive = replace(folds[0], train_ids=folds[0].train_ids + ("cross",))
    assert "walkforward_train_target_overlaps_validation" in _failing(audit_folds([naive], obs))


def test_observations_touching_lockbox_raise(observations):
    leaked = observations.copy()
    leaked.loc[len(leaked) - 1, "event_time"] = LOCKBOX + pd.Timedelta(days=1)
    leaked.loc[len(leaked) - 1, "target_end"] = LOCKBOX + pd.Timedelta(days=5)
    with pytest.raises(LockboxViolation):
        build_purged_walkforward_folds(leaked, lockbox_start=LOCKBOX)


def test_fold_touching_lockbox_fails_audit(folds, observations):
    bad = replace(folds[-1], validation_end=LOCKBOX + pd.Timedelta(days=30))
    assert "walkforward_lockbox" in _failing(audit_folds([*folds[:-1], bad], observations, lockbox_start=LOCKBOX))


def test_no_folds_is_unverified(observations):
    single_year = observations[observations["event_time"] < pd.Timestamp("2016-01-01", tz="UTC")]
    folds, skipped = build_purged_walkforward_folds(single_year)
    assert folds == []
    assert {f.status for f in audit_folds(folds, single_year)} == {"UNVERIFIED"}


def test_oof_alignment(folds):
    good = {f.fold: pd.DataFrame({"event_id": list(f.validation_ids), "prediction": 0.5}) for f in folds}
    assert {x.status for x in audit_oof_predictions(folds, good)} == {"PASS"}
    missing = dict(good)
    missing[0] = good[0].iloc[:-1]
    assert {x.status for x in audit_oof_predictions(folds, missing)} == {"FAIL"}
    duplicated = dict(good)
    duplicated[1] = pd.concat([good[1], good[1].iloc[[0]]])
    assert {x.status for x in audit_oof_predictions(folds, duplicated)} == {"FAIL"}
    nonfinite = dict(good)
    nonfinite[2] = good[2].assign(prediction=np.nan)
    assert {x.status for x in audit_oof_predictions(folds, nonfinite)} == {"FAIL"}
    assert {x.status for x in audit_oof_predictions(folds, {})} == {"UNVERIFIED"}
