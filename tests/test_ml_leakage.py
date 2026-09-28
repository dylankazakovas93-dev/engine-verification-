from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import research_toys as T
from verifier.lockbox import assert_model_tables_exclude_lockbox, parse_lockbox_start, split_lockbox
from verifier.ml_leakage import FORBIDDEN_TESTS, poison_features, poison_targets, scan_ml_source
from verifier.report import FAILED, INCOMPLETE, VERIFIED, evaluate_verdict
from verifier.research_orchestrator import RESEARCH_FAMILIES, run_research_audit
from verifier.walkforward import LockboxViolation

ML = "ml_leakage"


def _ml(name, mode="fast"):
    findings, coverage, artifacts = T.audit(name, mode=mode)
    return findings, coverage, artifacts


# --------------------------------------------------------------------------------------
# Clean candidate
# --------------------------------------------------------------------------------------
def test_clean_candidate_passes_every_mandatory_research_gate_in_strong_mode():
    findings, coverage, artifacts = T.audit("CleanToy", mode="strong")
    assert evaluate_verdict(findings, set(RESEARCH_FAMILIES)) == VERIFIED
    ml = T.statuses(findings, ML)
    for test in FORBIDDEN_TESTS:
        assert ml[test] == {"PASS"}, (test, ml[test])
    assert ml["control_train_label_sensitivity"] == {"PASS"}
    assert ml["control_train_feature_sensitivity"] == {"PASS"}
    assert ml["split_compliance"] == {"PASS"}
    assert coverage["ml_leakage"]["folds_audited"] == [0, 1, 2, 3]
    assert artifacts["lockbox"]["withheld_lockbox_event_count"] > 0


# --------------------------------------------------------------------------------------
# Deliberate cheats: each must fail for its intended reason
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name, intended", [
    ("ValidationLabelLeak", {"validation_label_poisoning", "split_compliance"}),
    ("FullSampleScaling", {"future_feature_poisoning", "intra_validation_feature_poisoning", "split_compliance"}),
    ("FullSampleLabels", {"validation_label_poisoning", "future_label_poisoning", "split_compliance"}),
    ("SelfSelectedUnpurged", {"purged_label_poisoning", "split_compliance"}),
    ("RandomSplitModel", {"future_label_poisoning", "future_feature_poisoning", "split_compliance"}),
])
def test_cheating_model_fails_intended_poisoning_check(name, intended):
    findings, _, _ = _ml(name)
    failing = T.failing_checks(findings, ML)
    assert intended <= failing, failing
    assert evaluate_verdict(findings, set(RESEARCH_FAMILIES)) == FAILED
    # The cheat is in the model only: data-generation families stay clean.
    for family in ("research_contract", "research_causality", "walkforward"):
        assert not T.failing_checks(findings, family), family


def test_validation_label_leak_is_not_misattributed_to_future_rows():
    failing = T.failing_checks(_ml("ValidationLabelLeak")[0], ML)
    assert "future_label_poisoning" not in failing and "future_feature_poisoning" not in failing


def test_self_selected_unpurged_training_does_not_touch_validation_or_future():
    failing = T.failing_checks(_ml("SelfSelectedUnpurged")[0], ML)
    assert "validation_label_poisoning" not in failing and "future_label_poisoning" not in failing


def test_fail_evidence_names_fold_and_changed_event_ids():
    findings, _, _ = _ml("ValidationLabelLeak")
    evidence = [f.evidence for f in findings if f.check == "validation_label_poisoning" and f.status == "FAIL"]
    assert evidence and evidence[0]["changed"] > 0 and evidence[0]["changed_ids"]
    assert "fold" in evidence[0]


def test_nondeterministic_model_fails():
    findings, _, _ = _ml("NondeterministicModel")
    assert any(c.endswith("_determinism") for c in T.failing_checks(findings, ML))


def test_wrong_prediction_ids_fail_oof_alignment():
    findings, _, _ = _ml("WrongPredictionIds")
    assert "walkforward_oof_alignment" in T.failing_checks(findings, "walkforward")


def test_label_blind_model_is_low_power_not_pass():
    findings, _, _ = _ml("ConstantModel")
    ml = T.statuses(findings, ML)
    assert ml["control_train_label_sensitivity"] == {"WARN"}
    assert evaluate_verdict(findings, set(RESEARCH_FAMILIES)) == INCOMPLETE


def test_missing_fit_predict_fold_is_unverified_not_pass():
    findings, _, _ = _ml("NoModel")
    assert T.statuses(findings, ML)["ml_leakage"] == {"UNVERIFIED"}
    assert evaluate_verdict(findings, set(RESEARCH_FAMILIES)) == INCOMPLETE


def test_output_invariant_full_sample_ols_scaling_is_a_documented_dynamic_blind_spot():
    """Full-sample scaling + OLS-with-intercept cannot change predictions (affine invariance).
    Dynamic poisoning therefore passes it; the static ML scan flags it for manual review."""
    findings, _, _ = _ml("FullSampleScalingOLS")
    assert not T.failing_checks(findings, ML)


# --------------------------------------------------------------------------------------
# Lockbox
# --------------------------------------------------------------------------------------
def test_model_never_receives_lockbox_rows():
    T.RecordingModel.seen = set()
    findings, _, artifacts = run_research_audit(
        lambda: T.RecordingModel, T.make_bars(), target_name=T.TARGET, lockbox_start=T.LOCKBOX, mode="fast",
        bar_interval=T.BAR,
    )
    bars = T.make_bars()
    events = T.CleanToy.events(bars)
    times = events.set_index("event_id")["event_time"]
    assert T.RecordingModel.seen
    assert times.loc[list(T.RecordingModel.seen)].max() < parse_lockbox_start(T.LOCKBOX)
    lockbox = T.statuses(findings, "lockbox")
    assert lockbox["lockbox_model_access_log"] == {"PASS"}
    assert lockbox["lockbox_model_tables"] == {"PASS"}
    counts = artifacts["lockbox"]
    assert counts["development_event_count"] + counts["withheld_lockbox_event_count"] + counts["withheld_boundary_event_count"] == len(events)


def test_boundary_crossing_labels_are_withheld():
    bars = T.make_bars()
    events = T.CleanToy.events(bars)
    split = split_lockbox(events, T.CleanToy.features(bars, events), T.CleanToy.targets(bars, events), parse_lockbox_start(T.LOCKBOX))
    assert split.boundary_ids
    assert not set(split.boundary_ids) & set(split.targets["event_id"])
    assert (split.targets["target_end"] < parse_lockbox_start(T.LOCKBOX)).all()


def test_lockbox_boundary_uses_declared_resolution_not_understated_claim():
    from verifier.research_contracts import TargetHorizon, declared_resolution_times
    bars = T.make_bars()
    events = T.TargetEndAtFinalBarOpen.events(bars)
    targets = T.TargetEndAtFinalBarOpen.targets(bars, events)  # claims one bar early
    lockbox = parse_lockbox_start(T.LOCKBOX)
    claimed_only = split_lockbox(events, None, targets, lockbox)
    declared = declared_resolution_times(events, bars.index, T.BAR, TargetHorizon(bars=T.HORIZON)).dropna()
    widened = split_lockbox(events, None, targets, lockbox, resolution_times=declared)
    assert set(claimed_only.boundary_ids) <= set(widened.boundary_ids)
    assert (declared.reindex(pd.Index(widened.development_ids, dtype=object)).dropna() < lockbox).all()


def test_purge_uses_declared_resolution_not_understated_claim():
    clean = T.audit("CleanToy")[1]["walkforward"]["folds"]
    understated = T.audit("TargetEndAtFinalBarOpen")[1]["walkforward"]["folds"]
    assert [f["purged_count"] for f in understated] == [f["purged_count"] for f in clean]
    assert [f["train_count"] for f in understated] == [f["train_count"] for f in clean]


def test_guard_refuses_lockbox_rows():
    bars = T.make_bars()
    events = T.CleanToy.events(bars)
    features, targets = T.CleanToy.features(bars, events), T.CleanToy.targets(bars, events)
    times = pd.Series(events["event_time"].array, index=pd.Index(events["event_id"], dtype=object))
    with pytest.raises(LockboxViolation):
        assert_model_tables_exclude_lockbox(features, targets, times, parse_lockbox_start(T.LOCKBOX))


def test_no_lockbox_is_unverified():
    findings, _, _ = T.audit("CleanToy", lockbox=None)
    assert T.statuses(findings, "lockbox")["lockbox_declared"] == {"UNVERIFIED"}
    assert evaluate_verdict(findings, set(RESEARCH_FAMILIES)) == INCOMPLETE


def test_lockbox_after_all_data_is_unverified():
    findings, _, _ = T.audit("CleanToy", lockbox="2030-01-01")
    assert T.statuses(findings, "lockbox")["lockbox_counts"] == {"UNVERIFIED"}


# --------------------------------------------------------------------------------------
# Poisoning primitives
# --------------------------------------------------------------------------------------
def test_target_poisoning_changes_only_selected_rows_and_keeps_window_structure():
    bars = T.make_bars()
    events = T.CleanToy.events(bars)
    targets = T.CleanToy.targets(bars, events)
    ids = set(targets["event_id"].iloc[10:20])
    ceiling = pd.Timestamp("2021-06-01", tz="UTC")
    poisoned = poison_targets(targets, ids, seed=3, time_ceiling=ceiling)
    mask = targets["event_id"].isin(ids)
    assert (poisoned.loc[~mask].to_numpy() == targets.loc[~mask].to_numpy()).all()
    assert not np.isclose(poisoned.loc[mask, T.TARGET], targets.loc[mask, T.TARGET]).any()
    assert (poisoned["target_start"] >= targets["target_start"]).all()
    assert (poisoned["target_end"] >= poisoned["target_start"]).all()
    assert (poisoned.loc[mask, "target_end"] < ceiling).all()


def test_feature_poisoning_changes_only_selected_rows_and_never_keys():
    bars = T.make_bars()
    events = T.CleanToy.events(bars)
    features = T.CleanToy.features(bars, events)
    ids = set(features["event_id"].iloc[50:60])
    poisoned, untouched = poison_features(features, ids, seed=5)
    mask = features["event_id"].isin(ids)
    assert untouched == []
    assert poisoned[["event_id", "feature_asof_time"]].equals(features[["event_id", "feature_asof_time"]])
    assert poisoned.loc[~mask].equals(features.loc[~mask])
    for column in T.FEATURE_COLUMNS:
        assert not np.isclose(poisoned.loc[mask, column], features.loc[mask, column]).any()


def test_discrete_labels_are_remapped_to_a_different_class():
    frame = pd.DataFrame({
        "event_id": list("abcdef"),
        "target_start": pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC"),
        "target_end": pd.date_range("2020-01-02", periods=6, freq="D", tz="UTC"),
        "hit": [0, 1, 0, 1, 1, 0],
    })
    poisoned = poison_targets(frame, {"a", "b", "c"}, seed=1)
    assert poisoned["hit"].tolist()[:3] == [1, 0, 1]
    assert poisoned["hit"].tolist()[3:] == [1, 1, 0]


# --------------------------------------------------------------------------------------
# ML-specific heuristic static scan
# --------------------------------------------------------------------------------------
def _scan(tmp_path, source):
    path = tmp_path / "candidate.py"
    path.write_text(source, encoding="utf-8")
    return {f.check: f for f in scan_ml_source(path)}


def test_static_ml_scan_flags_shuffled_and_nonchronological_splits(tmp_path):
    found = _scan(tmp_path, """
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, GridSearchCV, TimeSeriesSplit
a = train_test_split(X, y)
b = KFold(5, shuffle=True)
c = ShuffleSplit(3)
d = GridSearchCV(model, grid)
e = TimeSeriesSplit(5)
""")
    for check in ("static_ml_shuffled_split", "static_ml_shuffled_kfold", "static_ml_default_cv"):
        assert found[check].status == "WARN" and found[check].family == "ml_static_scan"
    assert found["static_ml_timeseriessplit_no_gap"].status == "INFO"
    assert not [f for f in found.values() if f.status == "FAIL"]


def test_static_ml_scan_accepts_chronological_split_forms(tmp_path):
    found = _scan(tmp_path, """
a = train_test_split(X, y, shuffle=False)
b = TimeSeriesSplit(5, gap=10)
c = GridSearchCV(model, grid, cv=TimeSeriesSplit(5, gap=10))
""")
    assert set(found) == {"static_ml_scan"} and found["static_ml_scan"].status == "PASS"


def test_static_ml_scan_flags_full_table_fit_inside_fit_predict_fold(tmp_path):
    found = _scan(tmp_path, """
def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
    X = features.drop(columns=["feature_asof_time"]).set_index("event_id")
    scaled = scaler.fit_transform(X)
    mu = features[FEATURE_COLUMNS].mean()
    return model.fit(X.loc[train_ids], y).predict(X.loc[validation_ids])
""")
    assert found["static_ml_full_table_fit"].status == "WARN"
    assert found["static_ml_full_table_statistic"].status == "WARN"


def test_static_ml_scan_does_not_flag_ordinary_training_fit(tmp_path):
    found = _scan(tmp_path, """
def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
    X = features.set_index("event_id")[FEATURE_COLUMNS]
    train = X.loc[train_ids]
    mu = train.mean()
    model.fit(train, targets.set_index("event_id").loc[train_ids, target_name])
    return model.predict(X.loc[validation_ids])
""")
    assert set(found) == {"static_ml_scan"}


def test_lockbox_before_all_events_leaves_no_development_evidence():
    findings, _, artifacts = T.audit("CleanToy", lockbox="2014-01-01")
    assert artifacts["lockbox"]["development_event_count"] == 0
    for family in ("walkforward", "ml_leakage", "lockbox"):
        assert not T.failing_checks(findings, family)
    assert evaluate_verdict(findings, set(RESEARCH_FAMILIES)) == INCOMPLETE
