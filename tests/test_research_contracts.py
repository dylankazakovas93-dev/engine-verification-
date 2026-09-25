from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import research_toys as T
from verifier.research_contracts import (
    audit_events, audit_features, audit_firewall, audit_research_contract, audit_targets,
    reserved_feature_reason, select_target, tokenize_column_name,
)


def _status(findings, check):
    return {f.status for f in findings if f.check == check}


@pytest.fixture(scope="module")
def tables():
    bars = T.make_bars()
    events = T.CleanToy.events(bars)
    return bars, events, T.CleanToy.features(bars, events), T.CleanToy.targets(bars, events)


# --------------------------------------------------------------------------------------
# Reserved-name firewall logic
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "target", "Target", "label", "y_label", "labels", "future_return", "forward_return_60m", "fwdReturn",
    "fwd_ret_5", "mfe", "mfe_60", "MAE", "mae_20", "time_to_mfe", "time_to_mae", "barrier_result",
    "barrierHit", "forward_vol", "forward_volatility_30", "future_close", "FutureHigh", "next_return",
    "targets_encoded", "mfe60", "some_future_thing",
])
def test_reserved_feature_names_are_rejected(name):
    assert reserved_feature_reason(name) is not None, tokenize_column_name(name)


@pytest.mark.parametrize("name", [
    "ret_5", "vol_20", "dist_ma_20", "name", "maestro", "futures_basis", "targeted_vol", "relabelled_regime",
    "hour_of_day", "rsi_14", "atr_ratio", "close", "mean_reversion_score", "prior_outcome", "gamma_exposure",
    "email_count", "next_session_gap_open_is_unknown_here_but_not_reserved",
])
def test_innocent_feature_names_are_not_rejected(name):
    assert reserved_feature_reason(name) is None, tokenize_column_name(name)


def test_tokenizer_splits_camel_case_and_digits():
    assert tokenize_column_name("fwdReturn60m") == ("fwd", "return", "60", "m")
    assert tokenize_column_name("MAE_20") == ("mae", "20")
    assert tokenize_column_name("futures_basis") == ("futures", "basis")


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------
def test_clean_events_pass(tables):
    bars, events, _, _ = tables
    findings, ok = audit_events(events, bars.index)
    assert ok and not [f for f in findings if f.status == "FAIL"]


def test_duplicate_event_ids_and_rows_fail(tables):
    bars, events, _, _ = tables
    broken = pd.concat([events, events.iloc[[5]]], ignore_index=True)
    findings, ok = audit_events(broken, bars.index)
    assert not ok
    assert _status(findings, "events_id_unique") == {"FAIL"}
    assert _status(findings, "events_no_duplicate_rows") == {"FAIL"}


def test_naive_event_time_fails(tables):
    bars, events, _, _ = tables
    naive = events.assign(event_time=events["event_time"].dt.tz_localize(None))
    findings, ok = audit_events(naive, bars.index)
    assert not ok and _status(findings, "events_event_time_timezone") == {"FAIL"}


def test_nonchronological_events_fail(tables):
    bars, events, _, _ = tables
    findings, _ = audit_events(events.iloc[::-1].reset_index(drop=True), bars.index)
    assert _status(findings, "events_chronological_order") == {"FAIL"}


def test_missing_event_columns_and_null_direction_fail(tables):
    bars, events, _, _ = tables
    findings, ok = audit_events(events.drop(columns=["direction"]), bars.index)
    assert not ok and _status(findings, "events_schema") == {"FAIL"}
    findings, _ = audit_events(events.assign(direction=None), bars.index)
    assert _status(findings, "events_direction_present") == {"FAIL"}


def test_event_after_final_bar_fails(tables):
    bars, events, _, _ = tables
    late = events.copy()
    late.loc[len(late) - 1, "event_time"] = bars.index[-1] + pd.Timedelta(days=3)
    findings, ok = audit_events(late, bars.index)
    assert not ok and _status(findings, "events_within_data") == {"FAIL"}


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
def test_clean_features_pass(tables):
    _, events, features, _ = tables
    findings, ok = audit_features(features, events)
    assert ok and not [f for f in findings if f.status == "FAIL"]


def test_feature_asof_after_event_fails(tables):
    _, events, features, _ = tables
    late = features.assign(feature_asof_time=features["feature_asof_time"] + pd.Timedelta(minutes=1))
    findings, ok = audit_features(late, events)
    assert not ok and _status(findings, "features_asof_not_after_event") == {"FAIL"}


def test_duplicate_unknown_and_missing_feature_rows_fail(tables):
    _, events, features, _ = tables
    duplicated = pd.concat([features, features.iloc[[3]]], ignore_index=True)
    assert _status(audit_features(duplicated, events)[0], "features_one_row_per_event") == {"FAIL"}
    unknown = features.copy()
    unknown.loc[0, "event_id"] = "NOT_AN_EVENT"
    assert _status(audit_features(unknown, events)[0], "features_known_event_ids") == {"FAIL"}
    missing = features.iloc[1:].reset_index(drop=True)
    assert _status(audit_features(missing, events)[0], "features_cover_events") == {"FAIL"}


def test_documented_missing_features_are_info_not_fail(tables):
    _, events, features, _ = tables
    findings, ok = audit_features(features.iloc[1:].reset_index(drop=True), events, allow_missing_events=True)
    assert ok and _status(findings, "features_cover_events") == {"INFO"}


def test_naive_feature_asof_time_fails(tables):
    _, events, features, _ = tables
    naive = features.assign(feature_asof_time=features["feature_asof_time"].dt.tz_localize(None))
    findings, ok = audit_features(naive, events)
    assert not ok and _status(findings, "features_feature_asof_time_timezone") == {"FAIL"}


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------
def test_clean_targets_pass(tables):
    bars, events, _, targets = tables
    findings, ok = audit_targets(targets, events, bars.index)
    assert ok and not [f for f in findings if f.status == "FAIL"]


def test_target_start_before_event_fails(tables):
    bars, events, _, targets = tables
    early = targets.assign(target_start=targets["target_start"] - pd.Timedelta(days=30))
    findings, ok = audit_targets(early, events, bars.index)
    assert not ok and _status(findings, "targets_start_not_before_event") == {"FAIL"}


def test_target_end_before_start_fails(tables):
    bars, events, _, targets = tables
    inverted = targets.assign(target_end=targets["target_start"] - pd.Timedelta(hours=1))
    findings, ok = audit_targets(inverted, events, bars.index)
    assert not ok and _status(findings, "targets_end_not_before_start") == {"FAIL"}


def test_duplicate_wide_targets_fail_but_long_format_with_name_key_passes(tables):
    bars, events, _, targets = tables
    duplicated = pd.concat([targets, targets.iloc[[2]]], ignore_index=True)
    assert _status(audit_targets(duplicated, events, bars.index)[0], "targets_unique_key") == {"FAIL"}
    long = pd.concat([
        targets[["event_id", "target_start", "target_end"]].assign(target_name=name, target_value=targets[name])
        for name in (T.TARGET, "mfe_10")
    ], ignore_index=True)
    findings, ok = audit_targets(long, events, bars.index)
    assert ok and _status(findings, "targets_unique_key") == {"PASS"}
    assert np.allclose(select_target(long, T.TARGET)["value"].to_numpy(), targets[T.TARGET].to_numpy())
    duplicated_long = pd.concat([long, long.iloc[[0]]], ignore_index=True)
    assert _status(audit_targets(duplicated_long, events, bars.index)[0], "targets_unique_key") == {"FAIL"}


def test_unresolved_target_beyond_data_fails(tables):
    bars, events, _, targets = tables
    beyond = targets.copy()
    beyond.loc[len(beyond) - 1, "target_end"] = bars.index[-1] + pd.Timedelta(days=1)
    findings, ok = audit_targets(beyond, events, bars.index)
    assert not ok and _status(findings, "targets_resolved_within_data") == {"FAIL"}


def test_unknown_target_event_ids_and_naive_times_fail(tables):
    bars, events, _, targets = tables
    unknown = targets.copy()
    unknown.loc[0, "event_id"] = "GHOST"
    assert _status(audit_targets(unknown, events, bars.index)[0], "targets_known_event_ids") == {"FAIL"}
    naive = targets.assign(target_end=targets["target_end"].dt.tz_localize(None))
    assert _status(audit_targets(naive, events, bars.index)[0], "targets_target_end_timezone") == {"FAIL"}


# --------------------------------------------------------------------------------------
# Feature/target firewall
# --------------------------------------------------------------------------------------
def test_clean_firewall_passes(tables):
    _, _, features, targets = tables
    assert {f.status for f in audit_firewall(features, targets)} == {"PASS"}


def test_named_target_copy_fails_firewall_three_ways():
    findings, _, _ = T.audit("TargetCopiedNamed")
    failing = T.failing_checks(findings, "research_contract")
    assert {"firewall_reserved_names", "firewall_target_column_overlap", "firewall_target_value_identity"} <= failing


def test_disguised_target_copy_fails_value_identity():
    findings, _, _ = T.audit("TargetCopiedDisguised")
    assert "firewall_target_value_identity" in T.failing_checks(findings, "research_contract")
    assert "firewall_reserved_names" not in T.failing_checks(findings, "research_contract")


def test_scaled_target_copy_is_near_copy_warning(tables):
    _, _, features, targets = tables
    scaled = features.assign(momentum_pct=features["event_id"].map(targets.set_index("event_id")[T.TARGET]) * 100.0)
    assert _status(audit_firewall(scaled, targets), "firewall_target_near_copy") == {"WARN"}


# --------------------------------------------------------------------------------------
# Whole-contract behaviour
# --------------------------------------------------------------------------------------
def test_clean_toy_passes_full_contract():
    findings, _, _ = T.audit("CleanToy")
    contract = [f for f in findings if f.family == "research_contract"]
    assert contract and {f.status for f in contract} <= {"PASS", "INFO"}


def test_nondeterministic_event_ids_fail_determinism():
    counter = iter(range(10**6))

    class RandomIds(T.CleanToy):
        @staticmethod
        def events(bars):
            out = T.CleanToy.events(bars)
            out["event_id"] = [f"E{next(counter)}" for _ in range(len(out))]
            return out

    findings, _, _ = audit_research_contract(lambda: RandomIds, T.make_bars(), target_name=T.TARGET)
    assert _status(findings, "determinism_events") == {"FAIL"}


def test_missing_required_functions_are_unverified_not_pass():
    class EventsOnly:
        events = staticmethod(T.CleanToy.events)

    findings, _, ready = audit_research_contract(lambda: EventsOnly, T.make_bars(), target_name=T.TARGET)
    assert _status(findings, "adapter_interface") == {"UNVERIFIED"}
    assert _status(findings, "features_contract") == {"UNVERIFIED"}
    assert _status(findings, "targets_contract") == {"UNVERIFIED"}
    assert not ready["features_ok"] and not ready["targets_ok"]


def test_unknown_or_missing_target_name():
    findings, _, ready = audit_research_contract(lambda: T.CleanToy, T.make_bars(), target_name="nope")
    assert _status(findings, "target_selection") == {"FAIL"} and not ready["target_ok"]
    findings, _, ready = audit_research_contract(lambda: T.CleanToy, T.make_bars(), target_name=None)
    assert _status(findings, "target_selection") == {"UNVERIFIED"} and not ready["target_ok"]
