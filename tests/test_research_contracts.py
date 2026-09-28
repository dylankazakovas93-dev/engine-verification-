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


def known(bars):
    """Information clock: an open-stamped bar is known at open + bar_interval."""
    return bars.index + T.BAR


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
    findings, ok = audit_events(events, known(bars))
    assert ok and not [f for f in findings if f.status == "FAIL"]


def test_duplicate_event_ids_and_rows_fail(tables):
    bars, events, _, _ = tables
    broken = pd.concat([events, events.iloc[[5]]], ignore_index=True)
    findings, ok = audit_events(broken, known(bars))
    assert not ok
    assert _status(findings, "events_id_unique") == {"FAIL"}
    assert _status(findings, "events_no_duplicate_rows") == {"FAIL"}


def test_naive_event_time_fails(tables):
    bars, events, _, _ = tables
    naive = events.assign(event_time=events["event_time"].dt.tz_localize(None))
    findings, ok = audit_events(naive, known(bars))
    assert not ok and _status(findings, "events_event_time_timezone") == {"FAIL"}


def test_nonchronological_events_fail(tables):
    bars, events, _, _ = tables
    findings, _ = audit_events(events.iloc[::-1].reset_index(drop=True), known(bars))
    assert _status(findings, "events_chronological_order") == {"FAIL"}


def test_missing_event_columns_and_null_direction_fail(tables):
    bars, events, _, _ = tables
    findings, ok = audit_events(events.drop(columns=["direction"]), known(bars))
    assert not ok and _status(findings, "events_schema") == {"FAIL"}
    findings, _ = audit_events(events.assign(direction=None), known(bars))
    assert _status(findings, "events_direction_present") == {"FAIL"}


def test_event_after_final_bar_fails(tables):
    bars, events, _, _ = tables
    late = events.copy()
    late.loc[len(late) - 1, "event_time"] = bars.index[-1] + pd.Timedelta(days=3)
    findings, ok = audit_events(late, known(bars))
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


def test_no_candidate_controlled_feature_omission_even_for_warm_up():
    """Warm-up eligibility belongs to the frozen feature spec, not the candidate: a head omission
    FAILs even when the adapter sets the retired FEATURES_ALLOW_MISSING_EVENTS flag."""
    class HeadOmission(T.CleanToy):
        FEATURES_ALLOW_MISSING_EVENTS = True

        @staticmethod
        def features(bars, events):
            return T.CleanToy.features(bars, events).iloc[5:].reset_index(drop=True)

    findings, _, ready = audit_research_contract(lambda: HeadOmission, T.make_bars(), target_name=T.TARGET,
                                                 bar_interval=T.BAR, target_horizon=T.HORIZON_SPEC)
    assert _status(findings, "features_cover_events") == {"FAIL"} and not ready["features_ok"]


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
    findings, ok = audit_targets(targets, events, known(bars))
    assert ok and not [f for f in findings if f.status == "FAIL"]


def test_target_start_before_event_fails(tables):
    bars, events, _, targets = tables
    early = targets.assign(target_start=targets["target_start"] - pd.Timedelta(days=30))
    findings, ok = audit_targets(early, events, known(bars))
    assert not ok and _status(findings, "targets_start_not_before_event") == {"FAIL"}


def test_target_end_before_start_fails(tables):
    bars, events, _, targets = tables
    inverted = targets.assign(target_end=targets["target_start"] - pd.Timedelta(hours=1))
    findings, ok = audit_targets(inverted, events, known(bars))
    assert not ok and _status(findings, "targets_end_not_before_start") == {"FAIL"}


def test_duplicate_wide_targets_fail_but_long_format_with_name_key_passes(tables):
    bars, events, _, targets = tables
    duplicated = pd.concat([targets, targets.iloc[[2]]], ignore_index=True)
    assert _status(audit_targets(duplicated, events, known(bars))[0], "targets_unique_key") == {"FAIL"}
    long = pd.concat([
        targets[["event_id", "target_start", "target_end"]].assign(target_name=name, target_value=targets[name])
        for name in (T.TARGET, "mfe_10")
    ], ignore_index=True)
    findings, ok = audit_targets(long, events, known(bars))
    assert ok and _status(findings, "targets_unique_key") == {"PASS"}
    assert np.allclose(select_target(long, T.TARGET)["value"].to_numpy(), targets[T.TARGET].to_numpy())
    duplicated_long = pd.concat([long, long.iloc[[0]]], ignore_index=True)
    assert _status(audit_targets(duplicated_long, events, known(bars))[0], "targets_unique_key") == {"FAIL"}


def test_unresolved_target_beyond_data_fails(tables):
    bars, events, _, targets = tables
    beyond = targets.copy()
    beyond.loc[len(beyond) - 1, "target_end"] = known(bars)[-1] + pd.Timedelta(minutes=1)
    findings, ok = audit_targets(beyond, events, known(bars))
    assert not ok and _status(findings, "targets_resolved_within_data") == {"FAIL"}


def test_unknown_target_event_ids_and_naive_times_fail(tables):
    bars, events, _, targets = tables
    unknown = targets.copy()
    unknown.loc[0, "event_id"] = "GHOST"
    assert _status(audit_targets(unknown, events, known(bars))[0], "targets_known_event_ids") == {"FAIL"}
    naive = targets.assign(target_end=targets["target_end"].dt.tz_localize(None))
    assert _status(audit_targets(naive, events, known(bars))[0], "targets_target_end_timezone") == {"FAIL"}


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

    findings, _, _ = audit_research_contract(lambda: RandomIds, T.make_bars(), target_name=T.TARGET, bar_interval=T.BAR)
    assert _status(findings, "determinism_events") == {"FAIL"}


def test_missing_required_functions_are_unverified_not_pass():
    class EventsOnly:
        events = staticmethod(T.CleanToy.events)

    findings, _, ready = audit_research_contract(lambda: EventsOnly, T.make_bars(), target_name=T.TARGET, bar_interval=T.BAR)
    assert _status(findings, "adapter_interface") == {"UNVERIFIED"}
    assert _status(findings, "features_contract") == {"UNVERIFIED"}
    assert _status(findings, "targets_contract") == {"UNVERIFIED"}
    assert not ready["features_ok"] and not ready["targets_ok"]


def test_unknown_or_missing_target_name():
    findings, _, ready = audit_research_contract(lambda: T.CleanToy, T.make_bars(), target_name="nope", bar_interval=T.BAR)
    assert _status(findings, "target_selection") == {"FAIL"} and not ready["target_ok"]
    findings, _, ready = audit_research_contract(lambda: T.CleanToy, T.make_bars(), target_name=None, bar_interval=T.BAR)
    assert _status(findings, "target_selection") == {"UNVERIFIED"} and not ready["target_ok"]


# --------------------------------------------------------------------------------------
# Information clock (bar availability)
# --------------------------------------------------------------------------------------
def test_bar_interval_is_mandatory_and_positive():
    from verifier.research_contracts import parse_bar_interval
    with pytest.raises(ValueError):
        parse_bar_interval(None)
    with pytest.raises(ValueError):
        parse_bar_interval("0s")
    assert parse_bar_interval("1min") == pd.Timedelta(minutes=1)


def test_declared_bar_interval_shorter_than_bar_spacing_fails():
    from verifier.research_contracts import audit_bar_interval
    bars = T.make_bars()
    assert _status(audit_bar_interval(bars, pd.Timedelta("1D")), "bar_interval_declaration") == {"PASS"}
    assert _status(audit_bar_interval(bars, pd.Timedelta("1h")), "bar_interval_declaration") == {"FAIL"}


def _minute_bars(index):
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=index)


def test_bar_interval_tolerates_missing_bars_and_session_gaps():
    from verifier.research_contracts import audit_bar_interval
    full = pd.date_range("2024-01-02 14:30", periods=20_000, freq="1min", tz="UTC")
    rng = np.random.default_rng(3)
    kept = full[np.sort(rng.choice(len(full), size=int(len(full) * 0.6), replace=False))]  # 40% of bars missing
    sessions = kept[(kept.hour >= 14) & (kept.hour < 21)]  # plus daily session gaps
    findings = audit_bar_interval(_minute_bars(sessions), pd.Timedelta("1min"))
    assert _status(findings, "bar_interval_declaration") == {"PASS"}
    assert findings[0].evidence["gaps_longer_than_declared"] > 1000


def test_bar_interval_too_short_for_coarse_bars_fails_despite_stray_stamps():
    from verifier.research_contracts import audit_bar_interval
    coarse = pd.date_range("2024-01-02", periods=5_000, freq="5min", tz="UTC")
    stray = coarse[:20] + pd.Timedelta("1min")  # 0.4% off-grid stamps must not make the check lenient
    index = coarse.append(stray).sort_values()
    assert _status(audit_bar_interval(_minute_bars(index), pd.Timedelta("1min")), "bar_interval_declaration") == {"FAIL"}
    assert _status(audit_bar_interval(_minute_bars(index), pd.Timedelta("5min")), "bar_interval_declaration") == {"PASS"}


def test_adapter_receives_bar_interval_in_attrs():
    from verifier.research_contracts import run_pipeline
    seen = {}

    class Spy(T.CleanToy):
        @staticmethod
        def events(bars):
            seen["interval"] = bars.attrs.get("bar_interval")
            return T.CleanToy.events(bars)

    bars = T.make_bars()
    bars.attrs.clear()
    run_pipeline(Spy, bars, pd.Timedelta("1D"))
    assert seen["interval"] == pd.Timedelta("1D")


def test_target_window_starting_on_the_signal_bar_fails():
    findings, _, _ = T.audit("TargetStartsOnSignalBar")
    assert "targets_start_not_before_event" in T.failing_checks(findings, "research_contract")


# --------------------------------------------------------------------------------------
# Outcome-dependent sample selection
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["OutcomeSelectedTargets", "OutcomeNaNTargets"])
def test_outcome_selected_labels_fail_target_coverage(name):
    findings, _, artifacts = T.audit(name)
    assert "target_coverage" in T.failing_checks(findings, "research_contract")
    assert artifacts["research_readiness"]["target_ok"] is False


def test_clean_tail_censoring_passes_target_coverage():
    findings, _, artifacts = T.audit("CleanToy")
    coverage = [f for f in findings if f.check == "target_coverage"]
    assert [f.status for f in coverage] == ["PASS"]
    ev = coverage[0].evidence
    assert ev["declared_horizon"] == "10 bars"
    assert ev["unresolvable_tail"] > 0 and ev["missing_required"] == 0
    assert artifacts["research_readiness"]["eligible_event_count"] == ev["required"]


def _selection(tables, targets, horizon):
    from verifier.research_contracts import audit_target_selection, parse_target_horizon
    bars, events, _, _ = tables
    return audit_target_selection(targets, T.TARGET, events, bars.index, T.BAR, parse_target_horizon(horizon))


def test_single_mid_sample_missing_label_fails(tables):
    _, _, _, targets = tables
    findings, ok, _ = _selection(tables, targets.drop(index=200).reset_index(drop=True), T.HORIZON_SPEC)
    failing = [f for f in findings if f.check == "target_coverage"]
    assert not ok and [f.status for f in failing] == ["FAIL"]
    assert targets.loc[200, "event_id"] in failing[0].evidence["illegal_ids"]


def test_coverage_is_unverified_without_a_declared_horizon(tables):
    findings, ok, eligible = _selection(tables, tables[3], None)
    assert _status(findings, "target_coverage") == {"UNVERIFIED"}
    assert ok and len(eligible) == len(tables[3])


def test_coverage_does_not_depend_on_which_labels_the_candidate_emits(tables):
    """The old H_max inference let a candidate widen its own tail by emitting a long label. The
    declared horizon fixes the required set from timestamps alone."""
    _, _, _, targets = tables
    _, _, eligible_full = _selection(tables, targets, T.HORIZON_SPEC)
    trimmed = targets.iloc[:-40].reset_index(drop=True)  # candidate withholds its last 40 labels
    findings, ok, _ = _selection(tables, trimmed, T.HORIZON_SPEC)
    assert not ok and _status(findings, "target_coverage") == {"FAIL"}
    assert len(eligible_full) == len(targets)


def test_labels_beyond_declared_horizon_fail():
    findings, _, _ = T.audit("TargetBeyondDeclaredHorizon")
    assert "target_horizon_respected" in T.failing_checks(findings, "research_contract")


def test_labels_on_events_with_unresolvable_declared_horizon_are_excluded_by_verifier(tables):
    _, _, _, targets = tables
    findings, ok, eligible = _selection(tables, targets, "15bars")  # declared horizon longer than emitted
    assert ok and _status(findings, "target_tail_exclusion") == {"INFO"}
    # Events on the 5 extra tail bars drop out (events fall on every 3rd bar -> 1 or 2 events).
    assert 1 <= len(targets) - len(eligible) <= 2


def test_duration_horizon_is_supported(tables):
    _, _, _, targets = tables
    findings, ok, _ = _selection(tables, targets, "30D")
    assert _status(findings, "target_coverage") == {"PASS"} and ok


def test_mid_sample_feature_omission_fails(tables):
    _, events, features, _ = tables
    findings, ok = audit_features(features.drop(index=200).reset_index(drop=True), events)
    assert not ok and _status(findings, "features_cover_events") == {"FAIL"}
