from __future__ import annotations

import pandas as pd

import research_toys as T
from verifier import causality, research_causality
from verifier.research_causality import audit_research_causality, generate_research_cutoffs, targets_before
from verifier.research_contracts import run_pipeline

FAMILY = "research_causality"


def test_regression_target_end_declared_at_final_bar_open_fails():
    """REGRESSION: a label built from the final window bar's close/high/low must declare
    target_end = that bar's open + bar_interval. Declaring the bar's OPEN claims resolution one bar
    early; the verifier must compare it at a cutoff where that bar is incomplete and FAIL."""
    findings, _ = _run(T.TargetEndAtFinalBarOpen)
    failing = [f for f in findings if f.status == "FAIL" and f.check in {"truncation_targets", "future_mutation_targets"}]
    assert {f.check for f in failing} == {"truncation_targets", "future_mutation_targets"}
    # The failure happens exactly inside the one-bar window between the claimed and true resolution.
    bars = T.make_bars()
    claimed = set(bars.index)  # claimed ends are bar opens
    cutoffs = {pd.Timestamp(f.evidence["cutoff"]) for f in failing}
    assert any((c - T.BAR) in claimed for c in cutoffs)
    # Contract checks alone cannot see it (the claim is earlier than the declared horizon); it is a
    # dynamic, proven causality failure.
    contract, _, _ = T.audit("TargetEndAtFinalBarOpen")
    assert not T.failing_checks(contract, "research_contract")
    assert T.failing_checks(contract, FAMILY) >= {"truncation_targets", "future_mutation_targets"}


def test_declared_resolution_time_is_nth_bar_open_plus_interval_with_gaps():
    from verifier.research_contracts import TargetHorizon, declared_resolution_times
    minute = pd.Timedelta("1min")
    opens = pd.date_range("2024-01-02 09:30", periods=120, freq="1min", tz="UTC")
    opens = opens.delete(opens.get_loc(pd.Timestamp("2024-01-02 10:00", tz="UTC")))  # one missing source bar
    events = pd.DataFrame({"event_id": ["a", "b"], "event_time": pd.to_datetime(
        ["2024-01-02 09:31", "2024-01-02 11:00"], utc=True)})
    ends = declared_resolution_times(events, opens, minute, TargetHorizon(bars=60))
    # 60 available bars opening at/after 09:31 with 10:00 missing: last opens at 10:31 -> resolved 10:32.
    assert ends["a"] == pd.Timestamp("2024-01-02 10:32", tz="UTC")
    assert pd.isna(ends["b"])  # fewer than 60 bars after 11:00 in the data -> unresolvable tail
    contiguous = pd.date_range("2024-01-02 09:30", periods=120, freq="1min", tz="UTC")
    assert declared_resolution_times(events.iloc[:1], contiguous, minute, TargetHorizon(bars=60))["a"] == \
        pd.Timestamp("2024-01-02 10:31", tz="UTC")  # final included bar opens 10:30 -> resolved 10:31


def _run(toy, mode="fast", cutoffs=None):
    bars = T.make_bars()
    full = run_pipeline(toy, bars, T.BAR)
    return audit_research_causality(lambda: toy, bars, full, bar_interval=T.BAR, mode=mode, cutoffs=cutoffs)


def test_reuses_existing_mutation_and_cutoff_primitives():
    assert research_causality.mutate_future is causality.mutate_future
    assert research_causality.generate_cutoffs is causality.generate_cutoffs
    assert research_causality._equal is causality._equal


def test_clean_toy_is_invariant_at_every_strong_cutoff():
    findings, coverage = _run(T.CleanToy, mode="strong")
    cov = coverage.to_dict()
    assert cov["cutoff_count"] >= 28
    assert cov["execution_errors"] == 0
    assert not [f for f in findings if f.status in {"FAIL", "UNVERIFIED", "WARN"}]
    for key in ("truncation_events", "future_mutation_events", "truncation_features",
                "future_mutation_features", "truncation_targets", "future_mutation_targets"):
        assert cov["rows_compared"][key] > 1000, key
        assert cov["failures"][key] == 0


def test_research_cutoffs_are_on_the_information_clock_right_after_events_and_target_ends():
    bars = T.make_bars()
    full = run_pipeline(T.CleanToy, bars, T.BAR)
    chosen = set(generate_research_cutoffs(bars, full, bar_interval=T.BAR, mode="strong"))
    known = bars.index + T.BAR
    assert chosen <= set(known)  # every generated cutoff is a bar-completion time
    first_event = full.events["event_time"].iloc[0]
    first_target_end = full.targets["target_end"].min()
    assert known[known.searchsorted(first_event, side="right")] in chosen
    assert known[known.searchsorted(first_target_end, side="right")] in chosen


def test_event_stamped_at_bar_open_but_using_its_close_fails():
    """The 09:30-bar problem: close is used, but the event claims to be known at the open."""
    findings, _ = _run(T.EventAtBarStart)
    failing = T.failing_checks(findings, FAMILY)
    assert {"truncation_events", "truncation_features"} <= failing


def test_same_cheat_passed_under_the_old_bar_open_clock():
    """Documents WHY the clock matters: with truncation on raw open stamps (the pre-fix
    behaviour, reproduced here with an explicit bar-open cutoff grid) the cheat was invisible."""
    bars = T.make_bars()
    toy = T.EventAtBarStart
    full = run_pipeline(toy, bars, T.BAR)
    open_grid = list(bars.index[25:1500:37])
    old = []
    for cutoff in open_grid:
        prefix = run_pipeline(toy, bars[bars.index < cutoff], T.BAR)
        old.append(research_causality._equal(
            research_causality.events_before(full.events, cutoff), research_causality.events_before(prefix.events, cutoff))[0])
    assert all(old)
    findings, _ = _run(toy, cutoffs=open_grid)  # same cutoffs, information-clock truncation
    assert "truncation_events" in T.failing_checks(findings, FAMILY)


def test_target_claiming_resolution_at_last_bar_open_fails():
    findings, _ = _run(T.TargetEndAtBarStart)
    assert {"truncation_targets", "future_mutation_targets"} <= T.failing_checks(findings, FAMILY)


def test_full_sample_mean_feature_fails_future_mutation_and_truncation():
    findings, _ = _run(T.GlobalMeanFeature)
    failing = T.failing_checks(findings, FAMILY)
    assert {"future_mutation_features", "truncation_features"} <= failing
    assert "truncation_events" not in failing  # the event generator itself is causal


def test_one_bar_event_lookahead_fails_event_causality():
    findings, _ = _run(T.OneBarLookaheadEvents)
    assert "truncation_events" in T.failing_checks(findings, FAMILY)
    evidence = [f.evidence for f in findings if f.check == "truncation_events" and f.status == "FAIL"]
    # Exact failing cutoff and the event IDs that only exist with future knowledge are reported.
    assert evidence and evidence[0]["cutoff"] and (evidence[0]["only_in_full"] or evidence[0]["only_in_comparison"])


def test_disguised_target_copy_fails_feature_causality():
    findings, _ = _run(T.TargetCopiedDisguised)
    assert {"future_mutation_features", "truncation_features"} <= T.failing_checks(findings, FAMILY)


def test_target_using_bar_after_declared_target_end_fails():
    findings, _ = _run(T.TargetPeeksBeyondEnd)
    failing = T.failing_checks(findings, FAMILY)
    assert {"truncation_targets", "future_mutation_targets"} <= failing
    assert "truncation_features" not in failing


def test_feature_depending_on_data_after_declared_asof_fails():
    findings, _ = _run(T.FeatureAsofLies)
    assert "future_mutation_features_asof" in T.failing_checks(findings, FAMILY)


def test_targets_crossing_the_cutoff_are_not_compared():
    targets = pd.DataFrame({
        "event_id": ["a", "b"],
        "target_start": pd.to_datetime(["2020-01-01 10:00", "2020-01-01 10:00"], utc=True),
        "target_end": pd.to_datetime(["2020-01-01 11:00", "2020-01-01 13:00"], utc=True),
        "y": [1.0, 2.0],
    })
    kept = targets_before(targets, pd.Timestamp("2020-01-01 12:00", tz="UTC"))
    assert kept["event_id"].tolist() == ["a"]


def test_no_comparable_rows_is_unverified_not_pass():
    bars = T.make_bars()
    early_cutoffs = [bars.index[3], bars.index[5]]  # before the first event (position 21)
    findings, _ = _run(T.CleanToy, cutoffs=early_cutoffs)
    assert {f.status for f in findings if f.check == "truncation_events"} == {"UNVERIFIED"}


def test_candidate_crashing_on_truncated_input_is_unverified():
    class CrashesOnPrefix(T.CleanToy):
        @staticmethod
        def events(bars):
            if len(bars) < 1000:
                raise RuntimeError("needs the full history")
            return T.CleanToy.events(bars)

    findings, coverage = _run(CrashesOnPrefix)
    assert coverage.execution_errors > 0
    assert any(f.check == "truncation_execution" and f.status == "UNVERIFIED" for f in findings)


def test_missing_features_and_targets_are_unverified():
    class EventsOnly:
        events = staticmethod(T.CleanToy.events)

    findings, _ = _run(EventsOnly)
    statuses = T.statuses(findings, FAMILY)
    assert statuses["truncation_features"] == {"UNVERIFIED"}
    assert statuses["future_mutation_targets"] == {"UNVERIFIED"}
    assert statuses["truncation_events"] == {"PASS"}


def test_schema_change_under_truncation_fails_instead_of_crashing():
    class SchemaShift(T.CleanToy):
        @staticmethod
        def features(bars, events):
            out = T.CleanToy.features(bars, events)
            return out if len(bars) > 1500 else out.rename(columns={"feature_asof_time": "asof"})

    findings, _ = _run(SchemaShift)
    assert "truncation_schema" in T.failing_checks(findings, FAMILY)
