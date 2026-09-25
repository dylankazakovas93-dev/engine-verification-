from __future__ import annotations

import pandas as pd

import research_toys as T
from verifier import causality, research_causality
from verifier.research_causality import audit_research_causality, generate_research_cutoffs, targets_before
from verifier.research_contracts import run_pipeline

FAMILY = "research_causality"


def _run(toy, mode="fast", cutoffs=None):
    bars = T.make_bars()
    full = run_pipeline(toy, bars)
    return audit_research_causality(lambda: toy, bars, full, mode=mode, cutoffs=cutoffs)


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


def test_research_cutoffs_include_bar_immediately_after_events_and_target_ends():
    bars = T.make_bars()
    full = run_pipeline(T.CleanToy, bars)
    chosen = set(generate_research_cutoffs(bars, full, mode="strong"))
    first_event = full.events["event_time"].iloc[0]
    first_target_end = full.targets["target_end"].min()
    assert bars.index[bars.index.searchsorted(first_event, side="right")] in chosen
    assert bars.index[bars.index.searchsorted(first_target_end, side="right")] in chosen


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
