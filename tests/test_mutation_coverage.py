from __future__ import annotations

import pandas as pd
import pytest

from verifier.ledger import audit_ledger
from verifier.reconciliation import reconcile_ledgers
from verifier.static_scan import scan_source


def status(findings, check):
    return [finding.status for finding in findings if finding.check == check]


def canonical_trade():
    return {
        "signal_time": "2026-01-01T10:00:00Z", "entry_time": "2026-01-01T10:01:00Z",
        "exit_time": "2026-01-01T10:02:00Z", "entry_price": 100.0, "exit_price": 90.0,
        "exit_reason": "SL", "initial_risk": 10.0, "stop_price": 90.0,
        "target_price": 105.0, "deadline": "2026-01-01T20:00:00Z",
        "atr_source_end": "2026-01-01T08:00:00Z", "gross_R": -1.0,
        "cost_R": 0.1, "net_R": -1.1, "direction": "long",
    }


@pytest.mark.parametrize(("mutator", "expected_check"), [
    (lambda rows: rows + [{**rows[0], "signal_time": "2026-01-01T10:01:00Z", "entry_time": "2026-01-01T10:01:30Z", "exit_time": "2026-01-01T10:03:00Z"}], "one_global_position"),
    (lambda rows: rows + [{**rows[0], "signal_time": "2026-01-01T10:01:00Z", "entry_time": "2026-01-01T10:02:00Z", "exit_time": "2026-01-01T10:03:00Z"}], "one_global_position"),
    (lambda rows: [{**rows[0], "entry_time": rows[0]["signal_time"]}], "entry_after_signal"),
    (lambda rows: [{**rows[0], "exit_time": "2026-01-01T20:01:00Z"}], "deadline"),
    (lambda rows: [{**rows[0], "atr_source_end": "2026-01-01T10:02:00Z"}], "atr_causality"),
    (lambda rows: [{**rows[0], "stop_price": 90.1}], "stop_price_tick_alignment"),
    (lambda rows: [{**rows[0], "cost_R": 0.0}], "r_accounting"),
    (lambda rows: [{**rows[0], "direction": "short"}], "long_only"),
])
def test_ledger_mutations_are_rejected(mutator, expected_check):
    candidate = pd.DataFrame(mutator([canonical_trade()]))
    findings = audit_ledger(candidate, strict_same_timestamp_reentry=True, tick_size=0.25)
    assert status(findings, expected_check) == ["FAIL"]


@pytest.mark.parametrize(("changes", "meaning"), [
    ({"exit_reason": "TP", "exit_price": 105.0, "gross_R": 0.5, "net_R": 0.4}, "wrong same-bar TP/SL ordering"),
    ({"exit_reason": "TP", "exit_price": 112.0, "gross_R": 0.5, "net_R": 0.4}, "target-gap price improvement"),
    ({"exit_price": 87.0, "gross_R": -1.0, "net_R": -1.1}, "stop gap capped at minus one R"),
    ({"cost_R": 0.0, "net_R": -1.0}, "incorrect transaction cost"),
])
def test_strategy_semantic_mutations_fail_trade_by_trade_reference(changes, meaning):
    reference = pd.DataFrame([canonical_trade()])
    candidate = reference.copy()
    for key, value in changes.items():
        candidate.loc[0, key] = value
    findings = reconcile_ledgers(candidate, reference)
    assert findings[0].status == "FAIL", meaning


def test_queued_signal_and_is_oos_logic_mutations_fail_reference_count():
    reference = pd.DataFrame([canonical_trade()])
    queued = pd.concat([reference, reference.assign(
        signal_time="2026-01-01T10:00:30Z", entry_time="2026-01-01T10:01:30Z", exit_time="2026-01-01T10:04:00Z"
    )], ignore_index=True)
    assert reconcile_ledgers(queued, reference)[0].status == "FAIL"
    removed_by_sample_label = reference.iloc[0:0]
    assert reconcile_ledgers(removed_by_sample_label, reference)[0].status == "FAIL"


def test_source_repair_and_leakage_mutations_are_flagged(tmp_path):
    source = tmp_path / "broken.py"
    source.write_text(
        "future = close.shift(-1)\n"
        "bars = bars.sort_index().drop_duplicates().bfill()\n",
        encoding="utf-8",
    )
    checks = {finding.check for finding in scan_source(source)}
    assert {"static_negative_shift", "static_silent_sort", "static_drop_duplicates", "static_backfill"} <= checks


def test_profile_required_deadline_and_direction_cannot_be_omitted():
    candidate = pd.DataFrame([{k: v for k, v in canonical_trade().items() if k not in {"deadline", "direction"}}])
    findings = audit_ledger(candidate, long_only=True, required_deadline=True)
    assert findings[0].check == "ledger_schema"
    assert findings[0].status == "FAIL"
