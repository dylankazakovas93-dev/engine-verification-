from __future__ import annotations

import json

from verifier.report import FAILED, INCOMPLETE, VERIFIED, VerificationReport, evaluate_verdict
from verifier.schema import Finding


def test_verdict_gate_distinguishes_failed_incomplete_and_verified():
    mandatory = {"data", "causality"}
    assert evaluate_verdict([Finding("data", "PASS", "ok"), Finding("causality", "PASS", "ok")], mandatory) == VERIFIED
    assert evaluate_verdict([Finding("data", "PASS", "ok"), Finding("causality", "UNVERIFIED", "missing")], mandatory) == INCOMPLETE
    assert evaluate_verdict([Finding("data", "PASS", "ok"), Finding("causality", "FAIL", "leak")], mandatory) == FAILED
    assert evaluate_verdict([Finding("data", "PASS", "ok")], mandatory) == INCOMPLETE


def test_optional_failure_does_not_override_all_mandatory_passes():
    findings = [
        Finding("data", "PASS", "ok", family="data"),
        Finding("optional_diagnostic", "FAIL", "optional discrepancy", family="optional"),
    ]
    assert evaluate_verdict(findings, {"data"}) == VERIFIED


def test_mandatory_failure_dominates_another_incomplete_mandatory_check():
    findings = [
        Finding("repository", "UNVERIFIED", "tests skipped", family="repository"),
        Finding("ledger", "FAIL", "overlapping positions", family="ledger"),
    ]
    assert evaluate_verdict(findings, {"repository", "ledger"}) == FAILED


def test_json_and_markdown_reports_preserve_machine_verdict_and_exit_code(tmp_path):
    report = VerificationReport(
        strategy="toy", candidate="bad.py",
        findings=[Finding("roll", "UNVERIFIED", "ROLL PROVENANCE UNVERIFIED", family="contract")],
        mandatory_checks={"contract"}, coverage={"cutoff_count": 12},
    )
    json_path = report.write_json(tmp_path / "report.json")
    md_path = report.write_markdown(tmp_path / "report.md")
    payload = json.loads(json_path.read_text())
    assert payload["verdict"] == INCOMPLETE
    assert payload["exit_code"] == 2
    assert "INCOMPLETE / UNVERIFIED" in md_path.read_text()
