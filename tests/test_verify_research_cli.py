from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import research_toys as T

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "templates" / "research" / "adapter.py"


def _write_bars(path: Path) -> None:
    bars = T.make_bars()
    bars.rename_axis("timestamp").reset_index().to_csv(path, index=False)


def _toy_adapter(path: Path, toy: str) -> Path:
    path.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO / 'tests')!r})\n"
        f"from research_toys import {toy} as _Toy\n"
        "events = _Toy.events\nfeatures = _Toy.features\ntargets = _Toy.targets\n"
        "if hasattr(_Toy, 'fit_predict_fold'):\n    fit_predict_fold = _Toy.fit_predict_fold\n",
        encoding="utf-8",
    )
    return path


def _run(tmp_path: Path, adapter: Path, target: str, *extra: str):
    data = tmp_path / "bars.csv"
    _write_bars(data)
    prefix = tmp_path / "report"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_research.py"), "--adapter", str(adapter),
         "--data", str(data), "--timestamp-col", "timestamp", "--target", target,
         "--lockbox-start", T.LOCKBOX, "--mode", "fast", "--skip-tests", "--report-prefix", str(prefix), *extra],
        cwd=REPO, text=True, capture_output=True,
    )
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert prefix.with_suffix(".md").exists()
    return result, payload


def test_clean_template_passes_all_research_families_and_is_incomplete_only_for_skipped_tests_and_roll_provenance(tmp_path):
    result, payload = _run(tmp_path, TEMPLATE, "forward_return_10")
    assert result.returncode == 2, result.stdout + result.stderr
    assert payload["verdict"] == "INCOMPLETE / UNVERIFIED"
    status = payload["artifacts"]["family_status"]
    for family in ("research_contract", "research_causality", "walkforward", "ml_leakage", "lockbox", "ml_static_scan", "data_audit"):
        assert status[family] == "PASS", (family, status)
    not_passing = {k for k, v in status.items() if v != "PASS" and k in payload["mandatory_checks"]}
    assert not_passing == {"repository_health", "contract_provenance"}
    assert payload["artifacts"]["lockbox"]["withheld_lockbox_event_count"] > 0
    assert "VERDICT: INCOMPLETE / UNVERIFIED" in result.stdout


def test_cheating_adapter_returns_failed_with_exit_code_1(tmp_path):
    adapter = _toy_adapter(tmp_path / "leaky_adapter.py", "ValidationLabelLeak")
    result, payload = _run(tmp_path, adapter, T.TARGET)
    assert result.returncode == 1, result.stdout + result.stderr
    assert payload["verdict"] == "FAILED"
    assert any(f["check"] == "validation_label_poisoning" and f["status"] == "FAIL" for f in payload["findings"])


def test_adapter_without_model_is_ml_leakage_unverified(tmp_path):
    adapter = _toy_adapter(tmp_path / "no_model_adapter.py", "NoModel")
    result, payload = _run(tmp_path, adapter, T.TARGET)
    assert result.returncode == 2
    assert payload["artifacts"]["family_status"]["ml_leakage"] == "UNVERIFIED"


def test_full_table_statistic_is_flagged_for_manual_review(tmp_path):
    source = tmp_path / "scaling_source.py"
    source.write_text(
        "def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):\n"
        "    x = features.set_index('event_id')[FEATURE_COLUMNS]\n"
        "    x = (x - x.mean()) / x.std()\n",
        encoding="utf-8",
    )
    adapter = _toy_adapter(tmp_path / "ols_adapter.py", "FullSampleScalingOLS")
    result, payload = _run(tmp_path, adapter, T.TARGET, "--source", str(source))
    assert payload["artifacts"]["family_status"]["ml_static_scan"] == "UNVERIFIED"
    assert result.returncode == 2
