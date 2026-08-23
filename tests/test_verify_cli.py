from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def write_bars(path: Path, n: int = 360):
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    close = 15000 + np.sin(np.arange(n) / 11)
    pd.DataFrame({
        "timestamp": timestamps, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": 1,
    }).to_csv(path, index=False)


def run_verify(repo: Path, *args: str):
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "verify.py"), *args],
        cwd=repo, text=True, capture_output=True,
    )


def test_one_command_known_good_is_incomplete_when_roll_provenance_is_missing(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "bars.csv"
    write_bars(data)
    prefix = tmp_path / "good_report"
    result = run_verify(
        repo, "--strategy", "nq_frozen_v1", "--source", str(repo / "nq_frozen"),
        "--data", str(data), "--timestamp-col", "timestamp", "--audit-mode", "fast",
        "--skip-tests", "--report-prefix", str(prefix),
    )
    assert result.returncode == 2, result.stdout + result.stderr
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["verdict"] == "INCOMPLETE / UNVERIFIED"
    assert any(f["check"] == "contract_provenance" and f["status"] == "UNVERIFIED" for f in payload["findings"])
    assert prefix.with_suffix(".md").exists()


def test_one_command_broken_candidate_returns_failed(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "bars.csv"
    write_bars(data)
    adapter = tmp_path / "broken_adapter.py"
    adapter.write_text("""
import pandas as pd
def run(bars):
    columns = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'deadline', 'direction']
    if len(bars) < 22:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({
        'entry_time':[bars.index[10], bars.index[11]],
        'exit_time':[bars.index[20], bars.index[21]],
        'entry_price':[100.0, 100.0], 'exit_price':[101.0, 101.0],
        'deadline':[bars.index[20] + pd.Timedelta(hours=1), bars.index[21] + pd.Timedelta(hours=1)],
        'direction':['long', 'long'],
    })
""", encoding="utf-8")
    prefix = tmp_path / "bad_report"
    result = run_verify(
        repo, "--strategy", "nq_frozen_v1", "--candidate", str(adapter),
        "--data", str(data), "--timestamp-col", "timestamp", "--audit-mode", "fast",
        "--skip-tests", "--report-prefix", str(prefix),
    )
    assert result.returncode == 1, result.stdout + result.stderr
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["verdict"] == "FAILED"
    assert any(f["check"] == "one_global_position" and f["status"] == "FAIL" for f in payload["findings"])
