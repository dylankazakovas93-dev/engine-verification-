from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def test_new_strategy_scaffold_is_isolated_and_spec_first(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / "templates", tmp_path / "templates")
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "new_strategy.py"), "my_strategy", "--root", str(tmp_path)],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    strategy = tmp_path / "strategies" / "my_strategy"
    assert {"SPECIFICATION.md", "REQUIREMENTS.md", "ORACLE.md", "profile.toml", "adapter.py", "tests"} <= {p.name for p in strategy.iterdir()}
    assert "STOP GATE" in result.stdout
    assert "Do not implement" in (strategy / "SPECIFICATION.md").read_text()


def test_new_strategy_rejects_unsafe_name(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "new_strategy.py"), "../escape", "--root", str(tmp_path), "--dry-run"],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
