from __future__ import annotations

from verifier.static_scan import HIGH, INFO, scan_source


def checks(tmp_path, source):
    path = tmp_path / "candidate.py"
    path.write_text(source, encoding="utf-8")
    return {finding.check: finding for finding in scan_source(path)}


def test_ast_detects_future_indexing_roll_center_fit_and_forward_join(tmp_path):
    found = checks(tmp_path, """
import numpy as np
x = s.iloc[i + 1]
y = np.roll(s, -2)
z = s.rolling(10, center=True).mean()
model.fit_transform(all_rows)
joined = merge_asof(a, b, direction='forward')
""")
    for name in ("static_future_iloc", "static_negative_roll", "static_centered_rolling", "static_full_sample_fit", "static_forward_asof"):
        assert name in found
        assert found[name].classification == HIGH


def test_ast_classifies_global_stat_and_future_label_as_info(tmp_path):
    found = checks(tmp_path, "future_return = close.shift(-1) / close\nmu = close.mean()\n")
    assert found["static_future_label"].classification == INFO
    assert found["static_full_sample_statistic"].classification == INFO
    assert found["static_negative_shift"].classification == HIGH


def test_ast_detects_input_repair_patterns(tmp_path):
    found = checks(tmp_path, "x = bars.sort_values('timestamp').drop_duplicates().bfill().interpolate()\n")
    for name in ("static_silent_sort", "static_drop_duplicates", "static_backfill", "static_interpolate"):
        assert name in found


def test_ast_does_not_call_current_bar_exclusive_slice_future_data(tmp_path):
    checks(tmp_path, "window = x[i - m + 1 : i + 1]\nfuture = x[i + 1 :]\n")
    future_slices = [
        finding for finding in scan_source(tmp_path / "candidate.py")
        if finding.check == "static_future_slice"
    ]
    assert len(future_slices) == 1
    assert future_slices[0].evidence["line"] == 2
