"""Regression: event-adjacent cutoffs in verifier.causality.generate_cutoffs must not depend on the
DatetimeIndex storage resolution. Under pandas >= 3 an index is often stored in microseconds
while Timestamp.value is always nanoseconds; comparing the two integer encodings silently dropped
every event-adjacent cutoff."""
from __future__ import annotations

import pandas as pd
import pytest

from verifier import causality


@pytest.mark.parametrize("unit", ["us", "ns", "ms", "s"])
def test_event_adjacent_cutoff_is_generated_for_any_index_resolution(unit, monkeypatch):
    # Disable downsampling so the assertion isolates position generation, not the mode limit.
    monkeypatch.setitem(causality.MODE_LIMITS, "strong", 10_000)
    index = pd.date_range("2024-01-02 14:00", periods=2000, freq="1min", tz="UTC").as_unit(unit)
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=index)
    event_position = 1234
    expected = index[event_position + 1]
    without_events = set(causality.generate_cutoffs(bars, {}, mode="strong", seed=1729))
    assert expected not in without_events  # precondition: only the event can produce this cutoff
    stages = {"signals": pd.DataFrame({"signal_time": [index[event_position]]})}
    with_events = set(causality.generate_cutoffs(bars, stages, mode="strong", seed=1729))
    assert expected in with_events
