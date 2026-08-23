from __future__ import annotations

import numpy as np
import pandas as pd

from verifier.causality import audit_adapter_causality


def bars(n=420):
    index = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    close = 100.0 + np.arange(n) * 0.01 + np.sin(np.arange(n) / 9)
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.ones(n),
    }, index=index)


class GoodStages:
    @staticmethod
    def audit_stages(frame):
        feature = frame[["close"]].copy()
        feature["causal_mean"] = frame["close"].expanding().mean()
        positions = np.arange(20, len(frame), 40)
        signal_times = frame.index[positions]
        signals = pd.DataFrame({"signal_time": signal_times, "score": frame.iloc[positions]["close"].to_numpy()})
        eligible = signals[signals["score"] > 0].copy()
        valid_positions = positions[positions + 1 < len(frame)]
        proposed = pd.DataFrame({"signal_time": frame.index[valid_positions], "entry_time": frame.index[valid_positions + 1]})
        accepted = proposed.copy()
        complete = valid_positions[valid_positions + 2 < len(frame)]
        trades = pd.DataFrame({
            "signal_time": frame.index[complete], "entry_time": frame.index[complete + 1],
            "exit_time": frame.index[complete + 2], "entry_price": frame.iloc[complete + 1]["open"].to_numpy(),
            "exit_price": frame.iloc[complete + 2]["close"].to_numpy(), "exit_reason": "TEST",
        })
        return {"features": feature, "signals": signals, "eligible_signals": eligible,
                "proposed_entries": proposed, "accepted_entries": accepted, "trades": trades}

    @staticmethod
    def run(frame):
        return GoodStages.audit_stages(frame)["trades"]


class FutureNormalized:
    @staticmethod
    def audit_stages(frame):
        features = pd.DataFrame({"normalized": frame["close"] / frame["close"].mean()}, index=frame.index)
        trades = pd.DataFrame(columns=["entry_time", "exit_time", "entry_price", "exit_price"])
        return {"features": features, "trades": trades}

    run = staticmethod(lambda frame: FutureNormalized.audit_stages(frame)["trades"])


class OneBarLookahead:
    @staticmethod
    def audit_stages(frame):
        # Deliberately illegal: the current row signals only if a next row exists.
        signal = frame["close"].shift(-1).notna()
        signals = pd.DataFrame({"signal_time": frame.index[signal.fillna(False)]})
        trades = pd.DataFrame(columns=["entry_time", "exit_time", "entry_price", "exit_price"])
        return {"signals": signals, "trades": trades}

    run = staticmethod(lambda frame: OneBarLookahead.audit_stages(frame)["trades"])


def test_good_adapter_reconciles_all_exposed_stages_at_many_cutoffs():
    findings, coverage = audit_adapter_causality(GoodStages, bars(), mode="standard", seed=123)
    assert coverage.to_dict()["cutoff_count"] > 3
    assert set(coverage.stages) == {"features", "signals", "eligible_signals", "proposed_entries", "accepted_entries", "trades"}
    assert not [f for f in findings if f.status == "FAIL"]


def test_future_normalization_is_caught_at_feature_level():
    findings, coverage = audit_adapter_causality(FutureNormalized, bars(), mode="fast")
    assert "features" in coverage.stages
    assert any(f.status == "FAIL" and f.family == "feature_causality" for f in findings)


def test_one_bar_lookahead_is_caught_at_signal_level():
    findings, _ = audit_adapter_causality(OneBarLookahead, bars(), mode="fast")
    assert any(f.status == "FAIL" and "signals" in f.check for f in findings)
