from __future__ import annotations

import pandas as pd

import nq_frozen.core as core
from nq_frozen.models import Candidate, Trade


def test_equal_timestamp_previous_exit_rejects_new_entry(monkeypatch, make_bars):
    bars = make_bars([
        ("2024-01-02 13:00+00:00",100,101,99,100),
        ("2024-01-02 13:01+00:00",100,101,99,100),
        ("2024-01-02 13:02+00:00",100,101,99,100),
        ("2024-01-02 13:03+00:00",100,101,99,100),
    ])
    features = bars.copy()
    features["signal_passes_keltner"] = False
    c1 = Candidate(0,1,bars.index[0],bars.index[1],5.1,4.9,1.0,bars.index[3]+pd.Timedelta(hours=7))
    c2 = Candidate(1,2,bars.index[1],bars.index[2],5.1,4.9,1.0,bars.index[3]+pd.Timedelta(hours=7))

    monkeypatch.setattr(core, "build_features", lambda x: features)
    monkeypatch.setattr(core, "build_candidates", lambda x: [c1,c2])
    atr_table = pd.DataFrame([{
        "source_start": bars.index[0]-pd.Timedelta(hours=4),
        "source_end": bars.index[0], "tr": 10.0, "available_at": bars.index[0]
    }])
    monkeypatch.setattr(core, "build_4h_true_range", lambda x: atr_table)
    monkeypatch.setattr(core, "atr_for_entry", lambda table, ts: pd.Series(table.iloc[0]))

    calls = []
    def fake_sim(b, c, a):
        calls.append(c.entry_time)
        exit_time = bars.index[2] if len(calls)==1 else bars.index[3]
        return Trade(c.signal_time,c.entry_time,100,5.1,4.9,1.0,a["source_start"],a["source_end"],10,90,90,105,105,10,c.deadline,exit_time,105,"TP",0.5,1,0.1,0.4,2024,"IS")
    monkeypatch.setattr(core, "simulate_trade", fake_sim)

    trades, _ = core.run_backtest(bars)
    assert len(trades) == 1
    assert calls == [bars.index[1]]
