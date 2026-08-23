from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    signal_idx: int
    entry_idx: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    signal_z: float
    previous_z: float
    signal_keltner: float
    deadline: pd.Timestamp


@dataclass(frozen=True)
class Trade:
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    signal_z: float
    previous_z: float
    signal_keltner: float
    atr_source_start: pd.Timestamp
    atr_source_end: pd.Timestamp
    atr_value: float
    stop_unrounded: float
    stop_price: float
    target_unrounded: float
    target_price: float
    initial_risk: float
    deadline: pd.Timestamp
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    gross_R: float
    cost_points: float
    cost_R: float
    net_R: float
    entry_year: int
    sample: str
