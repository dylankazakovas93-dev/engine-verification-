"""Minimal CLEAN research adapter demonstrating the research verification contract.

This is a verifier fixture and a starting template, NOT a research engine. Its toy ridge model
and toy features exist only so the verifier can be shown to pass causal code. Do not treat any
output of this file as evidence of edge.

Contract (see docs/RESEARCH_VERIFICATION.md):
    events(bars) -> event_id, event_time, direction
    features(bars, events) -> event_id, feature_asof_time, <features>
    targets(bars, events) -> event_id, target_start, target_end, <labels>
    fit_predict_fold(features, targets, train_ids, validation_ids, target_name) -> event_id, prediction
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 10
TARGET = "forward_return_10"
FEATURE_COLUMNS = ["ret_5", "vol_20", "dist_ma_20"]


def events(bars: pd.DataFrame) -> pd.DataFrame:
    """Every third bar after a 20-bar warm-up; knowable at the bar's own timestamp."""
    position = np.arange(len(bars))
    mask = (position >= 20) & (position % 3 == 0)
    close = bars["close"].to_numpy(float)
    moving_average = bars["close"].rolling(20).mean().to_numpy(float)
    times = bars.index[mask]
    return pd.DataFrame({
        "event_id": [f"E{t:%Y%m%dT%H%M%S}" for t in times],
        "event_time": pd.Series(times),
        "direction": np.where(close[mask] >= moving_average[mask], "long", "short"),
    })


def features(bars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Trailing-only features, as of the event bar."""
    close = bars["close"]
    table = pd.DataFrame({
        "ret_5": close / close.shift(5) - 1.0,
        "vol_20": close.pct_change().rolling(20).std(),
        "dist_ma_20": close / close.rolling(20).mean() - 1.0,
    }, index=bars.index)
    rows = table.reindex(pd.DatetimeIndex(events["event_time"]))
    out = pd.DataFrame({"event_id": events["event_id"].to_numpy(), "feature_asof_time": events["event_time"].reset_index(drop=True)})
    for column in FEATURE_COLUMNS:
        out[column] = rows[column].to_numpy()
    return out


def targets(bars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Forward return over HORIZON bars. Events whose window is unresolved are omitted."""
    close = bars["close"].to_numpy(float)
    position = bars.index.get_indexer(pd.DatetimeIndex(events["event_time"]))
    resolved = position + HORIZON < len(bars)
    p = position[resolved]
    end = p + HORIZON
    return pd.DataFrame({
        "event_id": events["event_id"].to_numpy()[resolved],
        "target_start": pd.Series(bars.index[p + 1] if len(p) else pd.DatetimeIndex([], tz=bars.index.tz)),
        "target_end": pd.Series(bars.index[end] if len(p) else pd.DatetimeIndex([], tz=bars.index.tz)),
        TARGET: close[end] / close[p] - 1.0 if len(p) else np.array([], dtype=float),
    })


def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
    """Ridge regression; every statistic is computed from train_ids rows only."""
    table = features.set_index("event_id")[FEATURE_COLUMNS]
    labels = targets.set_index("event_id")[target_name]
    train_x = table.loc[train_ids].to_numpy(float)
    train_y = labels.loc[train_ids].to_numpy(float)
    usable = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
    train_x, train_y = train_x[usable], train_y[usable]
    center, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(train_x)), (train_x - center) / scale])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    validation_x = np.nan_to_num((table.loc[validation_ids].to_numpy(float) - center) / scale)
    prediction = np.column_stack([np.ones(len(validation_x)), validation_x]) @ beta
    return pd.DataFrame({"event_id": list(validation_ids), "prediction": prediction})
