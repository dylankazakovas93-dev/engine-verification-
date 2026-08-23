from __future__ import annotations

import pandas as pd

from .core import validate_bars


def load_csv(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp column {timestamp_col!r} not found")
    parsed = pd.to_datetime(df[timestamp_col], errors="raise")
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            "CSV timestamps must carry explicit timezone/UTC information; naive timestamps are rejected"
        )
    parsed = parsed.dt.tz_convert("UTC")
    df = df.drop(columns=[timestamp_col])
    df.index = pd.DatetimeIndex(parsed)
    return validate_bars(df)
