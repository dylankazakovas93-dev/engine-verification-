from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .core import run_backtest
from .reference import run_reference

KEY_COLUMNS = [
    "signal_time", "entry_time", "entry_price", "atr_source_start", "atr_source_end",
    "atr_value", "stop_price", "target_price", "initial_risk", "deadline",
    "exit_time", "exit_price", "exit_reason", "gross_R", "cost_R", "net_R",
]


def ledger_hash(trades: pd.DataFrame) -> str:
    # Hash only reconciled strategy/execution fields, not floating diagnostic columns
    # such as signal_z whose independent summation path may differ by machine epsilon.
    canonical = trades.loc[:, KEY_COLUMNS].copy()
    for col in canonical.columns:
        if pd.api.types.is_float_dtype(canonical[col]):
            canonical[col] = canonical[col].round(9)
    payload = canonical.to_csv(index=False, lineterminator="\n", float_format="%.9f")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reconcile(bars: pd.DataFrame, atol: float = 1e-10) -> dict:
    prod, prod_f = run_backtest(bars)
    ref, ref_f = run_reference(bars)
    if len(prod) != len(ref):
        raise AssertionError(f"trade count mismatch: production={len(prod)} reference={len(ref)}")

    time_cols = {"signal_time", "entry_time", "atr_source_start", "atr_source_end", "deadline", "exit_time"}
    str_cols = {"exit_reason"}
    for col in KEY_COLUMNS:
        a = prod[col].reset_index(drop=True)
        b = ref[col].reset_index(drop=True)
        if col in time_cols or col in str_cols:
            if not a.equals(b):
                mismatch = np.flatnonzero((a != b).to_numpy())[:10].tolist()
                raise AssertionError(f"ledger mismatch in {col} at rows {mismatch}")
        else:
            if not np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=0, atol=atol, equal_nan=True):
                mismatch = np.flatnonzero(~np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=0, atol=atol, equal_nan=True))[:10].tolist()
                raise AssertionError(f"ledger mismatch in {col} at rows {mismatch}")

    # Feature reconciliation is tolerance-based because independent rolling summation paths
    # may differ by machine epsilon. Signal booleans must match exactly.
    for col in ["fracdiff", "ema60", "mean_range60", "keltner"]:
        if not np.allclose(prod_f[col].to_numpy(dtype=float), ref_f[col].to_numpy(dtype=float), rtol=0, atol=1e-10, equal_nan=True):
            raise AssertionError(f"feature mismatch in {col}")
    # Rolling variance can amplify machine-epsilon summation differences when the
    # 100-observation window has extremely low variance. Numerical z values therefore
    # reconcile to 1e-6, while the actual threshold/crossing booleans below must match exactly.
    if not np.allclose(prod_f["z"].to_numpy(dtype=float), ref_f["z"].to_numpy(dtype=float), rtol=0, atol=1e-6, equal_nan=True):
        raise AssertionError("feature mismatch in z")
    for col in ["raw_long_signal", "signal_passes_keltner"]:
        if not prod_f[col].equals(ref_f[col]):
            raise AssertionError(f"signal mismatch in {col}")

    return {
        "production_trades": len(prod),
        "reference_trades": len(ref),
        "production_hash": ledger_hash(prod),
        "reference_hash": ledger_hash(ref),
        "matched": True,
    }
