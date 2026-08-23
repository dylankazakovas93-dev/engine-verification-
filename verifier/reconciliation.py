from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Finding

DEFAULT_RECONCILIATION_COLUMNS = [
    "signal_time", "entry_time", "direction", "entry_price", "atr_source_start",
    "atr_source_end", "atr_value", "stop_price", "target_price", "deadline",
    "exit_time", "exit_price", "exit_reason", "gross_R", "cost_R", "net_R",
]


def reconcile_ledgers(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    atol: float = 1e-9,
) -> list[Finding]:
    requested = columns or DEFAULT_RECONCILIATION_COLUMNS
    comparable = [c for c in requested if c in candidate.columns and c in reference.columns]
    missing_candidate = [c for c in requested if c in reference.columns and c not in candidate.columns]
    if missing_candidate:
        return [Finding("reference_reconciliation", "FAIL", f"candidate missing reference columns: {missing_candidate}", family="reference_reconciliation")]
    if len(candidate) != len(reference):
        return [Finding("reference_reconciliation", "FAIL", f"trade count mismatch: candidate={len(candidate)} reference={len(reference)}", family="reference_reconciliation")]
    for column in comparable:
        left = candidate[column].reset_index(drop=True)
        right = reference[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            a = pd.to_numeric(left, errors="coerce").to_numpy(float)
            b = pd.to_numeric(right, errors="coerce").to_numpy(float)
            equal = np.isclose(a, b, rtol=0, atol=atol, equal_nan=True)
        else:
            if "time" in column or column == "deadline":
                a = pd.to_datetime(left, utc=True, errors="coerce").astype(str).to_numpy()
                b = pd.to_datetime(right, utc=True, errors="coerce").astype(str).to_numpy()
            else:
                a, b = left.astype(str).to_numpy(), right.astype(str).to_numpy()
            equal = a == b
        if not bool(equal.all()):
            rows = np.flatnonzero(~equal)[:20].tolist()
            return [Finding(
                "reference_reconciliation", "FAIL", f"first mismatch in {column!r} at rows {rows}",
                rows=rows, family="reference_reconciliation", classification="PROVEN FAILURE",
            )]
    return [Finding(
        "reference_reconciliation", "PASS",
        f"{len(candidate)} trades reconciled across {len(comparable)} fields",
        family="reference_reconciliation",
    )]
