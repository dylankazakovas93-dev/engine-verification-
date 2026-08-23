from __future__ import annotations

import pandas as pd


def run(bars: pd.DataFrame) -> pd.DataFrame:
    """Map the untouched candidate engine into the canonical trade ledger."""
    raise NotImplementedError("wire the candidate only after specification/oracle freeze")


# Optional: features, signals, eligible_signals, proposed_entries, accepted_entries,
# audit_stages, and reference. See candidate_engines/adapter_template.py.
