from __future__ import annotations
import pandas as pd

def run(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Invoke the candidate engine and return one row per accepted trade.
    Map the candidate's names into canonical fields.
    Do not change candidate strategy semantics here.
    """
    raise NotImplementedError("wire candidate engine here")

# Optional stage interfaces. Expose as many as the candidate genuinely supports:
# features(bars) -> time-indexed feature/indicator DataFrame
# signals(bars) -> DataFrame with signal_time
# eligible_signals(bars) -> DataFrame with signal_time
# proposed_entries(bars) -> DataFrame with signal_time and entry_time
# accepted_entries(bars) -> DataFrame with entry_time
# audit_stages(bars) -> dict containing any/all of the above plus "trades"
# If features are absent, the verifier must report FEATURE-LEVEL CAUSALITY UNVERIFIED.
