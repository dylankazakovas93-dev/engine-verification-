from __future__ import annotations
import pandas as pd

def run(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Invoke the candidate engine and return one row per accepted trade.
    Map the candidate's names into canonical fields.
    Do not change candidate strategy semantics here.
    """
    raise NotImplementedError("wire candidate engine here")
