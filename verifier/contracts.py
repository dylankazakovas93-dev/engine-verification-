from __future__ import annotations
import pandas as pd
from .schema import Finding

def audit_selection_map(
    selected: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    timestamp_col="timestamp",
    contract_col="contract_id",
    expected_contract_col="expected_contract_id",
) -> list[Finding]:
    for df,name,col in [(selected,"selected",contract_col),(expected,"expected",expected_contract_col)]:
        if timestamp_col not in df or col not in df:
            return [Finding("roll_selection","FAIL",f"{name} data missing {timestamp_col!r} or {col!r}")]
    a=selected[[timestamp_col,contract_col]].copy()
    b=expected[[timestamp_col,expected_contract_col]].copy()
    a[timestamp_col]=pd.to_datetime(a[timestamp_col],utc=True,errors="coerce")
    b[timestamp_col]=pd.to_datetime(b[timestamp_col],utc=True,errors="coerce")
    m=a.merge(b,on=timestamp_col,how="inner")
    if m.empty:
        return [Finding("roll_selection","FAIL","no overlapping timestamps between continuous selection and expected selection map")]
    bad=m[contract_col].astype(str)!=m[expected_contract_col].astype(str)
    return [Finding("roll_selection","FAIL" if bad.any() else "PASS",
                    f"{int(bad.sum())}/{len(m)} overlapping rows use a contract different from the upstream selection oracle")]
