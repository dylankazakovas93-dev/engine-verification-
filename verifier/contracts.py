from __future__ import annotations
from collections.abc import Callable
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
    missing_expected = b.loc[~b[timestamp_col].isin(a[timestamp_col])]
    duplicate_selected = a.duplicated(timestamp_col, keep=False)
    findings = [
        Finding("roll_selection", "FAIL" if bad.any() else "PASS",
                f"{int(bad.sum())}/{len(m)} overlapping rows use a contract different from the upstream selection oracle",
                family="contract_provenance"),
        Finding("roll_selection_coverage", "FAIL" if len(missing_expected) else "PASS",
                f"{len(missing_expected)} expected timestamps lack a selected contract",
                family="contract_provenance"),
        Finding("roll_duplicate_selection", "FAIL" if duplicate_selected.any() else "PASS",
                f"{int(duplicate_selected.sum())} selected rows share a timestamp",
                family="contract_provenance"),
    ]
    return findings


def audit_contract_timeline(
    selected: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    contract_col: str = "contract_id",
    permitted_switch: Callable[[pd.Timestamp, str, str], bool] | None = None,
    expiry_by_contract: dict[str, pd.Timestamp] | None = None,
) -> list[Finding]:
    """Audit an already-selected continuous timeline without pretending it proves the selector."""
    if timestamp_col not in selected or contract_col not in selected:
        return [Finding(
            "contract_provenance", "UNVERIFIED",
            "ROLL PROVENANCE UNVERIFIED: selected data has no timestamp/contract identifier timeline",
            family="contract_provenance",
        )]
    work = selected[[timestamp_col, contract_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    missing = work[timestamp_col].isna() | work[contract_col].isna()
    duplicate = work.duplicated(timestamp_col, keep=False)
    work = work.dropna().sort_values(timestamp_col, kind="stable").reset_index(drop=True)
    switched = work[contract_col].astype(str).ne(work[contract_col].astype(str).shift(1))
    switch_rows = work.loc[switched & work.index.to_series().gt(0)]
    unexpected: list[int] = []
    if permitted_switch is not None:
        for i in switch_rows.index:
            if not permitted_switch(work.at[i, timestamp_col], str(work.at[i - 1, contract_col]), str(work.at[i, contract_col])):
                unexpected.append(int(i))
    expired: list[int] = []
    if expiry_by_contract:
        for i, row in work.iterrows():
            expiry = expiry_by_contract.get(str(row[contract_col]))
            if expiry is not None and row[timestamp_col] >= pd.Timestamp(expiry).tz_convert("UTC"):
                expired.append(int(i))
    findings = [
        Finding("contract_timeline_missing", "FAIL" if missing.any() else "PASS", f"{int(missing.sum())} rows lack timestamp/contract ID", family="contract_provenance"),
        Finding("contract_timeline_duplicates", "FAIL" if duplicate.any() else "PASS", f"{int(duplicate.sum())} duplicate selected timestamps", family="contract_provenance"),
        Finding("contract_switches", "INFO", f"{len(switch_rows)} contract switches observed", family="contract_provenance", evidence={"timestamps": [x.isoformat() for x in switch_rows[timestamp_col].head(20)]}),
    ]
    if permitted_switch is None:
        findings.append(Finding("roll_policy", "UNVERIFIED", "switch timeline exists but no frozen rollover policy was supplied", family="contract_provenance"))
    else:
        findings.append(Finding("roll_policy", "FAIL" if unexpected else "PASS", f"{len(unexpected)} switches violate the supplied roll policy", rows=unexpected[:20], family="contract_provenance"))
    if expiry_by_contract is not None:
        findings.append(Finding("expired_contract", "FAIL" if expired else "PASS", f"{len(expired)} rows select an expired/ineligible contract", rows=expired[:20], family="contract_provenance"))
    return findings


def compare_independent_selector(
    raw_contract_bars: pd.DataFrame,
    selected: pd.DataFrame,
    selector: Callable[[pd.DataFrame], pd.DataFrame] | None,
    **selection_map_kwargs,
) -> list[Finding]:
    """Run only an explicitly supplied independent selector; never invent a roll policy."""
    if selector is None:
        return [Finding("independent_roll_selector", "UNVERIFIED", "no explicit independent roll selector/policy supplied", family="contract_provenance")]
    expected = selector(raw_contract_bars.copy())
    if not isinstance(expected, pd.DataFrame):
        return [Finding("independent_roll_selector", "FAIL", "independent selector did not return a DataFrame", family="contract_provenance")]
    return audit_selection_map(selected, expected, **selection_map_kwargs)
