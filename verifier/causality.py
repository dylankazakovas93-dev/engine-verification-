from __future__ import annotations
import numpy as np
import pandas as pd
from .schema import Finding

TIME_KEYS=["signal_time","entry_time","exit_time"]
DEFAULT_KEYS=["signal_time","entry_time","exit_time","entry_price","exit_price","exit_reason","gross_R","cost_R","net_R"]

def _canon(df: pd.DataFrame, cutoff: pd.Timestamp, keys: list[str]) -> pd.DataFrame:
    if df is None or len(df)==0:
        return pd.DataFrame(columns=[c for c in keys if df is None or c in getattr(df,"columns",[])])
    x=df.copy()
    for c in TIME_KEYS:
        if c in x:
            x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
    if "exit_time" in x:
        x=x[x["exit_time"] < cutoff]
    elif "entry_time" in x:
        x=x[x["entry_time"] < cutoff]
    elif "signal_time" in x:
        x=x[x["signal_time"] < cutoff]
    avail=[c for c in keys if c in x]
    return x[avail].reset_index(drop=True)

def _equal(a: pd.DataFrame,b: pd.DataFrame,atol=1e-9)->bool:
    if list(a.columns)!=list(b.columns) or len(a)!=len(b):
        return False
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) or pd.api.types.is_numeric_dtype(b[c]):
            av=pd.to_numeric(a[c],errors="coerce").to_numpy(float)
            bv=pd.to_numeric(b[c],errors="coerce").to_numpy(float)
            if not np.allclose(av,bv,rtol=0,atol=atol,equal_nan=True): return False
        else:
            if not a[c].astype(str).equals(b[c].astype(str)): return False
    return True

def mutate_future(bars: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    x=bars.copy()
    idx=x.index
    mask=idx>=cutoff
    if not mask.any(): return x
    scale=np.linspace(1.7,2.3,int(mask.sum()))
    for c in ["open","close"]:
        if c in x: x.loc[mask,c]=x.loc[mask,c].to_numpy(float)*scale
    if {"open","close","high","low"}.issubset(x.columns):
        o=x.loc[mask,"open"].to_numpy(float); c=x.loc[mask,"close"].to_numpy(float)
        hi=np.maximum(o,c)*1.03
        lo=np.minimum(o,c)*0.97
        x.loc[mask,"high"]=hi; x.loc[mask,"low"]=lo
    return x

def audit_causality(run_fn, bars: pd.DataFrame, cutoffs: list[pd.Timestamp] | None=None) -> list[Finding]:
    if not isinstance(bars.index,pd.DatetimeIndex):
        return [Finding("causality","FAIL","bars must use DatetimeIndex for causality audit")]
    if bars.index.tz is None:
        return [Finding("causality","FAIL","bars index must be timezone-aware")]
    n=len(bars)
    if n<300:
        return [Finding("causality","WARN","dataset too short for robust truncation audit")]
    if cutoffs is None:
        pts=sorted(set([max(50,n//4),max(100,n//2),max(150,3*n//4)]))
        cutoffs=[bars.index[i] for i in pts if i<n]
    full=run_fn(bars)
    findings=[]
    for cutoff in cutoffs:
        prefix=bars[bars.index < cutoff]
        p=run_fn(prefix)
        a=_canon(full,cutoff,DEFAULT_KEYS); b=_canon(p,cutoff,DEFAULT_KEYS)
        eq1=_equal(a,b)
        findings.append(Finding("truncation_invariance","PASS" if eq1 else "FAIL",
                                f"cutoff={cutoff}: pre-cutoff completed ledger {'unchanged' if eq1 else 'changed'}"))
        m=mutate_future(bars,cutoff)
        fm=run_fn(m)
        c=_canon(fm,cutoff,DEFAULT_KEYS)
        eq2=_equal(a,c)
        findings.append(Finding("future_mutation","PASS" if eq2 else "FAIL",
                                f"cutoff={cutoff}: mutating future OHLC {'did not change' if eq2 else 'changed'} completed pre-cutoff ledger"))
    return findings
