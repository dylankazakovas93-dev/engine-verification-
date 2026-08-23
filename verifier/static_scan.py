from __future__ import annotations
from pathlib import Path
import re
from .schema import Finding

PATTERNS = [
    ("negative_shift", r"\.shift\s*\(\s*-\d+", "HIGH", "negative shift can import future values"),
    ("centered_rolling", r"rolling\s*\([^)]*center\s*=\s*True", "HIGH", "centered rolling window uses future observations"),
    ("backfill", r"\.(?:bfill|backfill)\s*\(", "HIGH", "backfill can move future information backward"),
    ("fillna_bfill", r"fillna\s*\([^)]*(?:bfill|backfill)", "HIGH", "backfill inside fillna can leak future values"),
    ("forward_asof", r"merge_asof\s*\([^)]*direction\s*=\s*[\"']forward[\"']", "HIGH", "forward merge_asof selects future rows"),
    ("negative_roll", r"np\.roll\s*\([^,]+,\s*-\d+", "HIGH", "negative np.roll can expose future observations"),
    ("future_slice", r"\[[A-Za-z_]\w*\s*:\s*\]", "MEDIUM", "open-ended slice from current index may be future-looking; inspect manually"),
    ("silent_sort", r"\.sort_(?:index|values)\s*\(", "MEDIUM", "sorting inside engine can conceal corrupted input rather than reject it"),
    ("drop_duplicates", r"\.drop_duplicates\s*\(", "MEDIUM", "dropping duplicates can conceal upstream corruption"),
    ("interpolate", r"\.interpolate\s*\(", "MEDIUM", "interpolation may synthesize missing market observations"),
]

def scan_source(path: str | Path) -> list[Finding]:
    p=Path(path)
    if not p.exists():
        return [Finding("static_source","FAIL",f"source path not found: {p}")]
    paths=[p] if p.is_file() else list(p.rglob("*.py"))
    out=[]
    for fp in paths:
        try: text=fp.read_text(encoding="utf-8",errors="replace")
        except Exception as e:
            out.append(Finding("static_source","WARN",f"could not read {fp}: {e}")); continue
        for name,pat,severity,msg in PATTERNS:
            hits=list(re.finditer(pat,text,re.I|re.S))
            if hits:
                lines=[]
                for m in hits[:10]:
                    lines.append(text.count("\n",0,m.start())+1)
                out.append(Finding(f"static_{name}","WARN",f"{severity}: {fp}: lines {lines}: {msg}"))
    if not out:
        out.append(Finding("static_scan","PASS","no configured high-risk source patterns found"))
    return out
