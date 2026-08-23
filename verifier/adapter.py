from __future__ import annotations
import importlib.util
from pathlib import Path

def load_adapter(path: str):
    p=Path(path).resolve()
    spec=importlib.util.spec_from_file_location("candidate_adapter",p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load adapter {p}")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod,"run",None)):
        raise TypeError("adapter must expose run(bars) -> DataFrame")
    return mod
