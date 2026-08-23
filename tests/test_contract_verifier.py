from __future__ import annotations
import pandas as pd
from verifier.contracts import audit_selection_map

def test_roll_selection_map_catches_wrong_contract():
    selected=pd.DataFrame({
        "timestamp":["2026-01-01T00:00:00Z","2026-01-01T00:01:00Z"],
        "contract_id":["NQH26","NQH26"],
    })
    expected=pd.DataFrame({
        "timestamp":["2026-01-01T00:00:00Z","2026-01-01T00:01:00Z"],
        "expected_contract_id":["NQH26","NQM26"],
    })
    f=audit_selection_map(selected,expected)
    assert f[0].status=="FAIL"
