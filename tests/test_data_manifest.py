from __future__ import annotations

import pandas as pd

from verifier.data import audit_dataset, sha256_file


def test_parquet_manifest_streams_identity_schema_gaps_and_raw_contract_provenance(tmp_path):
    path = tmp_path / "raw_1s.parquet"
    frame = pd.DataFrame({
        "ts_event": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:03Z"]),
        "instrument_id": [1, 1, 1], "symbol": ["NQH6-NQZ5"] * 3,
        "open": [-1.0, -1.25, -1.0], "high": [-0.75, -1.0, -0.5],
        "low": [-1.25, -1.5, -1.25], "close": [-1.0, -1.25, -0.75], "volume": [1, 2, 3],
    })
    frame.to_parquet(path, index=False, row_group_size=2)
    digest, size = sha256_file(path)
    manifest, findings = audit_dataset(
        path, timestamp_col="ts_event", expected_interval="1s",
        contract_col="instrument_id", symbol_col="symbol", allow_nonpositive_prices=True,
        batch_size=2, precomputed_sha256=digest,
    )
    assert manifest.bytes == size
    assert manifest.sha256 == digest
    assert manifest.row_count == 3
    assert manifest.first_timestamp == "2026-01-01T00:00:00+00:00"
    assert manifest.final_timestamp == "2026-01-01T00:00:03+00:00"
    assert manifest.missing_interval_count == 1
    assert manifest.negative_price_rows == 3
    assert manifest.malformed_row_count == 0
    assert "RAW CONTRACT" in manifest.contract_provenance_status
    assert any(f.check == "contract_provenance" and f.status == "UNVERIFIED" for f in findings)


def test_csv_manifest_catches_duplicate_and_bad_geometry(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        "open": [100, 105], "high": [101, 104], "low": [99, 100],
        "close": [100, 103], "volume": [1, 1],
    }).to_csv(path, index=False)
    manifest, findings = audit_dataset(path, timestamp_col="timestamp", expected_interval="1min")
    assert manifest.duplicate_count == 1
    assert manifest.malformed_row_count == 1
    assert any(f.check == "data_duplicates" and f.status == "FAIL" for f in findings)
    assert any(f.check == "data_geometry" and f.status == "FAIL" for f in findings)


def test_malformed_row_count_is_distinct_even_with_multiple_bad_conditions(tmp_path):
    path = tmp_path / "multi_bad.csv"
    pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00Z"],
        "open": [105], "high": [104], "low": [100], "close": [103], "volume": [-1],
    }).to_csv(path, index=False)
    manifest, _ = audit_dataset(path, timestamp_col="timestamp")
    assert manifest.malformed_row_count == 1
