from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .schema import Finding


@dataclass
class DataManifest:
    path: str
    filename: str
    bytes: int
    sha256: str
    format: str
    row_count: int
    first_timestamp: str | None
    final_timestamp: str | None
    timestamp_column: str
    timezone_interpretation: str
    schema: dict[str, str]
    chronological: bool
    duplicate_count: int
    malformed_row_count: int
    nonfinite_cell_count: int
    missing_interval_count: int | None
    missing_interval_samples: list[dict[str, Any]] = field(default_factory=list)
    contract_provenance_status: str = "ROLL PROVENANCE UNVERIFIED"
    contract_columns: list[str] = field(default_factory=list)
    negative_price_rows: int = 0
    zero_volume_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
        return target


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> tuple[str, int]:
    p = Path(path)
    digest = hashlib.sha256()
    size = 0
    with p.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _iter_csv(path: Path, columns: list[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, usecols=columns, chunksize=batch_size)


def _parquet_metadata(path: Path) -> tuple[Any, dict[str, str], int, str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet audit requires pyarrow; install the project dependencies") from exc
    pf = pq.ParquetFile(path)
    schema = {field.name: str(field.type) for field in pf.schema_arrow}
    return pf, schema, int(pf.metadata.num_rows), str(pf.schema_arrow)


def _iter_parquet(pf: Any, columns: list[str], batch_size: int) -> Iterator[pd.DataFrame]:
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns, use_threads=True):
        yield batch.to_pandas()


def _timestamp_timezone(dtype_text: str) -> str:
    lower = dtype_text.lower()
    if "tz=utc" in lower or ", utc" in lower or "datetime64[ns, utc]" in lower:
        return "UTC (explicit in source schema)"
    if "timestamp" in lower or "datetime" in lower:
        return "timezone not explicit in source schema; parsing must prove awareness"
    return "not a timestamp type"


def audit_dataset(
    path: str | Path,
    *,
    timestamp_col: str,
    expected_interval: str | pd.Timedelta | None = None,
    contract_col: str | None = None,
    symbol_col: str | None = None,
    allow_nonpositive_prices: bool = False,
    batch_size: int = 250_000,
    precomputed_sha256: str | None = None,
) -> tuple[DataManifest, list[Finding]]:
    """Stream an immutable CSV/Parquet source and return exact audit counters."""
    p = Path(path).resolve()
    suffix = p.suffix.lower()
    digest, size = (precomputed_sha256, p.stat().st_size) if precomputed_sha256 else sha256_file(p)
    if suffix in {".parquet", ".pq"}:
        pf, schema, metadata_rows, schema_text = _parquet_metadata(p)
        available = set(schema)
        iterator_factory = lambda cols: _iter_parquet(pf, cols, batch_size)
        source_rows = metadata_rows
        source_format = "parquet"
    elif suffix in {".csv", ".txt"}:
        header = pd.read_csv(p, nrows=0)
        schema = {name: str(dtype) for name, dtype in header.dtypes.items()}
        available = set(header.columns)
        iterator_factory = lambda cols: _iter_csv(p, cols, batch_size)
        source_rows = None
        source_format = "csv"
        schema_text = " ".join(f"{k}:{v}" for k, v in schema.items())
    else:
        raise ValueError(f"unsupported data format: {p.suffix}")

    required = {timestamp_col, "open", "high", "low", "close"}
    missing_columns = sorted(required - available)
    if missing_columns:
        raise ValueError(f"dataset missing required columns: {missing_columns}")
    selected = [timestamp_col, "open", "high", "low", "close"]
    for optional in ("volume", contract_col, symbol_col):
        if optional and optional in available and optional not in selected:
            selected.append(optional)

    interval = pd.Timedelta(expected_interval) if expected_interval is not None else None
    rows = duplicates = malformed = nonfinite = negative_rows = zero_volume = 0
    chronological = True
    first_ts: pd.Timestamp | None = None
    final_ts: pd.Timestamp | None = None
    last_global: pd.Timestamp | None = None
    last_by_contract: dict[str, pd.Timestamp] = {}
    trailing_keys: set[tuple[int, str]] = set()
    missing_count = 0
    missing_samples: list[dict[str, Any]] = []
    timezone_aware = True
    symbol_by_instrument: dict[str, str] = {}
    symbol_conflicts = 0

    for chunk in iterator_factory(selected):
        if chunk.empty:
            continue
        parsed = pd.to_datetime(chunk[timestamp_col], errors="coerce", utc=False)
        bad_timestamp = parsed.isna().to_numpy()
        try:
            index = pd.DatetimeIndex(parsed)
            aware = index.tz is not None
        except Exception:
            index = pd.DatetimeIndex(pd.to_datetime(chunk[timestamp_col], errors="coerce", utc=True))
            aware = False
        timezone_aware &= aware
        if aware:
            index = index.tz_convert("UTC")
        else:
            index = pd.DatetimeIndex(pd.to_datetime(chunk[timestamp_col], errors="coerce", utc=True))
        valid_time = ~index.isna()
        if valid_time.any():
            valid_index = index[valid_time]
            if first_ts is None:
                first_ts = valid_index[0]
            if last_global is not None and valid_index[0] < last_global:
                chronological = False
            if not valid_index.is_monotonic_increasing:
                chronological = False
            last_global = valid_index[-1]
            final_ts = valid_index[-1]

        numeric = chunk[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(float)
        bad_cells = ~np.isfinite(values)
        nonfinite += int(bad_cells.sum())
        geom = (
            (numeric["high"] < numeric["low"])
            | (numeric["open"] < numeric["low"]) | (numeric["open"] > numeric["high"])
            | (numeric["close"] < numeric["low"]) | (numeric["close"] > numeric["high"])
        ).to_numpy()
        negative = (numeric <= 0).any(axis=1).to_numpy()
        negative_rows += int(negative.sum())
        invalid_price = np.zeros(len(chunk), dtype=bool) if allow_nonpositive_prices else negative
        invalid_rows = bad_timestamp | bad_cells.any(axis=1) | geom | invalid_price
        if "volume" in chunk:
            volume = pd.to_numeric(chunk["volume"], errors="coerce").to_numpy(float)
            nonfinite += int((~np.isfinite(volume)).sum())
            invalid_rows |= (~np.isfinite(volume)) | (volume < 0)
            zero_volume += int((volume == 0).sum())
        malformed += int(invalid_rows.sum())

        labels = chunk[contract_col].astype(str).to_numpy() if contract_col and contract_col in chunk else np.repeat("__series__", len(chunk))
        times_ns = index.asi8
        key_frame = pd.DataFrame({"ts": times_ns, "contract": labels})
        duplicates += int(key_frame.duplicated(["ts", "contract"], keep="first").sum())
        if trailing_keys:
            duplicates += sum((int(ts), str(label)) in trailing_keys for ts, label in zip(times_ns, labels))
        if len(times_ns):
            max_ts = int(np.max(times_ns))
            trailing_keys = {(int(ts), str(label)) for ts, label in zip(times_ns, labels) if int(ts) == max_ts}

        if interval is not None:
            work = pd.DataFrame({"ts": index, "contract": labels}).dropna(subset=["ts"])
            for label, group in work.groupby("contract", sort=False):
                series = pd.DatetimeIndex(group["ts"])
                previous = last_by_contract.get(str(label))
                deltas = series.to_series(index=np.arange(len(series))).diff()
                if previous is not None and len(series):
                    cross_delta = series[0] - previous
                    if cross_delta > interval:
                        gaps = max(int(cross_delta // interval) - 1, 0)
                        missing_count += gaps
                        if len(missing_samples) < 20:
                            missing_samples.append({"contract": str(label), "after": previous.isoformat(), "before": series[0].isoformat(), "missing_intervals": gaps})
                large = deltas > interval
                for i in np.flatnonzero(large.to_numpy()):
                    delta = deltas.iloc[i]
                    gaps = max(int(delta // interval) - 1, 0)
                    missing_count += gaps
                    if len(missing_samples) < 20:
                        missing_samples.append({"contract": str(label), "after": series[i - 1].isoformat(), "before": series[i].isoformat(), "missing_intervals": gaps})
                if len(series):
                    last_by_contract[str(label)] = series[-1]

        if contract_col and symbol_col and contract_col in chunk and symbol_col in chunk:
            pairs = chunk[[contract_col, symbol_col]].drop_duplicates()
            for instrument, symbol in pairs.itertuples(index=False, name=None):
                key, value = str(instrument), str(symbol)
                prior = symbol_by_instrument.setdefault(key, value)
                if prior != value:
                    symbol_conflicts += 1
        rows += len(chunk)

    provenance_columns = [c for c in (contract_col, symbol_col) if c and c in available]
    if contract_col and symbol_col and contract_col in available and symbol_col in available:
        provenance = "RAW CONTRACT IDENTIFIERS/SYMBOLS PRESENT; CONTINUOUS ROLL SELECTION UNVERIFIED WITHOUT A FROZEN SELECTION POLICY/MAP"
    elif contract_col and contract_col in available:
        provenance = "CONTRACT IDENTIFIERS PRESENT; ROLL POLICY CORRECTNESS UNVERIFIED"
    else:
        provenance = "ROLL PROVENANCE UNVERIFIED"
    notes = []
    if allow_nonpositive_prices and negative_rows:
        notes.append("Nonpositive prices were classified, not treated as malformed; raw spread instruments can be negative.")
    if symbol_conflicts:
        notes.append(f"{symbol_conflicts} instrument-to-symbol conflicts observed across streamed batches.")
    if source_rows is not None and rows != source_rows:
        notes.append(f"Streamed row count {rows} differs from file metadata row count {source_rows}.")
    timezone_text = _timestamp_timezone(schema.get(timestamp_col, schema_text))
    if not timezone_aware:
        timezone_text += "; observed values were not uniformly timezone-aware"
    manifest = DataManifest(
        path=str(p), filename=p.name, bytes=int(size), sha256=str(digest), format=source_format,
        row_count=rows, first_timestamp=first_ts.isoformat() if first_ts is not None else None,
        final_timestamp=final_ts.isoformat() if final_ts is not None else None,
        timestamp_column=timestamp_col, timezone_interpretation=timezone_text, schema=schema,
        chronological=chronological, duplicate_count=duplicates, malformed_row_count=malformed,
        nonfinite_cell_count=nonfinite, missing_interval_count=missing_count if interval is not None else None,
        missing_interval_samples=missing_samples, contract_provenance_status=provenance,
        contract_columns=provenance_columns, negative_price_rows=negative_rows, zero_volume_rows=zero_volume,
        notes=notes,
    )
    findings = [
        Finding("data_identity", "PASS", f"SHA-256={digest}; bytes={size}; rows={rows}", family="data_audit"),
        Finding("data_timezone", "PASS" if timezone_aware else "FAIL", timezone_text, family="data_audit"),
        Finding("data_ordering", "PASS" if chronological else "FAIL", "chronological" if chronological else "out-of-order rows detected", family="data_audit"),
        Finding("data_duplicates", "PASS" if duplicates == 0 else "FAIL", f"{duplicates} duplicate timestamp/contract rows", family="data_audit"),
        Finding("data_geometry", "PASS" if malformed == 0 else "FAIL", f"{malformed} malformed rows; {nonfinite} nonfinite cells", family="data_audit"),
        Finding("contract_provenance", "UNVERIFIED" if "UNVERIFIED" in provenance else "PASS", provenance, family="contract_provenance"),
    ]
    if interval is not None:
        findings.append(Finding("missing_intervals", "INFO", f"{missing_count} absent expected intervals; never filled", family="data_audit", evidence={"samples": missing_samples}))
    return manifest, findings
