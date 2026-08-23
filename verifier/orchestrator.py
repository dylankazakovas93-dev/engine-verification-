from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

from .adapter import load_adapter
from .causality import audit_adapter_causality
from .contracts import audit_selection_map
from .data import audit_dataset, sha256_file
from .high_resolution import audit_high_resolution_execution, compare_one_second_aggregation
from .ledger import audit_ledger
from .profile import load_profile
from .reconciliation import reconcile_ledgers
from .report import VerificationReport
from .schema import Finding
from .static_scan import scan_source


@dataclass(frozen=True)
class VerificationOptions:
    strategy_dir: Path
    candidate_adapter: Path
    candidate_source: Path | None = None
    data: Path | None = None
    timestamp_col: str = "timestamp"
    contract_col: str | None = None
    selection_map: Path | None = None
    expected_contract_col: str = "expected_contract_id"
    high_res_data: Path | None = None
    high_res_timestamp_col: str = "ts_event"
    high_res_contract_col: str | None = "instrument_id"
    high_res_symbol_col: str | None = "symbol"
    audit_mode: str | None = None
    run_tests: bool = True
    report_prefix: Path | None = None


def hash_source(path: str | Path) -> str:
    p = Path(path).resolve()
    digest = hashlib.sha256()
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*.py") if "__pycache__" not in x.parts)
    for file in files:
        digest.update(file.relative_to(p.parent if p.is_file() else p).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_candidate_bars(path: Path, timestamp_col: str, *, max_rows: int = 12_000_000) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows > max_rows:
            raise ValueError(f"candidate dataset has {pf.metadata.num_rows} rows; refuse full-RAM load above {max_rows}")
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
        if len(raw) > max_rows:
            raise ValueError(f"candidate dataset has {len(raw)} rows; refuse full-RAM load above {max_rows}")
    if timestamp_col not in raw:
        raise ValueError(f"timestamp column {timestamp_col!r} not found")
    parsed = pd.to_datetime(raw[timestamp_col], errors="raise", utc=False)
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamps must carry an explicit timezone")
    index = pd.DatetimeIndex(parsed.dt.tz_convert("UTC"))
    return raw.drop(columns=[timestamp_col]).set_axis(index, axis=0)


def load_high_resolution_trade_windows(
    path: Path,
    trades: pd.DataFrame,
    *,
    timestamp_col: str,
    high_res_contract_col: str,
    candidate_contract_col: str,
    interval: str | pd.Timedelta,
    max_windows: int = 500,
) -> pd.DataFrame:
    """Use Parquet predicate pushdown to load only execution windows needed by trades."""
    import pyarrow.dataset as ds
    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("bounded high-resolution window loading currently requires Parquet")
    required = {"exit_time", candidate_contract_col}
    if not required.issubset(trades.columns):
        raise ValueError(f"trade ledger missing {sorted(required - set(trades.columns))}")
    duration = pd.Timedelta(interval)
    windows = trades[["exit_time", candidate_contract_col]].copy()
    windows["exit_time"] = pd.to_datetime(windows["exit_time"], utc=True, errors="coerce").dt.floor(duration)
    windows = windows.dropna().drop_duplicates().head(max_windows)
    if windows.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", candidate_contract_col], index=pd.DatetimeIndex([], tz="UTC"))
    dataset = ds.dataset(path, format="parquet")
    frames = []
    columns = [timestamp_col, "open", "high", "low", "close", "volume", high_res_contract_col]
    for event_time, contract in windows.itertuples(index=False, name=None):
        predicate = (
            (ds.field(timestamp_col) >= event_time.to_pydatetime())
            & (ds.field(timestamp_col) < (event_time + duration).to_pydatetime())
            & (ds.field(high_res_contract_col) == contract)
        )
        table = dataset.to_table(columns=columns, filter=predicate, use_threads=True)
        if table.num_rows:
            frame = table.to_pandas().rename(columns={high_res_contract_col: candidate_contract_col})
            frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
            frames.append(frame.set_index(timestamp_col))
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", candidate_contract_col], index=pd.DatetimeIndex([], tz="UTC"))
    return pd.concat(frames).sort_index(kind="stable")


def _pytest_finding(root: Path) -> tuple[list[Finding], dict[str, Any]]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=root,
        text=True, capture_output=True,
    )
    output = (completed.stdout + completed.stderr).strip()
    status = "PASS" if completed.returncode == 0 else "FAIL"
    findings = [
        Finding("repository_health", status, output or f"pytest exit code {completed.returncode}", family="repository_health"),
        Finding("deterministic_tests", status, output or f"pytest exit code {completed.returncode}", family="deterministic_tests"),
    ]
    return findings, {"pytest_exit_code": completed.returncode, "pytest_output": output}


def verify(options: VerificationOptions) -> VerificationReport:
    strategy_dir = options.strategy_dir.resolve()
    root = strategy_dir.parents[1]
    profile = load_profile(strategy_dir / "profile.toml")
    adapter = load_adapter(str(options.candidate_adapter))
    findings: list[Finding] = []
    artifacts: dict[str, Any] = {}
    coverage: dict[str, Any] = {}

    if options.run_tests:
        test_findings, test_artifact = _pytest_finding(root)
        findings += test_findings
        artifacts["repository_tests"] = test_artifact
    else:
        findings += [
            Finding("repository_health", "UNVERIFIED", "repository tests explicitly skipped", family="repository_health"),
            Finding("deterministic_tests", "UNVERIFIED", "deterministic tests explicitly skipped", family="deterministic_tests"),
        ]

    source = options.candidate_source or options.candidate_adapter
    artifacts["candidate_source"] = {"path": str(source.resolve()), "sha256": hash_source(source)}
    findings += scan_source(source)

    bars: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    if options.data is None:
        for family, check in (
            ("data_audit", "canonical_data"), ("ledger", "ledger"), ("causality", "causality"),
            ("feature_causality", "feature_causality"), ("reference_reconciliation", "reference_reconciliation"),
            ("contract_provenance", "contract_provenance"),
        ):
            findings.append(Finding(check, "UNVERIFIED", "canonical strategy dataset not supplied", family=family))
    else:
        manifest, data_findings = audit_dataset(
            options.data, timestamp_col=options.timestamp_col,
            expected_interval=profile.get("bar_interval", "1min"), contract_col=options.contract_col,
        )
        findings += data_findings
        artifacts["canonical_data_manifest"] = manifest.to_dict()
        if not any(f.status == "FAIL" and f.family == "data_audit" for f in data_findings):
            try:
                bars = load_candidate_bars(options.data, options.timestamp_col)
                trades = adapter.run(bars)
                if not isinstance(trades, pd.DataFrame):
                    findings.append(Finding("adapter", "FAIL", "adapter run() did not return a DataFrame", family="ledger"))
                else:
                    findings += audit_ledger(
                        trades,
                        strict_same_timestamp_reentry=bool(profile.get("strict_same_timestamp_reentry", True)),
                        tick_size=float(profile.get("tick_size")) if profile.get("tick_size") is not None else None,
                        long_only=bool(profile.get("long_only")) if profile.get("long_only") is not None else None,
                        required_deadline=profile.get("required_deadline") is not None,
                    )
                    mode = options.audit_mode or str(profile.causality.get("mode", "strong"))
                    seed = int(profile.causality.get("seed", 1729))
                    causal_findings, causal_coverage = audit_adapter_causality(adapter, bars, mode=mode, seed=seed)
                    findings += causal_findings
                    coverage["causality"] = causal_coverage.to_dict()
                    reference_fn = getattr(adapter, "reference", None)
                    if callable(reference_fn):
                        findings += reconcile_ledgers(trades, reference_fn(bars))
                    else:
                        findings.append(Finding("reference_reconciliation", "UNVERIFIED", "adapter exposes no independent reference", family="reference_reconciliation"))
            except Exception as exc:
                findings.append(Finding("candidate_execution", "FAIL", f"{type(exc).__name__}: {exc}", family="ledger"))

        if options.selection_map is not None and options.contract_col:
            selected = pd.read_parquet(options.data) if options.data.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(options.data)
            expected = pd.read_csv(options.selection_map)
            findings += audit_selection_map(
                selected, expected, timestamp_col=options.timestamp_col,
                contract_col=options.contract_col, expected_contract_col=options.expected_contract_col,
            )

    if options.high_res_data is not None:
        high_manifest, high_findings = audit_dataset(
            options.high_res_data, timestamp_col=options.high_res_timestamp_col,
            expected_interval="1s", contract_col=options.high_res_contract_col,
            symbol_col=options.high_res_symbol_col, allow_nonpositive_prices=True,
        )
        artifacts["high_resolution_data_manifest"] = high_manifest.to_dict()
        findings += [Finding(
            "high_resolution_data_audit", f.status, f.message,
            rows=f.rows, family="high_resolution", classification=f.classification, evidence=f.evidence,
        ) for f in high_findings if f.family == "data_audit"]
        if trades is None:
            findings.append(Finding("high_resolution_execution", "UNVERIFIED", "canonical candidate trades unavailable", family="high_resolution"))
        elif options.contract_col is None or options.contract_col not in trades.columns:
            findings.append(Finding(
                "high_resolution_execution", "UNVERIFIED",
                "one-second raw contracts cannot be matched to continuous trades without selected contract identity",
                family="high_resolution",
            ))
        elif options.high_res_contract_col is None:
            findings.append(Finding("high_resolution_execution", "UNVERIFIED", "one-second contract column not configured", family="high_resolution"))
        else:
            try:
                interval = str(profile.get("bar_interval", "1min"))
                seconds = load_high_resolution_trade_windows(
                    options.high_res_data, trades,
                    timestamp_col=options.high_res_timestamp_col,
                    high_res_contract_col=options.high_res_contract_col,
                    candidate_contract_col=options.contract_col,
                    interval=interval,
                )
                high_execution, resolutions = audit_high_resolution_execution(
                    trades, bars, seconds, coarse_interval=interval, contract_col=options.contract_col,
                )
                findings += high_execution
                findings += compare_one_second_aggregation(seconds, bars, contract_col=options.contract_col)
                coverage["high_resolution"] = {
                    "loaded_rows": len(seconds), "ambiguous_trade_count": len(resolutions),
                    "classifications": [item.to_dict() for item in resolutions[:100]],
                }
            except Exception as exc:
                findings.append(Finding("high_resolution_execution", "UNVERIFIED", f"bounded one-second audit unavailable: {type(exc).__name__}: {exc}", family="high_resolution"))
    elif profile.checks.get("high_resolution", False):
        findings.append(Finding("high_resolution_execution", "UNVERIFIED", "mandatory one-second dataset not supplied", family="high_resolution"))

    report = VerificationReport(
        strategy=f"{profile.name} v{profile.version}", candidate=str(options.candidate_adapter.resolve()),
        findings=findings, mandatory_checks=profile.mandatory_checks, artifacts=artifacts, coverage=coverage,
    )
    prefix = options.report_prefix or (root / "reports" / strategy_dir.name)
    report.write_json(str(prefix) + ".json")
    report.write_markdown(str(prefix) + ".md")
    return report
