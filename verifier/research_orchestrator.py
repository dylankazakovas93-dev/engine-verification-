"""One-command research verification (isolated from frozen strategy verification).

Pipeline:
data audit -> static scan -> events/features/targets -> research contract -> research causality
-> lockbox split -> purged walk-forward folds -> ML leakage poisoning -> OOF alignment
-> lockbox access audit -> JSON + Markdown report with the repository's machine verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data import audit_dataset
from .lockbox import assert_model_tables_exclude_lockbox, audit_lockbox, parse_lockbox_start, split_lockbox
from .ml_leakage import ML_STATIC_FAMILY, audit_ml_leakage, scan_ml_source
from .orchestrator import _pytest_finding, hash_source, load_candidate_bars
from .report import VerificationReport
from .research_causality import audit_research_causality
from .research_contracts import (
    FEATURE_KEYS, TARGET_KEYS, ResearchTables, audit_research_contract, load_research_adapter, select_target,
)
from .schema import Finding
from .static_scan import scan_source
from .walkforward import audit_folds, audit_oof_predictions, build_purged_walkforward_folds

RESEARCH_FAMILIES = ("research_contract", "research_causality", "walkforward", "ml_leakage", "lockbox")
# Data/provenance/repository gates are mandatory exactly as in the frozen strategy verifier.
# ML-specific static heuristics are mandatory-for-review (WARN -> INCOMPLETE, never FAIL); the
# generic ``static_scan`` family flags every .fit() and therefore stays informational.
MANDATORY_FAMILIES = frozenset({"repository_health", "data_audit", "contract_provenance", ML_STATIC_FAMILY, *RESEARCH_FAMILIES})


@dataclass(frozen=True)
class ResearchVerificationOptions:
    adapter: Path
    data: Path | None
    source: Path | None = None
    timestamp_col: str = "timestamp"
    target: str | None = None
    lockbox_start: str | None = None
    mode: str = "strong"
    seed: int = 1729
    contract_col: str | None = None
    expected_interval: str | None = None
    min_train_events: int = 20
    run_tests: bool = True
    report_prefix: Path | None = None


def _blocked(family: str, check: str, why: str) -> Finding:
    return Finding(check, "UNVERIFIED", f"blocked: {why}", family=family)


def family_status(findings: list[Finding], families: tuple[str, ...] | frozenset[str]) -> dict[str, str]:
    """PASS / FAIL / UNVERIFIED per family using the same precedence as evaluate_verdict."""
    out = {}
    for family in sorted(families):
        related = [f for f in findings if (f.family or f.check) == family]
        if any(f.status == "FAIL" for f in related):
            out[family] = "FAIL"
        elif not related or any(f.status in {"UNVERIFIED", "WARN"} for f in related) or not any(f.status in {"PASS", "INFO"} for f in related):
            out[family] = "UNVERIFIED"
        else:
            out[family] = "PASS"
    return out


def run_research_audit(
    adapter_factory: Callable[[], Any], bars: pd.DataFrame, *, target_name: str | None,
    lockbox_start: pd.Timestamp | str | None, mode: str = "strong", seed: int = 1729,
    min_train_events: int = 20, cutoffs: list[pd.Timestamp] | None = None,
) -> tuple[list[Finding], dict[str, Any], dict[str, Any]]:
    """Library entry point used by the CLI and the self-tests (no file IO)."""
    lockbox_ts = parse_lockbox_start(lockbox_start)
    coverage: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}

    # 1. contract ------------------------------------------------------------------------
    findings, tables, ready = audit_research_contract(adapter_factory, bars, target_name=target_name)
    artifacts["research_readiness"] = dict(ready)
    if tables is None or tables.events is None or not ready["events_ok"]:
        why = "research contract did not yield a usable events table"
        findings += [_blocked("research_causality", "research_causality", why), _blocked("walkforward", "walkforward_folds", why),
                     _blocked("ml_leakage", "ml_leakage", why), _blocked("lockbox", "lockbox_declared", why)]
        return findings, coverage, artifacts

    # 2. causality (features/targets only when their key columns exist) ------------------
    causal_tables = ResearchTables(
        tables.events,
        tables.features if tables.features is not None and set(FEATURE_KEYS).issubset(tables.features.columns) else None,
        tables.targets if tables.targets is not None and set(TARGET_KEYS).issubset(tables.targets.columns) else None,
    )
    causal_findings, causal_coverage = audit_research_causality(
        adapter_factory, bars, causal_tables, mode=mode, seed=seed, cutoffs=cutoffs,
    )
    findings += causal_findings
    coverage["research_causality"] = causal_coverage.to_dict()

    # 3. lockbox split -------------------------------------------------------------------
    model_ready = ready["features_ok"] and ready["targets_ok"] and ready["target_ok"]
    split = split_lockbox(tables.events, tables.features if ready["features_ok"] else None,
                          tables.targets if ready["targets_ok"] else None, lockbox_ts)
    artifacts["lockbox"] = split.counts()
    accessed: set | None = None
    if not model_ready:
        why = "features/targets/selected target failed or missing in research_contract"
        findings += [_blocked("walkforward", "walkforward_folds", why), _blocked("ml_leakage", "ml_leakage", why)]
        findings += audit_lockbox(split, None)
        return findings, coverage, artifacts

    # 4. walk-forward observations: development events with a feature row and a finite label.
    try:
        selected = select_target(split.targets, target_name)
    except ValueError:  # every labelled event was withheld by the lockbox
        selected = pd.DataFrame({"event_id": pd.Series([], dtype=object), "target_start": pd.Series(pd.DatetimeIndex([], tz="UTC")),
                                 "target_end": pd.Series(pd.DatetimeIndex([], tz="UTC")), "value": pd.Series([], dtype=float)})
    selected = selected[np.isfinite(selected["value"].to_numpy(float))]
    selected = selected[selected["event_id"].isin(set(split.features["event_id"].tolist()))].reset_index(drop=True)
    event_time = split.event_times
    observations = pd.DataFrame({
        "event_id": selected["event_id"],
        "event_time": pd.Series(event_time.reindex(pd.Index(selected["event_id"].to_numpy(), dtype=object)).array),
        "target_start": selected["target_start"],
        "target_end": selected["target_end"],
    })
    order = observations.assign(__id=observations["event_id"].astype(str)).sort_values(["event_time", "__id"], kind="mergesort").index
    observations = observations.loc[order].reset_index(drop=True)
    obs_ids = observations["event_id"].tolist()
    position = {i: k for k, i in enumerate(obs_ids)}
    model_features = split.features[split.features["event_id"].isin(position)].copy()
    model_features = model_features.iloc[np.argsort(model_features["event_id"].map(position).to_numpy(), kind="stable")].reset_index(drop=True)
    model_targets = split.targets[split.targets["event_id"].isin(position)].copy()
    model_targets = model_targets.iloc[np.argsort(model_targets["event_id"].map(position).to_numpy(), kind="stable")].reset_index(drop=True)

    folds, skipped = build_purged_walkforward_folds(observations, lockbox_start=lockbox_ts, min_train_events=min_train_events)
    findings += audit_folds(folds, observations, lockbox_start=lockbox_ts)
    coverage["walkforward"] = {
        "observations": len(observations), "folds": [f.to_dict() for f in folds], "skipped_blocks": skipped,
        "block": "UTC calendar year", "purge_rule": "train iff event_time < validation_start and target_end < validation_start",
    }

    # 5. ML leakage poisoning --------------------------------------------------------------
    def guard(features: pd.DataFrame, targets: pd.DataFrame) -> None:
        assert_model_tables_exclude_lockbox(features, targets, event_time, lockbox_ts)

    ml_findings, ml_coverage, predictions, accessed = audit_ml_leakage(
        adapter_factory, model_features, model_targets, event_time, folds, str(target_name),
        guard=guard, time_ceiling=lockbox_ts, mode=mode, seed=seed,
    )
    findings += ml_findings
    coverage["ml_leakage"] = ml_coverage.to_dict()
    if not ready["model_available"]:
        findings.append(Finding("walkforward_oof_alignment", "UNVERIFIED", "no fit_predict_fold(); OOF alignment not evaluated", family="walkforward"))
    else:
        findings += audit_oof_predictions(folds, predictions)

    # 6. lockbox isolation proof (tables + access log) -------------------------------------
    findings += audit_lockbox(split, accessed if ready["model_available"] and folds else None)
    return findings, coverage, artifacts


def verify_research(options: ResearchVerificationOptions) -> VerificationReport:
    root = Path(__file__).resolve().parents[1]
    adapter_path = options.adapter.resolve()
    findings: list[Finding] = []
    artifacts: dict[str, Any] = {"options": {
        "target": options.target, "lockbox_start": options.lockbox_start, "mode": options.mode,
        "seed": options.seed, "min_train_events": options.min_train_events, "timestamp_col": options.timestamp_col,
    }}
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

    source = (options.source or options.adapter).resolve()
    artifacts["candidate_source"] = {"path": str(source), "sha256": hash_source(source)}
    artifacts["candidate_adapter"] = {"path": str(adapter_path), "sha256": hash_source(adapter_path)}
    findings += scan_source(source)
    findings += scan_ml_source(source)

    if options.data is None:
        for family in ("data_audit", "contract_provenance", *RESEARCH_FAMILIES):
            findings.append(Finding(family, "UNVERIFIED", "research dataset not supplied", family=family))
    else:
        manifest, data_findings = audit_dataset(
            options.data, timestamp_col=options.timestamp_col,
            expected_interval=options.expected_interval, contract_col=options.contract_col,
        )
        findings += data_findings
        artifacts["data_manifest"] = manifest.to_dict()
        if any(f.status == "FAIL" and f.family == "data_audit" for f in data_findings):
            for family in RESEARCH_FAMILIES:
                findings.append(_blocked(family, family, "data audit failed; research candidate not executed"))
        else:
            try:
                bars = load_candidate_bars(options.data, options.timestamp_col)
                research_findings, research_coverage, research_artifacts = run_research_audit(
                    lambda: load_research_adapter(adapter_path), bars,
                    target_name=options.target, lockbox_start=options.lockbox_start,
                    mode=options.mode, seed=options.seed, min_train_events=options.min_train_events,
                )
                findings += research_findings
                coverage.update(research_coverage)
                artifacts.update(research_artifacts)
            except Exception as exc:
                findings.append(Finding("candidate_execution", "FAIL", f"{type(exc).__name__}: {exc}", family="research_contract"))

    artifacts["family_status"] = family_status(findings, MANDATORY_FAMILIES | {"static_scan"})
    report = VerificationReport(
        strategy="research verification", candidate=str(adapter_path), findings=findings,
        mandatory_checks=set(MANDATORY_FAMILIES), artifacts=artifacts, coverage=coverage,
    )
    prefix = options.report_prefix or (root / "reports" / f"research_{adapter_path.stem}")
    report.write_json(str(prefix) + ".json")
    report.write_markdown(str(prefix) + ".md")
    return report
