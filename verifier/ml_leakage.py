"""Dynamic ML leakage poisoning for verifier-controlled walk-forward folds.

The verifier owns the folds and calls the untrusted ``fit_predict_fold``. For every audited fold
it runs a baseline, then re-runs with deliberately poisoned rows that the model is NOT allowed
to use. Validation predictions must be identical (within a tiny tolerance):

A. validation-label poisoning        - labels of the SAME validation fold
B. future-label poisoning             - labels of development rows after the validation block
B2. purged-label poisoning            - labels of rows removed by purging (window overlaps validation)
C. future-feature poisoning           - features of rows after the validation block
C2. purged-feature poisoning          - features of purged rows (not legal training rows)
E. intra-validation feature poisoning - features of LATER validation rows must not move
                                        predictions of EARLIER validation rows

Positive controls (training labels / training features poisoned) must CHANGE predictions;
otherwise the forbidden-row tests have no demonstrated power and the family is not a PASS.

Every call receives a fresh adapter instance, deep copies of the verifier's tables, and is
preceded by a hard lockbox guard. A candidate that mutates its inputs, is nondeterministic, or
raises when forbidden rows are poisoned is reported as FAIL.

``scan_ml_source`` adds ML-specific *heuristic* static warnings. It never FAILs on its own;
dynamic poisoning is the stronger evidence.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .research_contracts import FEATURE_KEYS, MODEL_FUNCTION, TARGET_KEYS, TARGET_NAME_COL, to_utc
from .schema import Finding
from .walkforward import Fold, prediction_frame_problem

FAMILY = "ml_leakage"
PROVEN = "PROVEN FAILURE"
HIGH = "HIGH-RISK / MANUAL REVIEW"
INFO = "INFO"
MODE_FOLD_LIMITS = {"fast": 3, "standard": 6, "strong": None}
FORBIDDEN_TESTS = (
    "validation_label_poisoning", "future_label_poisoning", "purged_label_poisoning",
    "future_feature_poisoning", "purged_feature_poisoning", "intra_validation_feature_poisoning",
)
CONTROL_TESTS = ("control_train_label_sensitivity", "control_train_feature_sensitivity")
REQUIRED_FORBIDDEN = ("validation_label_poisoning", "future_label_poisoning", "future_feature_poisoning")


def _f(check: str, status: str, message: str, *, classification: str | None = None, evidence: dict | None = None) -> Finding:
    return Finding(check, status, message, family=FAMILY, classification=classification, evidence=evidence)


# ---------------------------------------------------------------------------
# Poisoning primitives
# ---------------------------------------------------------------------------
def _poison_series(series: pd.Series, mask: np.ndarray, rng: np.random.Generator) -> pd.Series:
    """Drastically change ``series`` on ``mask`` rows; dtype-preserving where possible."""
    out = series.copy()
    if not mask.any():
        return out
    if pd.api.types.is_bool_dtype(series):
        out.loc[mask] = ~series.loc[mask].astype(bool)
        return out
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy(dtype=float, copy=True)
        finite = values[np.isfinite(values)]
        uniques = np.unique(finite)
        selected = values[mask]
        if len(uniques) >= 2 and len(uniques) <= 10 and np.all(np.equal(np.round(uniques), uniques)):
            # Discrete labels/classes: cyclically remap to a different observed class.
            position = {v: i for i, v in enumerate(uniques)}
            remapped = np.array([uniques[(position[v] + 1) % len(uniques)] if np.isfinite(v) else v for v in selected])
            new = remapped
        else:
            scale = float(np.nanstd(finite)) if finite.size > 1 else 1.0
            scale = scale if np.isfinite(scale) and scale > 0 else 1.0
            noise = rng.normal(0.0, 1.0, size=selected.shape)
            new = -37.0 * selected + scale * (1_000.0 + 50.0 * noise)
        if pd.api.types.is_integer_dtype(series) and np.all(np.isfinite(new)):
            new = np.round(new).astype(series.dtype)
        out.loc[mask] = new
        return out
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        out.loc[mask] = series.loc[mask].astype(str) + "~poisoned"
        return out
    return out  # categorical/datetime payloads are left untouched and reported as such


def poison_features(features: pd.DataFrame, ids: set, seed: int) -> tuple[pd.DataFrame, list[str]]:
    """Return poisoned copy and the list of feature columns that could not be mutated."""
    rng = np.random.default_rng(seed)
    mask = features["event_id"].isin(ids).to_numpy()
    out = features.copy(deep=True)
    untouched = []
    for column in features.columns:
        if column in FEATURE_KEYS:
            continue
        before = out[column]
        out[column] = _poison_series(before, mask, rng)
        if mask.any() and before.loc[mask].astype(str).equals(out.loc[mask, column].astype(str)):
            untouched.append(str(column))
    return out, untouched


def poison_targets(targets: pd.DataFrame, ids: set, seed: int, *, time_ceiling: pd.Timestamp | None = None) -> pd.DataFrame:
    """Poison every label column and move the target window later.

    Keeps ``event_time <= target_start <= target_end``. With ``time_ceiling`` (the lockbox
    boundary) the poisoned window stays strictly before it, so poisoning can never manufacture a
    row that looks like lockbox data.
    """
    rng = np.random.default_rng(seed)
    mask = targets["event_id"].isin(ids).to_numpy()
    out = targets.copy(deep=True)
    keys = set(TARGET_KEYS) | {TARGET_NAME_COL}
    for column in targets.columns:
        if column not in keys:
            out[column] = _poison_series(out[column], mask, rng)
    if mask.any():
        count = int(mask.sum())
        start = to_utc(out.loc[mask, "target_start"])
        end = to_utc(out.loc[mask, "target_end"])
        if time_ceiling is None:
            shift = pd.to_timedelta(rng.integers(1, 97, size=count), unit="h")
            new_end = end + shift + pd.to_timedelta(rng.integers(1, 97, size=count), unit="h")
            new_start = start + shift
        else:
            room = (time_ceiling - pd.Timedelta(microseconds=1)) - end
            room = room.where(room > pd.Timedelta(0), pd.Timedelta(0))
            new_end = end + room * rng.uniform(0.2, 0.8, size=count)
            new_start = start + (new_end - start) * rng.uniform(0.1, 0.9, size=count)
        out.loc[mask, "target_start"] = new_start.dt.floor("us").to_numpy()
        out.loc[mask, "target_end"] = new_end.dt.floor("us").to_numpy()
        out["target_start"] = to_utc(out["target_start"])
        out["target_end"] = to_utc(out["target_end"])
    return out


# ---------------------------------------------------------------------------
# Model invocation
# ---------------------------------------------------------------------------
@dataclass
class CallResult:
    frame: pd.DataFrame | None
    error: str | None
    mutated_inputs: list[str]


@dataclass
class MLLeakageCoverage:
    mode: str
    seed: int
    folds_available: int = 0
    folds_audited: list[int] = field(default_factory=list)
    model_calls: int = 0
    per_fold: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "seed": self.seed, "folds_available": self.folds_available,
            "folds_audited": list(self.folds_audited), "model_calls": self.model_calls,
            "per_fold": self.per_fold,
        }


class ModelRunner:
    """Owns every call into the untrusted model; enforces the lockbox guard and logs access."""

    def __init__(self, adapter_factory: Callable[[], Any], target_name: str,
                 guard: Callable[[pd.DataFrame, pd.DataFrame], None] | None = None) -> None:
        self.adapter_factory = adapter_factory
        self.target_name = target_name
        self.guard = guard
        self.accessed_ids: set = set()
        self.calls = 0

    def __call__(self, features: pd.DataFrame, targets: pd.DataFrame, train_ids: tuple, validation_ids: tuple) -> CallResult:
        if self.guard is not None:
            self.guard(features, targets)  # LockboxViolation propagates: it is a verifier bug, never swallowed
        self.accessed_ids.update(features["event_id"].tolist())
        self.accessed_ids.update(targets["event_id"].tolist())
        f_in, t_in = features.copy(deep=True), targets.copy(deep=True)
        train_in, validation_in = list(train_ids), list(validation_ids)
        self.calls += 1
        try:
            fn = getattr(self.adapter_factory(), MODEL_FUNCTION)
            frame = fn(f_in, t_in, train_in, validation_in, self.target_name)
        except Exception as exc:
            return CallResult(None, f"{type(exc).__name__}: {exc}", [])
        mutated = []
        if not f_in.equals(features):
            mutated.append("features")
        if not t_in.equals(targets):
            mutated.append("targets")
        if train_in != list(train_ids):
            mutated.append("train_ids")
        if validation_in != list(validation_ids):
            mutated.append("validation_ids")
        return CallResult(frame, None, mutated)


def _aligned(frame: pd.DataFrame, ids: list) -> np.ndarray:
    series = pd.Series(pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(float),
                       index=pd.Index(frame["event_id"].to_numpy(), dtype=object))
    return series.reindex(pd.Index(ids, dtype=object)).to_numpy(float)


def compare_predictions(base: pd.DataFrame, other: pd.DataFrame, ids: list, *, atol: float, rtol: float) -> tuple[bool, dict[str, Any]]:
    a, b = _aligned(base, ids), _aligned(other, ids)
    same = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    changed = [str(ids[i]) for i in np.flatnonzero(~same)]
    diff = np.abs(a - b)
    max_diff = float(np.nanmax(diff)) if diff.size and np.isfinite(diff).any() else float("nan")
    return bool(same.all()), {"compared": len(ids), "changed": len(changed), "changed_ids": changed[:10], "max_abs_diff": max_diff}


def _describe_change(entry: dict[str, Any]) -> str:
    if entry.get("error"):
        return str(entry["error"])
    return (f"{entry['changed']} of {entry['compared']} predictions changed; "
            f"max |diff|={entry['max_abs_diff']:.6g}; e.g. {entry['changed_ids'][:5]}")


def select_folds(folds: list[Fold], mode: str) -> list[Fold]:
    limit = MODE_FOLD_LIMITS[mode]
    if limit is None or len(folds) <= limit:
        return list(folds)
    picks = set(np.linspace(0, len(folds) - 2, limit - 1, dtype=int).tolist()) | {len(folds) - 1}
    return [folds[i] for i in sorted(picks)]


def audit_ml_leakage(
    adapter_factory: Callable[[], Any], features: pd.DataFrame, targets: pd.DataFrame,
    event_times: pd.Series, folds: list[Fold], target_name: str, *,
    guard: Callable[[pd.DataFrame, pd.DataFrame], None] | None = None,
    time_ceiling: pd.Timestamp | None = None,
    mode: str = "strong", seed: int = 1729, atol: float = 1e-9, rtol: float = 1e-9,
) -> tuple[list[Finding], MLLeakageCoverage, dict[int, pd.DataFrame | None], set]:
    """Run baseline + poisoned fits for each audited fold.

    ``features``/``targets`` are the model-accessible (lockbox-free) development tables restricted
    to walk-forward observations. ``event_times`` maps event_id -> UTC event_time.
    Returns findings, coverage, baseline OOF predictions per fold, and the set of accessed IDs.
    """
    coverage = MLLeakageCoverage(mode, seed, folds_available=len(folds))
    if not callable(getattr(adapter_factory(), MODEL_FUNCTION, None)):
        return [_f("ml_leakage", "UNVERIFIED", f"adapter exposes no {MODEL_FUNCTION}(); ML leakage UNVERIFIED")], coverage, {}, set()
    if not folds:
        return [_f("ml_leakage", "UNVERIFIED", "no walk-forward folds; ML leakage poisoning could not run")], coverage, {}, set()
    runner = ModelRunner(adapter_factory, target_name, guard)
    audited = select_folds(folds, mode)
    coverage.folds_audited = [f.fold for f in audited]
    predictions: dict[int, pd.DataFrame | None] = {f.fold: None for f in folds}
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in (*FORBIDDEN_TESTS, *CONTROL_TESTS)}
    findings: list[Finding] = []
    lookup = event_times.reindex(pd.Index(features["event_id"].to_numpy(), dtype=object))
    times_by_id = dict(zip(features["event_id"].tolist(), lookup.tolist()))

    # Baseline for folds that are not poison-audited still feeds the OOF alignment audit.
    for fold in folds:
        base = runner(features, targets, fold.train_ids, fold.validation_ids)
        record: dict[str, Any] = {**fold.to_dict(), "audited": fold in audited}
        coverage.per_fold.append(record)
        if base.error is not None:
            findings.append(_f(f"fold{fold.fold}_baseline_execution", "FAIL", f"fold {fold.fold}: baseline fit raised {base.error}", classification=PROVEN))
            continue
        if base.mutated_inputs:
            record["mutates_inputs_in_place"] = base.mutated_inputs
        problem = prediction_frame_problem(base.frame, fold.validation_ids)
        predictions[fold.fold] = base.frame
        if problem is not None:
            record["baseline_problem"] = problem
            continue
        if fold not in audited:
            continue
        repeat = runner(features, targets, fold.train_ids, fold.validation_ids)
        validation = list(fold.validation_ids)
        if repeat.error is not None or prediction_frame_problem(repeat.frame, fold.validation_ids) is not None:
            findings.append(_f(f"fold{fold.fold}_determinism", "FAIL", f"fold {fold.fold}: identical re-run failed: {repeat.error or 'invalid prediction frame'}", classification=PROVEN))
            continue
        same, detail = compare_predictions(base.frame, repeat.frame, validation, atol=atol, rtol=rtol)
        if not same:
            findings.append(_f(
                f"fold{fold.fold}_determinism", "FAIL",
                f"fold {fold.fold}: identical inputs gave different predictions ({detail}); poisoning is uninterpretable",
                classification=PROVEN, evidence=detail,
            ))
            continue
        record["deterministic"] = True
        train = set(fold.train_ids)
        val_set = set(validation)
        purged = set(fold.purged_ids)
        future = {i for i in features["event_id"].tolist()
                  if i not in train and i not in val_set and times_by_id[i] >= fold.validation_end}
        val_times = sorted({times_by_id[i] for i in validation})
        late_val: set = set()
        early_val: list = []
        if len(val_times) >= 2:
            split_time = val_times[(len(val_times) - 1) // 2]
            late_val = {i for i in validation if times_by_id[i] > split_time}
            early_val = [i for i in validation if times_by_id[i] <= split_time]
        record.update({"future_rows": len(future), "purged_rows": len(purged), "intra_validation_poisoned": len(late_val)})
        plan: list[tuple[str, set, str, list]] = [
            ("validation_label_poisoning", val_set, "targets", validation),
            ("future_label_poisoning", future, "targets", validation),
            ("purged_label_poisoning", purged, "targets", validation),
            ("future_feature_poisoning", future, "features", validation),
            ("purged_feature_poisoning", purged, "features", validation),
            ("intra_validation_feature_poisoning", late_val, "features", early_val),
            ("control_train_label_sensitivity", train, "targets", validation),
            ("control_train_feature_sensitivity", train, "features", validation),
        ]
        for offset, (name, ids, table, compare_ids) in enumerate(plan):
            if not ids or not compare_ids:
                results[name].append({"fold": fold.fold, "applicable": False})
                continue
            test_seed = seed + 1000 * fold.fold + offset
            untouched: list[str] = []
            if table == "targets":
                p_features, p_targets = features, poison_targets(targets, ids, test_seed, time_ceiling=time_ceiling)
            else:
                p_features, untouched = poison_features(features, ids, test_seed)
                p_targets = targets
            poisoned = runner(p_features, p_targets, fold.train_ids, fold.validation_ids)
            entry: dict[str, Any] = {"fold": fold.fold, "applicable": True, "poisoned_rows": len(ids), "untouched_columns": untouched}
            if poisoned.error is not None:
                entry.update({"changed": True, "error": poisoned.error})
            else:
                problem = prediction_frame_problem(poisoned.frame, fold.validation_ids)
                if problem is not None:
                    entry.update({"changed": True, "error": f"invalid prediction frame under poisoning: {problem}"})
                else:
                    same, detail = compare_predictions(base.frame, poisoned.frame, compare_ids, atol=atol, rtol=rtol)
                    entry.update({"changed": not same, **detail})
            results[name].append(entry)
    coverage.model_calls = runner.calls

    for name in FORBIDDEN_TESTS:
        entries = results[name]
        applicable = [e for e in entries if e["applicable"]]
        failing = [e for e in applicable if e["changed"]]
        for e in failing:
            findings.append(_f(
                name, "FAIL",
                f"fold {e['fold']}: predictions changed after poisoning {e['poisoned_rows']} forbidden rows ({_describe_change(e)})",
                classification=PROVEN, evidence=e,
            ))
        if failing:
            continue
        if not applicable:
            status = "UNVERIFIED" if name in REQUIRED_FORBIDDEN else "INFO"
            findings.append(_f(name, status, "not applicable in any audited fold (no rows of that kind); no evidence produced"))
            continue
        compared = sum(e.get("compared", 0) for e in applicable)
        untouched = sorted({c for e in applicable for c in e.get("untouched_columns", [])})
        findings.append(_f(
            name, "PASS",
            f"predictions unchanged in {len(applicable)} folds ({compared} predictions compared)"
            + (f"; columns not mutable by the poisoner: {untouched}" if untouched else ""),
        ))
    for name in CONTROL_TESTS:
        applicable = [e for e in results[name] if e["applicable"]]
        sensitive = [e for e in applicable if e["changed"] and not e.get("error")]
        errors = [e for e in applicable if e.get("error")]
        if sensitive:
            findings.append(_f(name, "PASS", f"positive control: poisoning training rows changed predictions in {len(sensitive)}/{len(applicable)} folds; the poisoner reaches the model"))
        elif errors:
            findings.append(_f(name, "UNVERIFIED", f"positive control inconclusive: candidate raised under training-row poisoning: {errors[0].get('error')}", classification=HIGH))
        else:
            findings.append(_f(
                name, "WARN",
                "LOW POWER: poisoning legal training rows never changed predictions; forbidden-row poisoning cannot be shown to reach the model",
                classification=HIGH,
            ))
    mutators = [r["fold"] for r in coverage.per_fold if r.get("mutates_inputs_in_place")]
    if mutators:
        findings.append(_f(
            "input_mutation", "INFO",
            f"fit_predict_fold mutates its input tables/lists in place (folds {mutators}); harmless to verification because every call receives deep copies",
        ))
    split_tests = [n for n in FORBIDDEN_TESTS if n != "intra_validation_feature_poisoning"]
    violated = sorted({f.check for f in findings if f.status == "FAIL" and f.check in split_tests})
    executed = any(e["applicable"] for n in split_tests for e in results[n])
    if violated:
        findings.append(_f("split_compliance", "FAIL", f"candidate used rows outside the supplied train_ids: {violated}", classification=PROVEN))
    elif executed:
        findings.append(_f("split_compliance", "PASS", "no forbidden-row poisoning changed validation predictions; supplied train_ids were respected"))
    else:
        findings.append(_f("split_compliance", "UNVERIFIED", "no forbidden-row poisoning could be executed"))
    if not any(r.get("deterministic") for r in coverage.per_fold):
        findings.append(_f("ml_leakage", "UNVERIFIED", "no fold completed a deterministic baseline; poisoning produced no evidence"))
    return findings, coverage, predictions, runner.accessed_ids


# ---------------------------------------------------------------------------
# ML-specific heuristic static scan (does not modify verifier.static_scan)
# ---------------------------------------------------------------------------
SHUFFLE_SPLITTERS = {"ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit"}
KFOLD_SPLITTERS = {
    "KFold", "StratifiedKFold", "RepeatedKFold", "RepeatedStratifiedKFold", "GroupKFold",
    "StratifiedGroupKFold", "LeaveOneOut", "LeavePOut", "LeaveOneGroupOut",
}
DEFAULT_CV_CALLS = {
    "cross_val_score", "cross_val_predict", "cross_validate", "GridSearchCV", "RandomizedSearchCV",
    "HalvingGridSearchCV", "HalvingRandomSearchCV", "CalibratedClassifierCV", "RidgeCV", "LassoCV",
    "ElasticNetCV", "LogisticRegressionCV",
}
FIT_METHODS = {"fit", "fit_transform", "partial_fit", "fit_predict", "fit_resample"}
STAT_METHODS = {"mean", "std", "var", "min", "max", "median", "quantile", "rank", "describe", "corr", "cov"}
TABLE_PRESERVING = {
    "values", "to_numpy", "drop", "select_dtypes", "fillna", "astype", "filter", "copy",
    "set_index", "reset_index", "dropna", "replace", "clip", "abs", "T",
}
ROW_RESTRICTING = {"loc", "iloc", "query", "isin", "merge", "join", "head", "tail", "sample", "take", "xs"}


def _callee(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _kw(node: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in node.keywords if k.arg == name), None)


class _MLVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.risks: list[tuple[str, int, str, str]] = []

    def add(self, node: ast.AST, name: str, classification: str, message: str) -> None:
        self.risks.append((name, getattr(node, "lineno", 0), classification, message))

    def visit_Call(self, node: ast.Call) -> None:
        called = _callee(node.func)
        if called == "train_test_split":
            shuffle = _kw(node, "shuffle")
            if not (isinstance(shuffle, ast.Constant) and shuffle.value is False):
                self.add(node, "ml_shuffled_split", HIGH, "train_test_split shuffles by default; chronological research requires shuffle=False and verifier-owned folds")
        if called in SHUFFLE_SPLITTERS:
            self.add(node, "ml_shuffled_split", HIGH, f"{called} draws random, nonchronological splits")
        if called in KFOLD_SPLITTERS:
            shuffle = _kw(node, "shuffle")
            if isinstance(shuffle, ast.Constant) and shuffle.value is True:
                self.add(node, "ml_shuffled_kfold", HIGH, f"{called}(shuffle=True) mixes future and past observations")
            else:
                self.add(node, "ml_nonchronological_kfold", HIGH, f"{called} trains on blocks after the validation block; not a walk-forward split")
        if called in DEFAULT_CV_CALLS:
            cv = _kw(node, "cv")
            if cv is None or (isinstance(cv, ast.Constant) and isinstance(cv.value, int)):
                self.add(node, "ml_default_cv", HIGH, f"{called} without an explicit chronological cv splitter uses non-chronological K-fold")
        if called == "TimeSeriesSplit" and _kw(node, "gap") is None:
            self.add(node, "ml_timeseriessplit_no_gap", INFO, "TimeSeriesSplit without gap does not purge overlapping label windows")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == MODEL_FUNCTION:
            self._scan_model_function(node)
        self.generic_visit(node)

    def _scan_model_function(self, fn: ast.FunctionDef) -> None:
        params = {a.arg for a in fn.args.args[:2]}  # (features, targets) per the adapter contract
        history: dict[str, list[tuple[int, bool]]] = {}

        def is_full(expr: ast.AST, line: int) -> bool:
            """True when ``expr`` is (heuristically) the unrestricted features/targets table."""
            if isinstance(expr, ast.Name):
                earlier = [full for lineno, full in history.get(expr.id, []) if lineno < line]
                return earlier[-1] if earlier else expr.id in params
            if isinstance(expr, ast.Attribute):
                if expr.attr in ROW_RESTRICTING:
                    return False
                return expr.attr in TABLE_PRESERVING and is_full(expr.value, line)
            if isinstance(expr, ast.Call):
                return isinstance(expr.func, ast.Attribute) and expr.func.attr in TABLE_PRESERVING and is_full(expr.func.value, line)
            if isinstance(expr, ast.Subscript):
                sl = expr.slice
                column_select = (isinstance(sl, ast.Constant) and isinstance(sl.value, str)) or (
                    isinstance(sl, (ast.List, ast.Tuple))
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in sl.elts)
                ) or (
                    # Conventional column-list names (FEATURE_COLUMNS, cols, feature_names, ...).
                    # Row masks are normally named mask/idx/train_*; those stay "restricted".
                    isinstance(sl, ast.Name) and (sl.id.isupper() or any(t in sl.id.lower() for t in ("col", "feat", "field")))
                )
                return column_select and is_full(expr.value, line)
            return False

        # Straight-line alias tracking in source order (heuristic: branches/loops are not modelled).
        for assign in sorted((n for n in ast.walk(fn) if isinstance(n, ast.Assign)), key=lambda n: (n.lineno, n.col_offset)):
            full = is_full(assign.value, assign.lineno)
            for target in assign.targets:
                if isinstance(target, ast.Name):
                    history.setdefault(target.id, []).append((assign.lineno, full))
        for sub in ast.walk(fn):
            if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
                continue
            method = sub.func.attr
            line = sub.lineno  # strict '<': an assignment on the call's own line happens after the call
            if method in FIT_METHODS and any(is_full(arg, line) for arg in sub.args):
                self.add(sub, "ml_full_table_fit", HIGH,
                         f"{MODEL_FUNCTION}: .{method}() receives the unrestricted features/targets table; fitting must use train_ids rows only")
            if method in STAT_METHODS and is_full(sub.func.value, line):
                self.add(sub, "ml_full_table_statistic", HIGH,
                         f"{MODEL_FUNCTION}: .{method}() over the unrestricted features/targets table leaks validation/future rows into preprocessing")


ML_STATIC_FAMILY = "ml_static_scan"


def scan_ml_source(path: str | Path) -> list[Finding]:
    """ML-specific heuristic warnings (family ``ml_static_scan``).

    Never FAIL and never proof of absence. A WARN means manual review is required; the research
    CLI treats this family as mandatory so a targeted heuristic hit yields INCOMPLETE, not VERIFIED.
    """
    p = Path(path)
    if not p.exists():
        return [Finding("static_ml_source", "UNVERIFIED", f"source path not found: {p}", family=ML_STATIC_FAMILY)]
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*.py") if "__pycache__" not in x.parts)
    findings: list[Finding] = []
    for file in files:
        try:
            tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"), filename=str(file))
        except SyntaxError:
            continue  # verifier.static_scan already reports syntax errors as FAIL
        visitor = _MLVisitor()
        visitor.visit(tree)
        for name, line, classification, message in sorted(set(visitor.risks), key=lambda r: (r[1], r[0])):
            findings.append(Finding(
                f"static_{name}", "WARN" if classification == HIGH else "INFO", f"{file}: line {line}: {message}",
                family=ML_STATIC_FAMILY, classification=classification, evidence={"path": str(file), "line": line},
            ))
    if not findings:
        findings.append(Finding(
            "static_ml_scan", "PASS",
            "no ML-specific heuristic risk patterns found; a passing static scan does not prove absence of leakage",
            family=ML_STATIC_FAMILY, classification=INFO,
        ))
    return findings
