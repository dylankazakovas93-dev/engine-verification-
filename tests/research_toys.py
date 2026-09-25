"""Toy research candidates for verifier self-tests (NumPy/Pandas only; no ML dependencies).

``CleanToy`` is causal and respects the supplied folds. Every other class overrides exactly one
behaviour of ``CleanToy`` to commit one specific, deliberate violation. A verifier that only
passes clean code has not demonstrated anything; each cheat must be caught for its own reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HORIZON = 10
TARGET = "forward_return_10"
FEATURE_COLUMNS = ["ret_5", "vol_20", "dist_ma_20"]


def make_bars(start: str = "2015-01-01", end: str = "2020-12-31", seed: int = 7) -> pd.DataFrame:
    """Deterministic daily-close bars (21:00 UTC business days) with valid OHLC geometry."""
    index = pd.bdate_range(start, end, tz="UTC") + pd.Timedelta(hours=21)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(index))))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
        "close": close, "volume": np.ones(len(index)),
    }, index=index)


def _event_frame(bars: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    close = bars["close"]
    ma = close.rolling(20).mean()
    times = bars.index[mask]
    return pd.DataFrame({
        "event_id": [f"E{t:%Y%m%dT%H%M}" for t in times],
        "event_time": pd.Series(times),
        "direction": np.where(close.to_numpy()[mask] >= ma.to_numpy()[mask], "long", "short"),
    })


def _ridge_fit_predict(train_x: pd.DataFrame, train_y: pd.Series, predict_x: pd.DataFrame, ridge: float = 1.0,
                       standardize: bool = True) -> np.ndarray:
    """Ridge with an unpenalized intercept; standardization statistics come from ``train_x`` only."""
    ok = np.isfinite(train_y.to_numpy(float)) & np.isfinite(train_x.to_numpy(float)).all(axis=1)
    x, y = train_x.to_numpy(float)[ok], train_y.to_numpy(float)[ok]
    mu, sd = (x.mean(axis=0), x.std(axis=0)) if standardize else (np.zeros(x.shape[1]), np.ones(x.shape[1]))
    sd[sd == 0] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    z = np.nan_to_num((predict_x.to_numpy(float) - mu) / sd)
    return np.column_stack([np.ones(len(z)), z]) @ beta


class CleanToy:
    """Causal events/features/targets and a ridge model fitted on train_ids only."""

    @staticmethod
    def events(bars: pd.DataFrame) -> pd.DataFrame:
        position = np.arange(len(bars))
        return _event_frame(bars, (position >= 20) & (position % 3 == 0))

    @staticmethod
    def features(bars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        close = bars["close"]
        returns = close.pct_change()
        table = pd.DataFrame({
            "ret_5": close / close.shift(5) - 1.0,
            "vol_20": returns.rolling(20).std(),
            "dist_ma_20": close / close.rolling(20).mean() - 1.0,
        }, index=bars.index)
        rows = table.reindex(pd.DatetimeIndex(events["event_time"]))
        out = pd.DataFrame({"event_id": events["event_id"].to_numpy(), "feature_asof_time": events["event_time"].reset_index(drop=True)})
        for column in FEATURE_COLUMNS:
            out[column] = rows[column].to_numpy()
        return out

    @staticmethod
    def targets(bars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        close = bars["close"].to_numpy(float)
        high = bars["high"].to_numpy(float)
        position = bars.index.get_indexer(pd.DatetimeIndex(events["event_time"]))
        ok = position + HORIZON < len(bars)
        p = position[ok]
        if len(p) == 0:  # prefix too short to resolve any label: emit an empty, well-typed table
            return pd.DataFrame({
                "event_id": pd.Series([], dtype=object),
                "target_start": pd.Series(pd.DatetimeIndex([], tz="UTC")),
                "target_end": pd.Series(pd.DatetimeIndex([], tz="UTC")),
                TARGET: pd.Series([], dtype=float), "mfe_10": pd.Series([], dtype=float),
            })
        windows = sliding_window_view(high, HORIZON)
        return pd.DataFrame({
            "event_id": events["event_id"].to_numpy()[ok],
            "target_start": pd.Series(bars.index[p + 1]),
            "target_end": pd.Series(bars.index[p + HORIZON]),
            TARGET: close[p + HORIZON] / close[p] - 1.0,
            "mfe_10": windows[p + 1].max(axis=1) / close[p] - 1.0,
        })

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        x = features.set_index("event_id")[FEATURE_COLUMNS]
        y = targets.set_index("event_id")[target_name]
        prediction = _ridge_fit_predict(x.loc[train_ids], y.loc[train_ids], x.loc[validation_ids])
        return pd.DataFrame({"event_id": list(validation_ids), "prediction": prediction})


# --------------------------------------------------------------------------------------
# Broken feature / event / target generators (research_causality or research_contract)
# --------------------------------------------------------------------------------------
class GlobalMeanFeature(CleanToy):
    """close / close.mean() where the mean spans the ENTIRE dataset (future data)."""

    @staticmethod
    def features(bars, events):
        out = CleanToy.features(bars, events)
        normalized = (bars["close"] / bars["close"].mean()).reindex(pd.DatetimeIndex(events["event_time"]))
        out["close_norm"] = normalized.to_numpy()
        return out


class OneBarLookaheadEvents(CleanToy):
    """An event at t exists only if close[t+1] > close[t]."""

    @staticmethod
    def events(bars):
        position = np.arange(len(bars))
        next_up = (bars["close"].shift(-1) > bars["close"]).to_numpy()
        return _event_frame(bars, (position >= 20) & (position % 3 == 0) & next_up)


class TargetCopiedNamed(CleanToy):
    """The label is copied into the feature frame under its own reserved name."""

    @staticmethod
    def features(bars, events):
        out = CleanToy.features(bars, events)
        labels = CleanToy.targets(bars, events).set_index("event_id")[TARGET]
        out[TARGET] = out["event_id"].map(labels).to_numpy()
        return out


class TargetCopiedDisguised(CleanToy):
    """The label is copied into the feature frame under an innocent-looking name."""

    @staticmethod
    def features(bars, events):
        out = CleanToy.features(bars, events)
        labels = CleanToy.targets(bars, events).set_index("event_id")[TARGET]
        out["momentum_x"] = out["event_id"].map(labels).to_numpy()
        return out


class TargetPeeksBeyondEnd(CleanToy):
    """Declares target_end = t+10 but the label uses close[t+11]."""

    @staticmethod
    def targets(bars, events):
        out = CleanToy.targets(bars, events)
        close = bars["close"]
        start = pd.DatetimeIndex(events.set_index("event_id").loc[out["event_id"], "event_time"])
        position = bars.index.get_indexer(start)
        peek = np.minimum(position + HORIZON + 1, len(bars) - 1)
        out[TARGET] = close.to_numpy()[peek] / close.to_numpy()[position] - 1.0
        return out


class FeatureAsofLies(CleanToy):
    """Declares feature_asof_time five bars before the event but uses the event bar's close."""

    @staticmethod
    def features(bars, events):
        out = CleanToy.features(bars, events)
        position = bars.index.get_indexer(pd.DatetimeIndex(events["event_time"]))
        out["feature_asof_time"] = pd.Series(bars.index[np.maximum(position - 5, 0)])
        return out


class FeatureAsofAfterEvent(CleanToy):
    """feature_asof_time is one bar AFTER event_time."""

    @staticmethod
    def features(bars, events):
        out = CleanToy.features(bars, events)
        position = bars.index.get_indexer(pd.DatetimeIndex(events["event_time"]))
        out["feature_asof_time"] = pd.Series(bars.index[np.minimum(position + 1, len(bars) - 1)])
        return out


# --------------------------------------------------------------------------------------
# Broken models (ml_leakage)
# --------------------------------------------------------------------------------------
class ValidationLabelLeak(CleanToy):
    """Shifts predictions by the mean of the VALIDATION labels (e.g. 'calibration')."""

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        out = CleanToy.fit_predict_fold(features, targets, train_ids, validation_ids, target_name)
        validation_mean = targets.set_index("event_id").loc[validation_ids, target_name].mean()
        out["prediction"] = out["prediction"] + validation_mean
        return out


class FullSampleScaling(CleanToy):
    """Standardizes with mean/std over ALL feature rows (validation + future) before selecting
    train_ids, then fits the same ridge as CleanToy. Ridge is not scale-invariant, so the leaked
    statistics change predictions. (With plain OLS + intercept the same full-sample scaling is
    mathematically output-invariant and therefore undetectable by any output-based test.)"""

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        x = features.set_index("event_id")[FEATURE_COLUMNS]
        x = (x - x.mean()) / x.std()
        y = targets.set_index("event_id")[target_name]
        prediction = _ridge_fit_predict(x.loc[train_ids], y.loc[train_ids], x.loc[validation_ids], standardize=False)
        return pd.DataFrame({"event_id": list(validation_ids), "prediction": prediction})


class FullSampleScalingOLS(CleanToy):
    """Documented limitation: full-sample scaling feeding OLS-with-intercept. Predictions are
    exactly invariant to the leaked statistics, so dynamic poisoning cannot (and need not) flag
    the outputs; only the heuristic static scan can point at the pattern."""

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        x = features.set_index("event_id")[FEATURE_COLUMNS]
        x = (x - x.mean()) / x.std()
        y = targets.set_index("event_id")[target_name]
        design = np.column_stack([np.ones(len(train_ids)), x.loc[train_ids].to_numpy(float)])
        beta = np.linalg.lstsq(design, y.loc[train_ids].to_numpy(float), rcond=None)[0]
        predict_x = x.loc[validation_ids].to_numpy(float)
        return pd.DataFrame({"event_id": list(validation_ids), "prediction": np.column_stack([np.ones(len(predict_x)), predict_x]) @ beta})


class FullSampleLabels(CleanToy):
    """Ignores train_ids and fits on every row with a label (validation and future included)."""

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        x = features.set_index("event_id")[FEATURE_COLUMNS]
        y = targets.set_index("event_id")[target_name]
        prediction = _ridge_fit_predict(x.loc[y.index], y, x.loc[validation_ids])
        return pd.DataFrame({"event_id": list(validation_ids), "prediction": prediction})


class SelfSelectedUnpurged(CleanToy):
    """Ignores train_ids; trains on every row whose event precedes the validation block (no purge)."""

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        asof = features.set_index("event_id")["feature_asof_time"]
        start = asof.loc[validation_ids].min()
        train = asof.index[asof < start].tolist()
        return CleanToy.fit_predict_fold(features, targets, train, validation_ids, target_name)


class RandomSplitModel(CleanToy):
    """Ignores train_ids; trains on a seeded random 70% of all non-validation rows (future included)."""

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        validation = set(validation_ids)
        pool = [i for i in features["event_id"].tolist() if i not in validation]
        rng = np.random.default_rng(0)
        train = sorted(rng.choice(pool, size=int(0.7 * len(pool)), replace=False).tolist())
        return CleanToy.fit_predict_fold(features, targets, train, validation_ids, target_name)


class NondeterministicModel(CleanToy):
    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        out = CleanToy.fit_predict_fold(features, targets, train_ids, validation_ids, target_name)
        out["prediction"] = out["prediction"] + np.random.default_rng().normal(0, 1e-3, len(out))
        return out


class WrongPredictionIds(CleanToy):
    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        out = CleanToy.fit_predict_fold(features, targets, train_ids, validation_ids, target_name)
        return out.iloc[:-1]


class ConstantModel(CleanToy):
    """Leak-free but label-blind: positive controls cannot demonstrate test power."""

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        return pd.DataFrame({"event_id": list(validation_ids), "prediction": 0.0})


class NoModel:
    events = staticmethod(CleanToy.events)
    features = staticmethod(CleanToy.features)
    targets = staticmethod(CleanToy.targets)


class RecordingModel(CleanToy):
    """Clean model that records every event_id it is shown (lockbox access proof)."""
    seen: set = set()

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        RecordingModel.seen.update(features["event_id"].tolist())
        RecordingModel.seen.update(targets["event_id"].tolist())
        return CleanToy.fit_predict_fold(features, targets, train_ids, validation_ids, target_name)


# --------------------------------------------------------------------------------------
# Shared, cached audit runner for the self-tests
# --------------------------------------------------------------------------------------
LOCKBOX = "2020-01-01"
_CACHE: dict = {}


def audit(name: str, mode: str = "fast", lockbox: str | None = LOCKBOX):
    """Run the full research audit on toy ``name`` once per (name, mode, lockbox)."""
    from verifier.research_orchestrator import run_research_audit

    key = (name, mode, lockbox)
    if key not in _CACHE:
        toy = globals()[name]
        _CACHE[key] = run_research_audit(lambda: toy, make_bars(), target_name=TARGET, lockbox_start=lockbox, mode=mode)
    return _CACHE[key]


def statuses(findings, family: str | None = None) -> dict[str, set]:
    """check -> set of statuses (optionally restricted to one family)."""
    out: dict[str, set] = {}
    for f in findings:
        if family is None or f.family == family:
            out.setdefault(f.check, set()).add(f.status)
    return out


def failing_checks(findings, family: str | None = None) -> set:
    return {c for c, s in statuses(findings, family).items() if "FAIL" in s}
