"""Toy research candidates for verifier self-tests (NumPy/Pandas only; no ML dependencies).

``CleanToy`` is causal and respects the supplied folds. Every other class overrides exactly one
behaviour of ``CleanToy`` to commit one specific, deliberate violation. A verifier that only
passes clean code has not demonstrated anything; each cheat must be caught for its own reason.

Information clock: a bar stamped ``s`` (its OPEN time) is fully known only at
``s + bars.attrs["bar_interval"]``. The verifier supplies ``bar_interval``; toys never guess it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HORIZON = 10
TARGET = "forward_return_10"
FEATURE_COLUMNS = ["ret_5", "vol_20", "dist_ma_20"]
BAR = pd.Timedelta("1D")
HORIZON_SPEC = f"{HORIZON}bars"  # declared externally by the test, exactly as the CLI's --target-horizon


def make_bars(start: str = "2015-01-01", end: str = "2020-12-31", seed: int = 7) -> pd.DataFrame:
    """Deterministic daily bars (open-stamped 21:00 UTC business days) with valid OHLC geometry."""
    index = pd.bdate_range(start, end, tz="UTC") + pd.Timedelta(hours=21)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(index))))
    open_ = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
        "close": close, "volume": np.ones(len(index)),
    }, index=index)
    bars.attrs["bar_interval"] = BAR
    return bars


def interval(bars: pd.DataFrame) -> pd.Timedelta:
    value = bars.attrs.get("bar_interval")
    if value is None:
        raise ValueError("bars.attrs['bar_interval'] missing: the verifier must declare bar availability")
    return pd.Timedelta(value)


def _events(bars: pd.DataFrame, mask: np.ndarray, lag: pd.Timedelta) -> pd.DataFrame:
    """Events on masked bars; event_time = bar open + lag (lag = bar_interval is the legal choice)."""
    close = bars["close"]
    ma = close.rolling(20).mean()
    times = bars.index[mask]
    return pd.DataFrame({
        "event_id": [f"E{t:%Y%m%dT%H%M}" for t in times],
        "event_time": pd.Series(times + lag),
        "direction": np.where(close.to_numpy()[mask] >= ma.to_numpy()[mask], "long", "short"),
    })


def _signal_positions(bars: pd.DataFrame, events: pd.DataFrame, lag: pd.Timedelta) -> np.ndarray:
    return bars.index.get_indexer(pd.DatetimeIndex(events["event_time"]) - lag)


def _feature_table(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    return pd.DataFrame({
        "ret_5": close / close.shift(5) - 1.0,
        "vol_20": close.pct_change().rolling(20).std(),
        "dist_ma_20": close / close.rolling(20).mean() - 1.0,
    }, index=bars.index)


def _features(bars: pd.DataFrame, events: pd.DataFrame, lag: pd.Timedelta) -> pd.DataFrame:
    rows = _feature_table(bars).iloc[_signal_positions(bars, events, lag)]
    out = pd.DataFrame({"event_id": events["event_id"].to_numpy(), "feature_asof_time": events["event_time"].reset_index(drop=True)})
    for column in FEATURE_COLUMNS:
        out[column] = rows[column].to_numpy()
    return out


def _empty_targets() -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": pd.Series([], dtype=object),
        "target_start": pd.Series(pd.DatetimeIndex([], tz="UTC")),
        "target_end": pd.Series(pd.DatetimeIndex([], tz="UTC")),
        TARGET: pd.Series([], dtype=float), "mfe_10": pd.Series([], dtype=float),
    })


def _targets(bars: pd.DataFrame, events: pd.DataFrame, lag: pd.Timedelta, end_lag: pd.Timedelta) -> pd.DataFrame:
    """Forward window = bars p+1..p+H (all opening at/after the event). Resolved at open(p+H)+end_lag."""
    close = bars["close"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    position = _signal_positions(bars, events, lag)
    ok = position + HORIZON < len(bars)
    p = position[ok]
    if len(p) == 0:  # prefix too short to resolve any label: emit an empty, well-typed table
        return _empty_targets()
    windows = sliding_window_view(high, HORIZON)
    return pd.DataFrame({
        "event_id": events["event_id"].to_numpy()[ok],
        "target_start": pd.Series(bars.index[p + 1]),
        "target_end": pd.Series(bars.index[p + HORIZON] + end_lag),
        TARGET: close[p + HORIZON] / close[p] - 1.0,
        "mfe_10": windows[p + 1].max(axis=1) / close[p] - 1.0,
    })


def _base_mask(bars: pd.DataFrame) -> np.ndarray:
    position = np.arange(len(bars))
    return (position >= 20) & (position % 3 == 0)


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
    """Causal events/features/targets on the availability clock and a ridge fitted on train_ids only."""

    @staticmethod
    def events(bars: pd.DataFrame) -> pd.DataFrame:
        return _events(bars, _base_mask(bars), interval(bars))

    @staticmethod
    def features(bars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        return _features(bars, events, interval(bars))

    @staticmethod
    def targets(bars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        return _targets(bars, events, interval(bars), interval(bars))

    @staticmethod
    def fit_predict_fold(features, targets, train_ids, validation_ids, target_name):
        x = features.set_index("event_id")[FEATURE_COLUMNS]
        y = targets.set_index("event_id")[target_name]
        prediction = _ridge_fit_predict(x.loc[train_ids], y.loc[train_ids], x.loc[validation_ids])
        return pd.DataFrame({"event_id": list(validation_ids), "prediction": prediction})


# --------------------------------------------------------------------------------------
# Bar-availability cheats
# --------------------------------------------------------------------------------------
class EventAtBarStart(CleanToy):
    """Uses the event bar's CLOSE but stamps the event at the bar's OPEN (the 09:30-bar problem).
    Features and targets are internally consistent with that wrong clock."""

    @staticmethod
    def events(bars):
        return _events(bars, _base_mask(bars), pd.Timedelta(0))

    @staticmethod
    def features(bars, events):
        return _features(bars, events, pd.Timedelta(0))

    @staticmethod
    def targets(bars, events):
        return _targets(bars, events, pd.Timedelta(0), interval(bars))


class TargetEndAtBarStart(CleanToy):
    """Declares the label resolved at the OPEN of its last bar, although it uses that bar's close."""

    @staticmethod
    def targets(bars, events):
        return _targets(bars, events, interval(bars), pd.Timedelta(0))


class TargetEndAtFinalBarOpen(CleanToy):
    """Uses the Nth (final) window bar's close, high and low, but declares target_end equal to that
    bar's OPEN timestamp instead of its availability time (open + bar_interval)."""

    @staticmethod
    def targets(bars, events):
        close, high, low = (bars[c].to_numpy(float) for c in ("close", "high", "low"))
        position = _signal_positions(bars, events, interval(bars))
        ok = position + HORIZON < len(bars)
        p = position[ok]
        final = p + HORIZON
        return pd.DataFrame({
            "event_id": events["event_id"].to_numpy()[ok],
            "target_start": events["event_time"].iloc[np.flatnonzero(ok)].reset_index(drop=True),  # = T
            "target_end": pd.Series(bars.index[final]),  # WRONG: bar open, not open + interval
            TARGET: close[final] / close[p] - 1.0,
            "mfe_10": high[final] / close[p] - 1.0,
            "mae_10": low[final] / close[p] - 1.0,
        })


class TargetStartsOnSignalBar(CleanToy):
    """Forward window (MFE) includes the signal bar itself, i.e. price movement before the event."""

    @staticmethod
    def targets(bars, events):
        out = CleanToy.targets(bars, events)
        position = _signal_positions(bars, events.set_index("event_id").loc[out["event_id"]].reset_index(), interval(bars))
        close, high = bars["close"].to_numpy(float), bars["high"].to_numpy(float)
        out["target_start"] = pd.Series(bars.index[position])
        out["mfe_10"] = sliding_window_view(high, HORIZON)[position].max(axis=1) / close[position] - 1.0
        return out


# --------------------------------------------------------------------------------------
# Outcome-dependent sample selection cheats
# --------------------------------------------------------------------------------------
class OutcomeSelectedTargets(CleanToy):
    """Emits labels only for events whose forward return turned out positive."""

    @staticmethod
    def targets(bars, events):
        out = CleanToy.targets(bars, events)
        return out[out[TARGET] > 0].reset_index(drop=True)


class TargetBeyondDeclaredHorizon(CleanToy):
    """Labels use an 11-bar window while the frozen declaration is 10 bars."""

    @staticmethod
    def targets(bars, events):
        close = bars["close"].to_numpy(float)
        position = _signal_positions(bars, events, interval(bars))
        ok = position + HORIZON + 1 < len(bars)
        p = position[ok]
        return pd.DataFrame({
            "event_id": events["event_id"].to_numpy()[ok],
            "target_start": pd.Series(bars.index[p + 1]),
            "target_end": pd.Series(bars.index[p + HORIZON + 1] + interval(bars)),
            TARGET: close[p + HORIZON + 1] / close[p] - 1.0,
            "mfe_10": np.zeros(len(p)),
        })


class OutcomeNaNTargets(CleanToy):
    """Blanks the label (NaN) for events whose forward return turned out negative."""

    @staticmethod
    def targets(bars, events):
        out = CleanToy.targets(bars, events)
        out.loc[out[TARGET] < 0, TARGET] = np.nan
        return out


# --------------------------------------------------------------------------------------
# Broken feature / event / target generators (research_causality or research_contract)
# --------------------------------------------------------------------------------------
class GlobalMeanFeature(CleanToy):
    """close / close.mean() where the mean spans the ENTIRE dataset (future data)."""

    @staticmethod
    def features(bars, events):
        out = CleanToy.features(bars, events)
        normalized = (bars["close"] / bars["close"].mean()).to_numpy()
        out["close_norm"] = normalized[_signal_positions(bars, events, interval(bars))]
        return out


class OneBarLookaheadEvents(CleanToy):
    """An event on bar t exists only if close[t+1] > close[t]."""

    @staticmethod
    def events(bars):
        next_up = (bars["close"].shift(-1) > bars["close"]).to_numpy()
        return _events(bars, _base_mask(bars) & next_up, interval(bars))


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
    """Declares the window ends with bar t+10 but the label uses close[t+11]."""

    @staticmethod
    def targets(bars, events):
        out = CleanToy.targets(bars, events)
        close = bars["close"].to_numpy()
        position = _signal_positions(bars, events.set_index("event_id").loc[out["event_id"]].reset_index(), interval(bars))
        peek = np.minimum(position + HORIZON + 1, len(bars) - 1)
        out[TARGET] = close[peek] / close[position] - 1.0
        return out


class FeatureAsofLies(CleanToy):
    """Declares feature_asof_time five bars before the event but uses the event bar's close."""

    @staticmethod
    def features(bars, events):
        out = CleanToy.features(bars, events)
        position = _signal_positions(bars, events, interval(bars))
        out["feature_asof_time"] = pd.Series(bars.index[np.maximum(position - 5, 0)] + interval(bars))
        return out


class FeatureAsofAfterEvent(CleanToy):
    """feature_asof_time is one bar AFTER event_time."""

    @staticmethod
    def features(bars, events):
        out = CleanToy.features(bars, events)
        out["feature_asof_time"] = events["event_time"].reset_index(drop=True) + interval(bars)
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


def audit(name: str, mode: str = "fast", lockbox: str | None = LOCKBOX, horizon: str | None = HORIZON_SPEC):
    """Run the full research audit on toy ``name`` once per (name, mode, lockbox, horizon)."""
    from verifier.research_orchestrator import run_research_audit

    key = (name, mode, lockbox, horizon)
    if key not in _CACHE:
        toy = globals()[name]
        _CACHE[key] = run_research_audit(
            lambda: toy, make_bars(), target_name=TARGET, lockbox_start=lockbox, mode=mode, bar_interval=BAR,
            target_horizon=horizon,
        )
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
