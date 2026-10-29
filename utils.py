"""Helpers for the Stock Price Predictor app.

* input validation
* loading the saved model (robust to scikit-learn version changes)
* downloading live market data with yfinance
* training a fresh per-ticker model with an honest, chronological test split
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
PARAMS_PATH = BASE_DIR / "model_params.json"

FEATURES = ["Open", "High", "Low", "Volume"]
TICKER_RE = re.compile(r"^[A-Za-z0-9.\-=^]{1,15}$")
CACHE_TTL_SECONDS = 60 * 60
MIN_TRAINING_ROWS = 60

# Sample values used by the "Sample" button (medians from the original dataset)
DEFAULTS = {"Open": 1292.10, "High": 1296.30, "Low": 1286.70, "Volume": 140}


class ValidationError(ValueError):
    """The user supplied unusable input."""


class DataUnavailable(RuntimeError):
    """Live market data could not be fetched or was unusable."""


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LinearModel:
    """Tiny linear model: close = coef . [open, high, low, volume] + intercept.

    Using plain numpy means predictions never depend on the scikit-learn
    version that created the pickle (and never emit feature-name warnings).
    """

    def __init__(self, coef, intercept: float):
        self.coef = np.asarray(coef, dtype=float).reshape(-1)
        self.intercept = float(intercept)
        if self.coef.shape[0] != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} coefficients, got {self.coef.shape[0]}")

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X @ self.coef + self.intercept

    def to_dict(self) -> dict:
        return {"features": FEATURES, "coef": self.coef.tolist(), "intercept": self.intercept}


_default_model: LinearModel | None = None
_default_lock = threading.Lock()


def load_default_model() -> LinearModel:
    """Load model.pkl (Ridge regression); fall back to model_params.json."""
    global _default_model
    with _default_lock:
        if _default_model is not None:
            return _default_model

        model = None
        if MODEL_PATH.exists():
            try:
                import joblib

                with warnings.catch_warnings():
                    # pickle was written by scikit-learn 1.0.1; we only read coef_/intercept_
                    warnings.simplefilter("ignore")
                    ridge = joblib.load(MODEL_PATH)
                model = LinearModel(ridge.coef_, ridge.intercept_)
            except Exception:  # corrupted / incompatible pickle
                model = None

        if model is None and PARAMS_PATH.exists():
            params = json.loads(PARAMS_PATH.read_text())
            model = LinearModel(params["coef"], params["intercept"])

        if model is None:
            raise FileNotFoundError(
                "Could not load model.pkl or model_params.json. Run `python train_model.py` first."
            )
        _default_model = model
        return model


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _to_float(raw, label: str) -> float:
    if raw is None or str(raw).strip() == "":
        raise ValidationError(f"{label} is required.")
    try:
        value = float(str(raw).replace(",", "").strip())
    except ValueError:
        raise ValidationError(f"{label} must be a number (got '{raw}').") from None
    if not math.isfinite(value):
        raise ValidationError(f"{label} must be a finite number.")
    return value


def parse_inputs(open_, high, low, volume) -> dict:
    """Validate raw form/JSON values and return clean floats."""
    o = _to_float(open_, "Open")
    h = _to_float(high, "High")
    l = _to_float(low, "Low")
    v = _to_float(volume, "Volume")

    for label, val in (("Open", o), ("High", h), ("Low", l)):
        if val <= 0:
            raise ValidationError(f"{label} price must be greater than 0.")
    if v < 0:
        raise ValidationError("Volume cannot be negative.")
    if h < l:
        raise ValidationError("High must be greater than or equal to Low.")
    if not (l <= o <= h):
        raise ValidationError("Open must lie between Low and High.")
    return {"Open": o, "High": h, "Low": l, "Volume": v}


def normalize_ticker(ticker) -> str:
    ticker = (ticker or "").strip().upper()
    if not TICKER_RE.match(ticker):
        raise ValidationError(
            "Enter a valid ticker symbol, e.g. AAPL, MSFT, TCS.NS, RELIANCE.NS, GC=F."
        )
    return ticker


# --------------------------------------------------------------------------- #
# Live data + per-ticker training
# --------------------------------------------------------------------------- #
def fetch_history(ticker: str, years: int = 3) -> pd.DataFrame:
    """Download daily OHLCV data. Raises DataUnavailable on any problem."""
    ticker = normalize_ticker(ticker)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataUnavailable("The 'yfinance' package is not installed.") from exc

    try:
        df = yf.download(
            ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False
        )
    except Exception as exc:  # network errors, rate limits, ...
        raise DataUnavailable(f"Could not download data for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise DataUnavailable(f"No market data found for '{ticker}'. Check the symbol.")

    # Newer yfinance versions return MultiIndex columns like ('Close', 'AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataUnavailable(f"Data for '{ticker}' is missing columns: {', '.join(missing)}.")

    df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    if len(df) < MIN_TRAINING_ROWS:
        raise DataUnavailable(
            f"Only {len(df)} usable rows for '{ticker}'; need at least {MIN_TRAINING_ROWS}."
        )
    return df


def train_model(df: pd.DataFrame) -> tuple[LinearModel, dict]:
    """Fit Ridge on the first 80 % of days, evaluate on the *later* 20 %.

    A chronological split avoids the look-ahead leakage that a random
    train_test_split causes on time-series data.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["Close"].to_numpy(dtype=float)
    split = int(len(df) * 0.8)

    ridge = Ridge(alpha=1.0).fit(X[:split], y[:split])
    pred = ridge.predict(X[split:])
    metrics = {
        "mae": float(mean_absolute_error(y[split:], pred)),
        "rmse": float(math.sqrt(mean_squared_error(y[split:], pred))),
        "r2": float(r2_score(y[split:], pred)),
        "train_rows": int(split),
        "test_rows": int(len(df) - split),
    }

    # Final model uses all available data
    final = Ridge(alpha=1.0).fit(X, y)
    return LinearModel(final.coef_, final.intercept_), metrics


_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def get_ticker_bundle(ticker: str) -> dict:
    """Return {ticker, model, metrics, history df, latest row}, cached for an hour."""
    ticker = normalize_ticker(ticker)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(ticker)
        if hit and now - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]

    df = fetch_history(ticker)
    model, metrics = train_model(df)
    last = df.iloc[-1]
    bundle = {
        "ticker": ticker,
        "model": model,
        "metrics": metrics,
        "history": df,
        "latest": {
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "Open": float(last["Open"]),
            "High": float(last["High"]),
            "Low": float(last["Low"]),
            "Close": float(last["Close"]),
            "Volume": float(last["Volume"]),
        },
    }
    with _cache_lock:
        _cache[ticker] = (now, bundle)
    return bundle


def history_for_chart(df: pd.DataFrame, days: int = 120) -> dict:
    tail = df.tail(days)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in tail.index],
        "close": [round(float(v), 2) for v in tail["Close"]],
    }


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_close(values: dict, model: LinearModel | None = None) -> float:
    model = model or load_default_model()
    row = [[values[f] for f in FEATURES]]
    pred = float(model.predict(row)[0])
    if not math.isfinite(pred):
        raise ValidationError("The model produced an invalid result for these inputs.")
    return pred


def preprocess(Open, High, Low, Volume):
    """Backward-compatible wrapper for the original project API."""
    return predict_close(parse_inputs(Open, High, Low, Volume))
