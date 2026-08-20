"""Causal technical features and the next-day direction target.

Every feature at date t is computed from OHLCV with index <= t.
The label at date t is 1 iff close[t+1] > close[t]. That label uses a future
close, which is why the last row is dropped before training — it is never
used as an input feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_direction.config import FEATURE_COLUMNS


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. ewm(alpha=1/period, adjust=False) is the standard recursive form."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append causal technical features. Does not add the target.

    Rolling windows and ewm use only current and past rows. Ratios (close/SMA - 1)
    are used instead of raw moving averages so the logistic regression baseline
    is not dominated by the absolute price level of each name.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be sorted ascending by date")

    out = df.copy()
    close = out["close"]
    volume = out["volume"]

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_21"] = close.pct_change(21)

    for window in (5, 10, 20, 50):
        sma = close.rolling(window, min_periods=window).mean()
        out[f"sma_{window}_ratio"] = close / sma - 1.0

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["ema_12_ratio"] = close / ema_12 - 1.0
    out["ema_26_ratio"] = close / ema_26 - 1.0

    out["rsi_14"] = _rsi(close, 14)

    # MACD in price-relative units so it is comparable across tickers / time.
    macd_line = (ema_12 - ema_26) / close
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    rets = close.pct_change()
    out["vol_20"] = rets.rolling(20, min_periods=20).std()

    vol_sma_20 = volume.rolling(20, min_periods=20).mean()
    vol_std_20 = volume.rolling(20, min_periods=20).std()
    out["volume_ratio_20"] = volume / vol_sma_20
    out["volume_z_20"] = (volume - vol_sma_20) / vol_std_20.replace(0.0, np.nan)

    out["hl_range"] = (out["high"] - out["low"]) / close
    out["atr_14_ratio"] = _atr(out, 14) / close

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    band = (bb_upper - bb_lower).replace(0.0, np.nan)
    out["bb_pct_20"] = (close - bb_lower) / band

    out["dist_high_20"] = close / out["high"].rolling(20, min_periods=20).max() - 1.0
    out["dist_low_20"] = close / out["low"].rolling(20, min_periods=20).min() - 1.0

    out["dow"] = out.index.dayofweek.astype(float)
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary next-day direction. 1 = next close is strictly higher."""
    out = df.copy()
    next_close = out["close"].shift(-1)
    up = next_close > out["close"]
    out["target"] = up.astype("float")
    # shift(-1) leaves NaN on the last row; (NaN > x) is False, so restore NaN
    # so prepare_dataset() drops that row instead of keeping a fake "down" label.
    out.loc[next_close.isna(), "target"] = np.nan
    out["next_return"] = next_close / out["close"] - 1.0
    return out


def prepare_dataset(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Features + target, dropping indicator warmup and the last row (no label)."""
    featured = add_features(ohlcv)
    labeled = add_target(featured)
    cols = list(FEATURE_COLUMNS) + ["target", "next_return", "close"]
    clean = labeled[cols].replace([np.inf, -np.inf], np.nan).dropna()
    return clean
