"""Download and cache daily OHLCV from yfinance.

Cached CSVs under data/raw are a speed convenience, not the source of truth.
Pass refresh=True (or --refresh on the CLI) to pull a fresh copy.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_direction.config import (
    DATA_RAW_DIR,
    TICKERS,
    YF_INTERVAL,
    YF_PERIOD,
    YF_RETRIES,
    YF_RETRY_BASE_SLEEP,
)


def _flatten_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance output to a single-level OHLCV frame indexed by date."""
    if df is None or df.empty:
        raise ValueError(f"No price data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return (Price, Ticker) or (Ticker, Price)
        level0 = {str(x).lower() for x in df.columns.get_level_values(0)}
        ohlcv_names = {"open", "high", "low", "close", "adj close", "volume"}
        if level0 & ohlcv_names:
            df.columns = [str(col[0]) for col in df.columns]
        else:
            df.columns = [str(col[1]) for col in df.columns]

    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}; got {list(df.columns)}")

    df = df[required].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"])
    return df


def download_ticker(
    ticker: str,
    *,
    period: str = YF_PERIOD,
    interval: str = YF_INTERVAL,
    retries: int = YF_RETRIES,
) -> pd.DataFrame:
    """Fetch daily bars with retries. Uses auto_adjust=True so splits/dividends
    are already in the OHLC series we train on.
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            return _flatten_ohlcv(raw, ticker)
        except Exception as exc:  # noqa: BLE001 — yfinance raises a mix of types
            last_error = exc
            sleep_s = YF_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            print(f"  {ticker}: download failed (attempt {attempt}/{retries}): {exc}")
            if attempt < retries:
                print(f"  retrying in {sleep_s:.0f}s...")
                time.sleep(sleep_s)
    raise RuntimeError(f"Failed to download {ticker} after {retries} attempts") from last_error


def cache_path(ticker: str, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or DATA_RAW_DIR
    return cache_dir / f"{ticker.upper()}.csv"


def load_ticker(
    ticker: str,
    *,
    refresh: bool = False,
    cache_dir: Path | None = None,
    period: str = YF_PERIOD,
    interval: str = YF_INTERVAL,
) -> pd.DataFrame:
    """Load one ticker from cache, or download and cache it."""
    path = cache_path(ticker, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.sort_index()
        return df

    df = download_ticker(ticker, period=period, interval=interval)
    df.to_csv(path, index=True)
    return df


def load_all(
    tickers: tuple[str, ...] = TICKERS,
    *,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        print(f"Loading {ticker} (refresh={refresh})...")
        frames[ticker] = load_ticker(ticker, refresh=refresh, cache_dir=cache_dir)
        print(f"  {ticker}: {len(frames[ticker])} rows  {frames[ticker].index.min().date()} → {frames[ticker].index.max().date()}")
    return frames
