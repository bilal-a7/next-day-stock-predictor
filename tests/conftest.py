"""Shared synthetic OHLCV for leakage and feature tests. No network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 180
    dates = pd.bdate_range("2020-01-02", periods=n)
    # Geometric random walk so prices stay positive and look vaguely like a stock.
    rets = rng.normal(0.0004, 0.015, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    high = close * (1.0 + rng.uniform(0.001, 0.012, size=n))
    low = close * (1.0 - rng.uniform(0.001, 0.012, size=n))
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=n))
    volume = rng.integers(1_000_000, 8_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )
