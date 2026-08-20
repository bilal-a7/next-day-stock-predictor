from __future__ import annotations

import pandas as pd
import pytest

from stock_direction.data import _flatten_ohlcv


def test_flatten_single_level_columns() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.1, 1.2],
            "High": [1.2, 1.3, 1.4],
            "Low": [0.9, 1.0, 1.1],
            "Close": [1.1, 1.2, 1.25],
            "Volume": [100, 110, 120],
        },
        index=idx,
    )
    out = _flatten_ohlcv(raw, "SPY")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.is_monotonic_increasing
    assert out.index.name == "date"


def test_flatten_multiindex_price_first() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
    )
    raw = pd.DataFrame(
        [[1, 1, 1, 1, 10], [2, 2, 2, 2, 20], [3, 3, 3, 3, 30]],
        index=idx,
        columns=cols,
    )
    out = _flatten_ohlcv(raw, "AAPL")
    assert out["close"].tolist() == [1, 2, 3]


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        _flatten_ohlcv(pd.DataFrame(), "SPY")
