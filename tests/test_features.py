from __future__ import annotations

import numpy as np
import pandas as pd

from stock_direction.config import FEATURE_COLUMNS
from stock_direction.features import add_features, add_target, prepare_dataset


def test_feature_columns_present(synthetic_ohlcv: pd.DataFrame) -> None:
    out = add_features(synthetic_ohlcv)
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    assert missing == []


def test_rsi_bounds(synthetic_ohlcv: pd.DataFrame) -> None:
    rsi = add_features(synthetic_ohlcv)["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_warmup_nans_then_dropped(synthetic_ohlcv: pd.DataFrame) -> None:
    featured = add_features(synthetic_ohlcv)
    # SMA-50 is the longest hard window; early rows must be NaN.
    assert featured["sma_50_ratio"].iloc[:49].isna().all()
    clean = prepare_dataset(synthetic_ohlcv)
    assert not clean[list(FEATURE_COLUMNS)].isna().any().any()
    assert len(clean) < len(synthetic_ohlcv)


def test_target_alignment(synthetic_ohlcv: pd.DataFrame) -> None:
    labeled = add_target(synthetic_ohlcv)
    # Last row has no next close.
    assert np.isnan(labeled["target"].iloc[-1])
    for i in range(len(labeled) - 1):
        expected = 1.0 if labeled["close"].iloc[i + 1] > labeled["close"].iloc[i] else 0.0
        assert labeled["target"].iloc[i] == expected


def test_prepare_dataset_sorted_and_finite(synthetic_ohlcv: pd.DataFrame) -> None:
    clean = prepare_dataset(synthetic_ohlcv)
    assert clean.index.is_monotonic_increasing
    assert np.isfinite(clean[list(FEATURE_COLUMNS)].to_numpy()).all()
    assert set(clean["target"].unique()).issubset({0.0, 1.0})


def test_rejects_unsorted_index(synthetic_ohlcv: pd.DataFrame) -> None:
    shuffled = synthetic_ohlcv.sample(frac=1.0, random_state=0)
    try:
        add_features(shuffled)
        raise AssertionError("expected ValueError for unsorted index")
    except ValueError:
        pass
