"""Lookahead tests. These are the tests that actually matter for this project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from stock_direction.config import FEATURE_COLUMNS, N_SPLITS
from stock_direction.features import add_features, add_target, prepare_dataset


def test_truncating_future_does_not_change_past_features(synthetic_ohlcv: pd.DataFrame) -> None:
    """Features at t must be identical whether or not rows after t exist.

    Compute on the full series vs. the series with the last 15 days chopped off.
    Every overlapping date should match exactly for rolling/ewm causal features.
    """
    full = add_features(synthetic_ohlcv)
    chopped = add_features(synthetic_ohlcv.iloc[:-15])
    overlap = chopped.index.intersection(full.index)
    cols = list(FEATURE_COLUMNS)
    left = full.loc[overlap, cols]
    right = chopped.loc[overlap, cols]
    # EMA state is recursive from the *start*, which both series share, so this
    # should be exact. Use a tight tolerance for float drift.
    diff = (left - right).abs()
    assert (diff.max() < 1e-12).all(), diff.max().to_dict()


def test_feature_at_t_ignores_close_t_plus_1(synthetic_ohlcv: pd.DataFrame) -> None:
    """Perturb the last close; features on earlier rows must not move."""
    baseline = add_features(synthetic_ohlcv)
    perturbed = synthetic_ohlcv.copy()
    perturbed.iloc[-1, perturbed.columns.get_loc("close")] *= 1.25
    perturbed.iloc[-1, perturbed.columns.get_loc("high")] *= 1.25
    other = add_features(perturbed)
    cols = list(FEATURE_COLUMNS)
    # Compare everything except the last row.
    left = baseline.iloc[:-1][cols]
    right = other.iloc[:-1][cols]
    diff = (left - right).abs()
    assert (diff.max() < 1e-12).all(), diff.max().to_dict()


def test_target_uses_next_close_but_is_not_a_feature(synthetic_ohlcv: pd.DataFrame) -> None:
    labeled = add_target(add_features(synthetic_ohlcv))
    assert "target" not in FEATURE_COLUMNS
    assert "next_return" not in FEATURE_COLUMNS
    # Sanity: flipping tomorrow's close flips today's label, not today's features.
    i = 80
    original_label = labeled["target"].iloc[i]
    tweaked = synthetic_ohlcv.copy()
    tweaked.iloc[i + 1, tweaked.columns.get_loc("close")] = (
        tweaked["close"].iloc[i] * 0.5
        if original_label == 1.0
        else tweaked["close"].iloc[i] * 2.0
    )
    relabeled = add_target(tweaked)
    assert relabeled["target"].iloc[i] != original_label
    feats_a = add_features(synthetic_ohlcv).iloc[i][list(FEATURE_COLUMNS)]
    feats_b = add_features(tweaked).iloc[i][list(FEATURE_COLUMNS)]
    assert np.allclose(feats_a.to_numpy(dtype=float), feats_b.to_numpy(dtype=float), equal_nan=True)


def test_time_series_split_is_forward_only(synthetic_ohlcv: pd.DataFrame) -> None:
    data = prepare_dataset(synthetic_ohlcv)
    x = np.arange(len(data))
    tscv = TimeSeriesSplit(n_splits=min(N_SPLITS, 4))
    prev_test_end = -1
    for train_idx, test_idx in tscv.split(x):
        assert train_idx.max() < test_idx.min()
        assert train_idx.min() == 0  # expanding window, not sliding
        assert test_idx.min() > prev_test_end or prev_test_end == -1
        prev_test_end = test_idx.max()
        train_dates = data.index[train_idx]
        test_dates = data.index[test_idx]
        assert train_dates.max() < test_dates.min()


def test_holdout_is_strictly_after_dev(synthetic_ohlcv: pd.DataFrame) -> None:
    data = prepare_dataset(synthetic_ohlcv)
    holdout_days = 30
    dev = data.iloc[:-holdout_days]
    holdout = data.iloc[-holdout_days:]
    assert dev.index.max() < holdout.index.min()
    assert list(data.index) == list(dev.index) + list(holdout.index)
