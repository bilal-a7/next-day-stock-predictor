"""Train per-ticker logistic regression and XGBoost models.

Protocol
--------
1. Download (or cache-read) 5y of daily OHLCV.
2. Build causal features; label = next-day direction.
3. Hold out the last HOLDOUT_DAYS rows. Never touch them until the final report.
4. On the remaining development set, run sklearn TimeSeriesSplit (expanding
   window, N_SPLITS folds). Same folds for both models.
5. Fit final models on the full development set and score the holdout once.

Hyperparameters are fixed. No search is run against the holdout.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from stock_direction.config import (
    FEATURE_COLUMNS,
    HOLDOUT_DAYS,
    MODELS_DIR,
    N_SPLITS,
    RANDOM_STATE,
    REPORTS_DIR,
    TICKERS,
)
from stock_direction.data import load_all
from stock_direction.evaluate import (
    always_up_baseline,
    compute_metrics,
    plot_feature_importance,
    plot_naive_strategy,
    plot_roc,
    train_majority_baseline,
    write_metrics_json,
)
from stock_direction.features import prepare_dataset


def _make_logreg() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _make_xgb(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )


def _xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = frame["target"].to_numpy(dtype=np.int32)
    return x, y


def _mean_std(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[f"{key}_mean"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0)
    return out


METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def walk_forward(dev: pd.DataFrame) -> dict[str, Any]:
    """Expanding-window TimeSeriesSplit on the development set."""
    x, y = _xy(dev)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_rows: dict[str, list[dict[str, float]]] = {
        "logreg": [],
        "xgb": [],
        "majority": [],
        "always_up": [],
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(x), start=1):
        x_tr, y_tr = x[train_idx], y[train_idx]
        x_te, y_te = x[test_idx], y[test_idx]
        pos = float(y_tr.mean())
        spw = (1.0 - pos) / pos if pos not in (0.0, 1.0) else 1.0

        logreg = _make_logreg()
        logreg.fit(x_tr, y_tr)
        lr_proba = logreg.predict_proba(x_te)[:, 1]
        lr_pred = (lr_proba >= 0.5).astype(int)

        xgb = _make_xgb(scale_pos_weight=spw)
        xgb.fit(x_tr, y_tr)
        xgb_proba = xgb.predict_proba(x_te)[:, 1]
        xgb_pred = (xgb_proba >= 0.5).astype(int)

        fold_rows["logreg"].append(compute_metrics(y_te, lr_pred, lr_proba))
        fold_rows["xgb"].append(compute_metrics(y_te, xgb_pred, xgb_proba))
        fold_rows["majority"].append(train_majority_baseline(y_tr, y_te))
        fold_rows["always_up"].append(always_up_baseline(y_te))
        print(
            f"    fold {fold}/{N_SPLITS}  n_train={len(train_idx)} n_test={len(test_idx)}  "
            f"logreg AUC={fold_rows['logreg'][-1]['roc_auc']:.3f}  "
            f"xgb AUC={fold_rows['xgb'][-1]['roc_auc']:.3f}"
        )

    summary = {name: {**_mean_std(rows, METRIC_KEYS), "folds": rows} for name, rows in fold_rows.items()}
    return summary


def fit_final(dev: pd.DataFrame) -> tuple[Pipeline, XGBClassifier]:
    x, y = _xy(dev)
    pos = float(y.mean())
    spw = (1.0 - pos) / pos if pos not in (0.0, 1.0) else 1.0
    logreg = _make_logreg()
    logreg.fit(x, y)
    xgb = _make_xgb(scale_pos_weight=spw)
    xgb.fit(x, y)
    return logreg, xgb


def score_holdout(
    model: Any, holdout: pd.DataFrame
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    x, y = _xy(holdout)
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return compute_metrics(y, pred, proba), pred, proba


def train_ticker(ticker: str, ohlcv: pd.DataFrame) -> dict[str, Any]:
    print(f"\n=== {ticker} ===")
    data = prepare_dataset(ohlcv)
    if len(data) <= HOLDOUT_DAYS + 200:
        raise RuntimeError(f"{ticker}: not enough rows after warmup ({len(data)})")

    dev = data.iloc[:-HOLDOUT_DAYS]
    holdout = data.iloc[-HOLDOUT_DAYS:]
    print(
        f"  rows={len(data)}  dev={len(dev)} "
        f"({dev.index.min().date()} → {dev.index.max().date()})  "
        f"holdout={len(holdout)} ({holdout.index.min().date()} → {holdout.index.max().date()})"
    )
    print(f"  up-rate  dev={dev['target'].mean():.3f}  holdout={holdout['target'].mean():.3f}")

    print("  walk-forward CV...")
    cv = walk_forward(dev)

    print("  fitting final models on development set...")
    logreg, xgb = fit_final(dev)
    lr_metrics, _, lr_proba = score_holdout(logreg, holdout)
    xgb_metrics, _, xgb_proba = score_holdout(xgb, holdout)
    y_hold = holdout["target"].to_numpy(dtype=np.int32)
    majority = train_majority_baseline(dev["target"].to_numpy(dtype=np.int32), y_hold)
    always_up = always_up_baseline(y_hold)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, MODELS_DIR / f"{ticker}_logreg.joblib")
    joblib.dump(xgb, MODELS_DIR / f"{ticker}_xgb.joblib")

    plot_roc(
        [
            ("Logistic regression", y_hold, lr_proba),
            ("XGBoost", y_hold, xgb_proba),
        ],
        title=f"{ticker} holdout ROC (last {HOLDOUT_DAYS} sessions)",
        out_path=REPORTS_DIR / f"roc_{ticker}.png",
    )

    lr_coef = logreg.named_steps["clf"].coef_.ravel()
    plot_feature_importance(
        list(FEATURE_COLUMNS),
        lr_coef,
        title=f"{ticker} logistic regression coefficients",
        out_path=REPORTS_DIR / f"logreg_importance_{ticker}.png",
        xlabel="Coefficient (standardized features)",
    )
    plot_feature_importance(
        list(FEATURE_COLUMNS),
        xgb.feature_importances_.astype(float),
        title=f"{ticker} XGBoost feature importance (gain)",
        out_path=REPORTS_DIR / f"xgb_importance_{ticker}.png",
        xlabel="Relative importance",
    )
    plot_naive_strategy(
        holdout.index,
        holdout["next_return"].to_numpy(dtype=float),
        xgb_proba,
        title=f"{ticker} diagnostic equity — XGBoost (holdout, no costs)",
        out_path=REPORTS_DIR / f"naive_strategy_{ticker}.png",
    )

    result = {
        "ticker": ticker,
        "n_total": int(len(data)),
        "n_dev": int(len(dev)),
        "n_holdout": int(len(holdout)),
        "date_start": str(data.index.min().date()),
        "date_end": str(data.index.max().date()),
        "dev_end": str(dev.index.max().date()),
        "holdout_start": str(holdout.index.min().date()),
        "up_rate_dev": float(dev["target"].mean()),
        "up_rate_holdout": float(holdout["target"].mean()),
        "walk_forward": {
            name: {k: v for k, v in summary.items() if k != "folds"}
            for name, summary in cv.items()
        },
        "holdout": {
            "logreg": lr_metrics,
            "xgb": xgb_metrics,
            "majority": majority,
            "always_up": always_up,
        },
    }
    print(
        f"  holdout  logreg AUC={lr_metrics['roc_auc']:.3f} acc={lr_metrics['accuracy']:.3f}  "
        f"xgb AUC={xgb_metrics['roc_auc']:.3f} acc={xgb_metrics['accuracy']:.3f}  "
        f"always-up acc={always_up['accuracy']:.3f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train next-day direction models")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download OHLCV from yfinance instead of using data/raw cache",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(TICKERS),
        help="Tickers to train (default: SPY AAPL MSFT)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_STATE)
    tickers = tuple(t.upper() for t in args.tickers)
    frames = load_all(tickers, refresh=args.refresh)

    reports = []
    for ticker in tickers:
        reports.append(train_ticker(ticker, frames[ticker]))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "holdout_days": HOLDOUT_DAYS,
            "n_splits": N_SPLITS,
            "split": "sklearn TimeSeriesSplit expanding window on development set",
            "target": "1 if close[t+1] > close[t] else 0",
            "random_state": RANDOM_STATE,
            "features": list(FEATURE_COLUMNS),
        },
        "tickers": reports,
    }
    path = write_metrics_json(payload)
    meta = {
        "feature_columns": list(FEATURE_COLUMNS),
        "tickers": list(tickers),
        "holdout_days": HOLDOUT_DAYS,
        "random_state": RANDOM_STATE,
        "production_model": "xgb",
    }
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {path}")
    print(f"Wrote {MODELS_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
