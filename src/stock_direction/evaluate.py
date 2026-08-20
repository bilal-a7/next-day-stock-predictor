"""Metrics, baselines, and report plots. No training lives here."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from stock_direction.config import REPORTS_DIR


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, y_proba),
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "pred_pos_rate": float(np.mean(y_pred)),
    }


def majority_baseline(y_true: np.ndarray) -> dict[str, float]:
    """Predict the majority class observed in y_true (reported as a ceiling-check,
    not as a fitted baseline — the fitted version uses the *train* majority).
    """
    majority = 1 if np.mean(y_true) >= 0.5 else 0
    pred = np.full_like(y_true, majority, dtype=int)
    proba = np.full(len(y_true), float(majority), dtype=float)
    metrics = compute_metrics(y_true, pred, proba)
    metrics["majority_class"] = int(majority)
    return metrics


def always_up_baseline(y_true: np.ndarray) -> dict[str, float]:
    pred = np.ones_like(y_true, dtype=int)
    proba = np.ones(len(y_true), dtype=float)
    return compute_metrics(y_true, pred, proba)


def train_majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    majority = 1 if np.mean(y_train) >= 0.5 else 0
    pred = np.full_like(y_test, majority, dtype=int)
    proba = np.full(len(y_test), float(majority), dtype=float)
    metrics = compute_metrics(y_test, pred, proba)
    metrics["majority_class"] = int(majority)
    return metrics


def plot_roc(
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="chance")
    for name, y_true, y_proba in curves:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}  (AUC={auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_feature_importance(
    names: list[str],
    values: np.ndarray,
    title: str,
    out_path: Path,
    xlabel: str,
) -> None:
    order = np.argsort(np.abs(values))
    names_s = [names[i] for i in order]
    vals_s = values[order]
    colors = ["#4C78A8" if v >= 0 else "#E45756" for v in vals_s]

    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.32 * len(names))))
    ax.barh(names_s, vals_s, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axvline(0, color="#333333", linewidth=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_naive_strategy(
    dates: pd.DatetimeIndex,
    next_returns: np.ndarray,
    proba: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Equity of a long-when-P(up)>0.5 rule vs buy-and-hold on the holdout.

    Diagnostic only. No costs, no slippage, no position sizing. Not a backtest
    you should trade.
    """
    pred = (proba >= 0.5).astype(float)
    strat = pred * next_returns
    bh = np.cumprod(1.0 + next_returns)
    eq = np.cumprod(1.0 + strat)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dates, bh, label="Buy & hold", color="#4C78A8", linewidth=1.8)
    ax.plot(dates, eq, label="Long when P(up) ≥ 0.5", color="#F58518", linewidth=1.8)
    ax.set_ylabel("Growth of $1 (gross, no costs)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_metrics_json(payload: dict, out_path: Path | None = None) -> Path:
    import json

    out_path = out_path or (REPORTS_DIR / "metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path
