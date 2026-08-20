"""Fetch the latest daily bars and print tomorrow's predicted direction.

Uses the saved per-ticker models. Features are built from the full downloaded
history so indicators have enough warmup; only the last complete session is
scored. This is an on-demand daily call, not a tick-level stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from stock_direction.config import (
    FEATURE_COLUMNS,
    MIN_HISTORY_BARS,
    MODELS_DIR,
    TICKERS,
)
from stock_direction.data import download_ticker
from stock_direction.features import add_features


def _load_meta() -> dict:
    path = MODELS_DIR / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained models at {path}. Run: python -m stock_direction.train"
        )
    return json.loads(path.read_text())


def _latest_feature_row(ohlcv: pd.DataFrame) -> tuple[pd.Timestamp, np.ndarray, pd.Series]:
    featured = add_features(ohlcv)
    row = featured.iloc[-1]
    missing = [c for c in FEATURE_COLUMNS if pd.isna(row[c])]
    if missing:
        raise RuntimeError(f"Latest row still has NaN features {missing}; need more history")
    x = row.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64).reshape(1, -1)
    return featured.index[-1], x, row


def predict_ticker(
    ticker: str,
    model_name: str,
    models_dir: Path = MODELS_DIR,
) -> dict:
    ohlcv = download_ticker(ticker, period="2y", interval="1d")
    if len(ohlcv) < MIN_HISTORY_BARS:
        raise RuntimeError(f"{ticker}: only {len(ohlcv)} bars downloaded")

    as_of, x, row = _latest_feature_row(ohlcv)
    model_path = models_dir / f"{ticker}_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Train first.")
    model = joblib.load(model_path)
    proba = float(model.predict_proba(x)[0, 1])
    direction = "UP" if proba >= 0.5 else "DOWN"
    return {
        "ticker": ticker,
        "as_of": str(pd.Timestamp(as_of).date()),
        "last_close": float(row["close"]),
        "model": model_name,
        "p_up": proba,
        "direction": direction,
    }


def format_report(rows: list[dict]) -> str:
    lines = [
        "Next-day direction (close[t+1] vs close[t])",
        f"Features as of last completed session. Model threshold = 0.5.",
        "",
        f"{'Ticker':<8}{'As of':<12}{'Close':>10}  {'P(up)':>7}  {'Call':<6}  Model",
        "-" * 58,
    ]
    for r in rows:
        lines.append(
            f"{r['ticker']:<8}{r['as_of']:<12}{r['last_close']:>10.2f}  "
            f"{r['p_up']:>7.3f}  {r['direction']:<6}  {r['model']}"
        )
    lines.append("")
    lines.append("Not financial advice. These are model outputs, not trade recommendations.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next-day direction from latest daily bars")
    parser.add_argument(
        "--model",
        choices=("xgb", "logreg"),
        default="xgb",
        help="Which saved model family to use (default: xgb)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(TICKERS),
        help="Tickers to score",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_meta()
    rows = []
    for ticker in (t.upper() for t in args.tickers):
        print(f"Fetching {ticker}...")
        rows.append(predict_ticker(ticker, args.model))
    print()
    print(format_report(rows))


if __name__ == "__main__":
    main()
