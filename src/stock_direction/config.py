"""Project-wide constants. Kept in one place so train and predict stay aligned."""

from pathlib import Path

TICKERS: tuple[str, ...] = ("SPY", "AAPL", "MSFT")

# yfinance download settings
YF_PERIOD = "5y"
YF_INTERVAL = "1d"
YF_RETRIES = 5
YF_RETRY_BASE_SLEEP = 2.0

RANDOM_STATE = 42

# Last N trading days are a true holdout. Never used for training or model selection.
HOLDOUT_DAYS = 252
N_SPLITS = 5  # expanding-window TimeSeriesSplit on the development set

# Minimum history needed to compute the longest indicator (SMA-50) plus buffer.
MIN_HISTORY_BARS = 80

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_21",
    "sma_5_ratio",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "vol_20",
    "volume_ratio_20",
    "volume_z_20",
    "hl_range",
    "atr_14_ratio",
    "bb_pct_20",
    "dist_high_20",
    "dist_low_20",
    "dow",
)
