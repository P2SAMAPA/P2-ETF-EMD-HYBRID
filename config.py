"""
Configuration for P2-ETF-EMD-HYBRID engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-emd-hybrid-results"

# --- Universe Definitions (same as VAE engine) ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Features (available from 2008) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- EMD Parameters ---
EMD_METHOD = "ceemdan"       # "ceemdan", "eemd", or "emd"
MAX_IMFS = 6                 # number of IMFs (residual counts as last)
NUM_ENSEMBLE = 50            # for CEEMDAN/EEMD (noise repetitions)

# --- Model Hyperparameters ---
SVR_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'epsilon': 0.001
}
MLP_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'max_iter': 500,
    'random_state': 42
}
LGBM_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1
}

# --- Feature Engineering ---
LOOKBACK_DAYS = 20          # number of past returns used as features

# --- Training Splits ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1             # not used for model selection, only for final eval
MIN_OBSERVATIONS = 252       # minimum samples required to train

# --- Global Training ---
GLOBAL_TRAIN_START = "2008-01-01"
GLOBAL_EPOCHS_NOT_USED = 1   # kept for compatibility, not used

# --- Shrinking Windows ---
# Each window is 3 years: start_year → start_year+2
SHRINKING_WINDOW_START_YEARS = list(range(2008, 2025))

# --- Output ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
