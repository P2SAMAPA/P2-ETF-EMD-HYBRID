"""
Data loading and preprocessing for EMD-Hybrid engine.
Matches the structure of the VAE engine's data_manager.py.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    """Download and load master_data.parquet from HF."""
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df, tickers):
    """
    Convert price data to daily returns.
    df must have columns: Date, Ticker, Close.
    Returns DataFrame with dates as index, tickers as columns.
    """
    # Pivot: dates as index, tickers as columns, values = Close
    pivot = df.pivot(index='Date', columns='Ticker', values='Close')
    # Ensure we only have requested tickers
    pivot = pivot[[t for t in tickers if t in pivot.columns]]
    # Calculate returns
    returns = pivot.pct_change().dropna()
    return returns

def prepare_macro_features(df):
    """
    Extract macro columns and set Date as index.
    Returns DataFrame with dates as index and macro columns.
    """
    macro_df = df[['Date'] + config.MACRO_COLS].drop_duplicates('Date').copy()
    macro_df.set_index('Date', inplace=True)
    macro_df = macro_df.sort_index()
    return macro_df

def align_macro_returns(returns, macro):
    """
    Align macro data to returns index using forward fill.
    Returns aligned macro and trimmed returns.
    """
    macro_aligned = macro.reindex(returns.index, method='ffill')
    valid_mask = macro_aligned.notna().all(axis=1)
    returns_aligned = returns[valid_mask]
    macro_aligned = macro_aligned[valid_mask]
    return macro_aligned, returns_aligned
