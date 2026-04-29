"""
Data loading and preprocessing for EMD-Hybrid engine.
Based on the VAE engine's data_manager.py adapted for UNIX ms index.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    """Download and load master_data.parquet from HF.
    The parquet has a numeric index (UNIX milliseconds) and columns: Ticker, Close, VIX, DXY, ...
    There is no 'Date' column. We convert the index to datetime and name it 'Date'.
    """
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    
    # The index is UNIX milliseconds (numeric)
    if df.index.dtype.kind in 'iu' and df.index.name != 'Date':
        # Convert index to datetime
        df.index = pd.to_datetime(df.index, unit='ms')
        df.index.name = 'Date'
    # If there is a 'Date' column (fallback), use it
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        # Try to find a datetime column
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df.set_index(col, inplace=True)
                break
    df.sort_index(inplace=True)
    return df

def prepare_returns_matrix(df, tickers):
    """
    Convert price data to daily returns.
    df has index = Date, columns include 'Ticker' and 'Close'.
    """
    # Ensure we have the necessary columns
    if 'Ticker' not in df.columns:
        raise KeyError("Column 'Ticker' not found.")
    if 'Close' not in df.columns:
        raise KeyError("Column 'Close' not found.")
    
    # Pivot
    pivot = df.pivot(index=df.index, columns='Ticker', values='Close')
    # Keep only requested tickers
    pivot = pivot[[t for t in tickers if t in pivot.columns]]
    # Daily returns
    returns = pivot.pct_change().dropna()
    return returns

def prepare_macro_features(df):
    """
    Extract macro columns from the main DataFrame.
    df has index = Date, and macro columns directly.
    """
    # Ensure macro columns exist
    macro_cols_present = [col for col in config.MACRO_COLS if col in df.columns]
    if not macro_cols_present:
        raise ValueError(f"None of the macro columns {config.MACRO_COLS} found. Available: {list(df.columns)}")
    macro_df = df[macro_cols_present].copy()
    # Drop duplicate dates (should already be unique index)
    macro_df = macro_df[~macro_df.index.duplicated(keep='first')]
    macro_df.sort_index(inplace=True)
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
