"""
Data loading and preprocessing for EMD-Hybrid engine.
Handles UNIX timestamp index (milliseconds) and auto-detects columns.
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
    
    # Check if the index is a numeric UNIX timestamp (milliseconds)
    if isinstance(df.index, pd.Index) and pd.api.types.is_numeric_dtype(df.index):
        # Convert from milliseconds to datetime
        df.index = pd.to_datetime(df.index, unit='ms')
        df.index.name = 'Date'
    elif 'Date' in df.columns:
        # Fallback: if there is a Date column, use it as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        # Try to find any datetime column
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or col.lower() in ['date', 'timestamp', 'ds']:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break
    df.sort_index(inplace=True)
    return df

def prepare_returns_matrix(df, tickers):
    """
    Convert price data to daily returns.
    Auto-detects price column (Close, Adj Close, price, etc.).
    df index is datetime.
    """
    # Find price column
    price_col = None
    possible_price = ['Close', 'close', 'Adj Close', 'adj_close', 'price', 'adjusted_close', 'value']
    for col in possible_price:
        if col in df.columns:
            price_col = col
            break
    if price_col is None:
        raise KeyError("No price column found. Expected 'Close', 'Adj Close', or similar.")
    
    # Find ticker column
    ticker_col = None
    possible_ticker = ['Ticker', 'ticker', 'Symbol', 'symbol']
    for col in possible_ticker:
        if col in df.columns:
            ticker_col = col
            break
    if ticker_col is None:
        raise KeyError("No ticker column found. Expected 'Ticker' or 'ticker'.")
    
    # Pivot: rows = dates, columns = tickers, values = price
    pivot = df.pivot(columns=ticker_col, values=price_col)
    # Keep only requested tickers that exist
    available = [t for t in tickers if t in pivot.columns]
    if not available:
        raise ValueError(f"None of {tickers} found. Available: {list(pivot.columns[:5])}...")
    pivot = pivot[available]
    # Daily returns
    returns = pivot.pct_change().dropna()
    return returns

def prepare_macro_features(df):
    """
    Extract macro columns. Assumes df index is datetime.
    """
    available = [col for col in config.MACRO_COLS if col in df.columns]
    if not available:
        raise ValueError(f"None of {config.MACRO_COLS} found. Available columns: {list(df.columns[:10])}")
    macro_df = df[available].copy()
    macro_df = macro_df.apply(pd.to_numeric, errors='coerce')
    macro_df.dropna(how='all', inplace=True)
    return macro_df

def align_macro_returns(returns, macro):
    """
    Align macro data to returns index using forward fill.
    Returns aligned macro and trimmed returns.
    """
    macro_aligned = macro.reindex(returns.index, method='ffill')
    # Drop rows where any macro is still NaN (before start of macro)
    valid_mask = macro_aligned.notna().all(axis=1)
    returns_aligned = returns[valid_mask]
    macro_aligned = macro_aligned[valid_mask]
    return macro_aligned, returns_aligned
