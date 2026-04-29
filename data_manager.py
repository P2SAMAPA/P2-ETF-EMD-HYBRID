"""
Data loading and preprocessing for EMD-Hybrid engine.
Copied from the VAE engine and extended with align_macro_returns.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Prepare wide‑format log returns."""
    available_tickers = [t for t in tickers if t in df_wide.columns]
    df_long = pd.melt(
        df_wide, id_vars=['Date'], value_vars=available_tickers,
        var_name='ticker', value_name='price'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available_tickers].dropna()

def prepare_macro_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Extract macro columns and return as DataFrame with Date index."""
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def build_training_sequences(returns: pd.DataFrame, macro: pd.DataFrame) -> tuple:
    """
    Build (X_cond, y) pairs: condition = today's macro, target = next-day returns.
    Returns (cond_array, target_array) as float32 numpy arrays.
    """
    common_idx = returns.index.intersection(macro.index)
    returns = returns.loc[common_idx]
    macro = macro.loc[common_idx]

    cond_list = []
    target_list = []
    for i in range(len(returns) - 1):
        cond_list.append(macro.iloc[i].values.astype(np.float32))
        target_list.append(returns.iloc[i + 1].values.astype(np.float32))

    return np.stack(cond_list), np.stack(target_list)

# ------------------------------------------------------------
# New function for EMD-Hybrid (not in VAE engine)
# ------------------------------------------------------------
def align_macro_returns(returns: pd.DataFrame, macro: pd.DataFrame) -> tuple:
    """
    Align macro data to returns index using forward fill.
    Returns aligned macro and trimmed returns.
    """
    macro_aligned = macro.reindex(returns.index, method='ffill')
    valid_mask = macro_aligned.notna().all(axis=1)
    returns_aligned = returns[valid_mask]
    macro_aligned = macro_aligned[valid_mask]
    return macro_aligned, returns_aligned
