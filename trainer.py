"""
Main training script for EMD-Hybrid engine.
Runs global training (2008-current) and shrinking windows (2008-2024).
Fixed indexing in run_global_mode to use the actual feature index.
"""

import json
import pandas as pd
import numpy as np
import config
import data_manager
from emd_hybrid_model import EMDHybridForecaster
import push_results

def run_global_mode(returns, macro, tickers, lookback_days, macro_cols, svr_params, mlp_params, lgbm_params, max_imfs=6):
    """
    Global training: 80% train, 10% val, 10% test (chronological) based on the feature index.
    Returns forecast for the day after the last training date.
    """
    ticker_forecasts = []
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        series = returns[ticker].dropna()
        # Align macro to series dates (forward fill)
        macro_aligned = macro.reindex(series.index, method='ffill')
        valid = macro_aligned.notna().all(axis=1)
        series = series[valid]
        macro_aligned = macro_aligned[valid]
        if len(series) < config.MIN_OBSERVATIONS:
            continue

        # Decompose
        forecaster = EMDHybridForecaster(
            ticker=ticker,
            lookback_days=lookback_days,
            macro_cols=macro_cols,
            svr_params=svr_params,
            mlp_params=mlp_params,
            lgbm_params=lgbm_params,
            max_imfs=max_imfs,
            method=config.EMD_METHOD
        )
        imfs = forecaster.decompose(series)

        # For each IMF, build features and train
        # We will use the global split based on the feature index (which may have fewer rows)
        # but first we need the index after building features.
        feature_indices = []
        for i in range(len(imfs)):
            X, y, idx = forecaster.build_features(i, series, macro_aligned)
            feature_indices.append(idx)
            # Use the first IMF's index as the common reference (they should be identical)
            if i == 0:
                common_idx = idx
        # Check that all IMFs have same feature index
        for idx in feature_indices:
            if not idx.equals(common_idx):
                print(f"Warning: IMF indices differ for {ticker}, skipping.")
                continue

        n_feat = len(common_idx)
        train_end = int(n_feat * 0.8)
        val_end = int(n_feat * 0.9)
        train_dates = common_idx[:train_end]
        val_dates = common_idx[train_end:val_end]
        # test_dates = common_idx[val_end:]  # not used

        # Train each IMF model
        for i in range(len(imfs)):
            X, y, idx = forecaster.build_features(i, series, macro_aligned)
            # Split using the same date ranges (but they are aligned)
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            if len(X_train) > 0 and len(X_val) > 0:
                forecaster.train_imf_models(i, X_train, y_train, X_val, y_val)

        # Predict next day after last available data
        last_returns = series.iloc[-lookback_days:].values
        if len(last_returns) < lookback_days:
            continue
        latest_macro = macro_aligned.iloc[-1:].values.flatten()
        pred_return = forecaster.predict(last_returns, latest_macro)
        ticker_forecasts.append({
            'ticker': ticker,
            'predicted_return': float(pred_return),
            'selected_models': forecaster.get_selected_models()
        })

    if not ticker_forecasts:
        return None
    ticker_forecasts.sort(key=lambda x: x['predicted_return'], reverse=True)
    top3 = [{'ticker': t['ticker'], 'predicted_return': t['predicted_return']} for t in ticker_forecasts[:3]]
    return {
        'top_picks': top3,
        'all_scores': ticker_forecasts,
        'mode': 'global',
        'training_end_date': str(returns.index[-1].date())
    }

def run_shrinking_windows(df_master, macro, tickers, lookback_days, macro_cols, svr_params, mlp_params, lgbm_params, max_imfs=6):
    """Shrinking windows: each window 3 years, 80/10/10 split, collect top picks per window, then consensus."""
    windows_results = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        end_year = start_year + 2
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = pd.Timestamp(f"{end_year}-12-31")
        mask = (df_master['Date'] >= start_date) & (df_master['Date'] <= end_date)
        window_df = df_master[mask].copy()
        if len(window_df) < config.MIN_OBSERVATIONS:
            continue

        returns = data_manager.prepare_returns_matrix(window_df, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        macro_aligned, returns_aligned = data_manager.align_macro_returns(returns, macro)
        if len(returns_aligned) < config.MIN_OBSERVATIONS:
            continue

        result = run_global_mode(returns_aligned, macro_aligned, tickers, lookback_days, macro_cols, svr_params, mlp_params, lgbm_params, max_imfs)
        if result:
            top_ticker = result['top_picks'][0]['ticker']
            windows_results.append({
                'window_start': start_year,
                'window_end': end_year,
                'ticker': top_ticker,
                'predicted_return': result['top_picks'][0]['predicted_return']
            })
    if not windows_results:
        return None
    vote = {}
    for w in windows_results:
        vote[w['ticker']] = vote.get(w['ticker'], 0) + 1
    consensus_ticker = max(vote, key=vote.get)
    conviction = vote[consensus_ticker] / len(windows_results) * 100
    return {
        'consensus_ticker': consensus_ticker,
        'conviction': conviction,
        'num_windows': len(windows_results),
        'windows': windows_results
    }

def main():
    import os
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return
    df_master = data_manager.load_master_data()
    macro = data_manager.prepare_macro_features(df_master)
    macro = macro.sort_index()

    all_results = {}
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns_all = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns_all) < config.MIN_OBSERVATIONS:
            continue
        macro_aligned, returns_aligned = data_manager.align_macro_returns(returns_all, macro)
        if len(returns_aligned) < config.MIN_OBSERVATIONS:
            continue

        # Global mode
        global_res = run_global_mode(
            returns_aligned, macro_aligned, tickers,
            lookback_days=config.LOOKBACK_DAYS,
            macro_cols=config.MACRO_COLS,
            svr_params=config.SVR_PARAMS,
            mlp_params=config.MLP_PARAMS,
            lgbm_params=config.LGBM_PARAMS,
            max_imfs=config.MAX_IMFS
        )
        if global_res:
            all_results[universe_name] = {'global': global_res}
            print(f"  Global top: {global_res['top_picks'][0]['ticker']} ({global_res['top_picks'][0]['predicted_return']:.6f})")
        # Shrinking windows mode
        shrinking_res = run_shrinking_windows(
            df_master, macro, tickers,
            lookback_days=config.LOOKBACK_DAYS,
            macro_cols=config.MACRO_COLS,
            svr_params=config.SVR_PARAMS,
            mlp_params=config.MLP_PARAMS,
            lgbm_params=config.LGBM_PARAMS,
            max_imfs=config.MAX_IMFS
        )
        if shrinking_res:
            if 'global' in all_results[universe_name]:
                all_results[universe_name]['shrinking'] = shrinking_res
            else:
                all_results[universe_name] = {'shrinking': shrinking_res}
            print(f"  Shrinking consensus: {shrinking_res['consensus_ticker']} ({shrinking_res['conviction']:.0f}%)")

    output = {
        "run_date": config.TODAY,
        "universes": all_results
    }
    push_results.push_daily_result(output)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    main()
