"""
EMD Hybrid Forecaster using PyEMD (CEEMDAN, EEMD, EMD).
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from PyEMD import CEEMDAN, EEMD, EMD

class EMDHybridForecaster:
    def __init__(self, ticker, lookback_days, macro_cols, svr_params, mlp_params, lgbm_params,
                 max_imfs=6, method='ceemdan', random_seed=42):
        self.ticker = ticker
        self.lookback = lookback_days
        self.macro_cols = macro_cols
        self.svr_params = svr_params
        self.mlp_params = mlp_params
        self.lgbm_params = lgbm_params
        self.max_imfs = max_imfs
        self.method = method.lower()
        self.random_seed = random_seed
        self.imfs = None
        self.residual = None
        self.models = {}

    def decompose(self, series):
        values = series.values.flatten()
        if self.method == 'ceemdan':
            decomposer = CEEMDAN()
            imfs = decomposer.ceemdan(values, max_imf=self.max_imfs)
        elif self.method == 'eemd':
            decomposer = EEMD()
            imfs = decomposer.eemd(values, max_imf=self.max_imfs)
        else:  # standard EMD
            decomposer = EMD()
            imfs = decomposer.emd(values, max_imf=self.max_imfs)
        if imfs.shape[0] > self.max_imfs:
            imfs = imfs[:self.max_imfs]
        self.imfs = [imfs[i, :] for i in range(imfs.shape[0])]
        self.residual = self.imfs[-1]
        return self.imfs

    def build_features(self, imf_idx, returns_series, macro_df):
        macro_aligned = macro_df.reindex(returns_series.index, method='ffill')
        data = pd.DataFrame(index=returns_series.index)
        data['ret'] = returns_series
        for col in self.macro_cols:
            data[col] = macro_aligned[col]
        for lag in range(1, self.lookback + 1):
            data[f'ret_lag_{lag}'] = data['ret'].shift(lag)
        imf_series = pd.Series(self.imfs[imf_idx], index=returns_series.index)
        data['target'] = imf_series.shift(-1)
        data.dropna(inplace=True)
        feature_cols = [f'ret_lag_{lag}' for lag in range(1, self.lookback+1)] + self.macro_cols
        X = data[feature_cols].values
        y = data['target'].values
        return X, y, data.index

    def train_imf_models(self, imf_idx, X_train, y_train, X_val, y_val):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        best_mse = np.inf
        best_model = None
        best_name = None
        best_scaler = scaler

        # SVR
        svr = SVR(**self.svr_params)
        svr.fit(X_train_scaled, y_train)
        y_pred = svr.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = svr
            best_name = 'svr'
            best_scaler = scaler

        # MLP
        mlp = MLPRegressor(**self.mlp_params, random_state=self.random_seed)
        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = mlp
            best_name = 'mlp'
            best_scaler = scaler

        # LightGBM
        lgbm = lgb.LGBMRegressor(**self.lgbm_params, random_state=self.random_seed)
        lgbm.fit(X_train, y_train)
        y_pred = lgbm.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = lgbm
            best_name = 'lgbm'
            best_scaler = None

        self.models[imf_idx] = {
            'model': best_model,
            'name': best_name,
            'scaler': best_scaler,
            'val_mse': best_mse
        }
        return best_name

    def predict(self, last_returns, latest_macro):
        features = list(last_returns) + list(latest_macro)
        X = np.array(features).reshape(1, -1)
        total_pred = 0.0
        for imf_data in self.models.values():
            model = imf_data['model']
            scaler = imf_data['scaler']
            X_scaled = scaler.transform(X) if scaler is not None else X
            total_pred += model.predict(X_scaled)[0]
        return total_pred

    def get_selected_models(self):
        return {idx: data['name'] for idx, data in self.models.items()}
