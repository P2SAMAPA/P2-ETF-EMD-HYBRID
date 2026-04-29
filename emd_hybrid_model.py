"""
EMD Hybrid Forecaster: decompose returns into IMFs, train per-IMF models (SVR/MLP/LGBM),
select best based on validation MSE, and predict next absolute return.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from emd import sift, emd    # EMD-signal library (CEEMDAN via emd.sift)

class EMDHybridForecaster:
    def __init__(self, ticker, lookback_days, macro_cols, svr_params, mlp_params, lgbm_params, max_imfs=6, method='ceemdan'):
        self.ticker = ticker
        self.lookback = lookback_days
        self.macro_cols = macro_cols
        self.svr_params = svr_params
        self.mlp_params = mlp_params
        self.lgbm_params = lgbm_params
        self.max_imfs = max_imfs
        self.method = method.lower()
        self.imfs = None          # list of IMF arrays (n_obs each)
        self.residual = None
        self.models = {}          # {imf_idx: {'model': obj, 'name': str, 'scaler': scaler}}
        self.selected_model_names = {}

    def decompose(self, series):
        """
        Apply CEEMDAN (or EMD) to a 1D return series.
        series: pandas Series (index dates, values returns)
        """
        values = series.values.flatten()
        # Use emd.sift with ensemble mode for CEEMDAN
        if self.method == 'ceemdan':
            # CEEMDAN: sift with ensemble_mode='ceemdan'
            imf = emd.sift.mask_sift(values, max_imfs=self.max_imfs, ensemble_mode='ceemdan')
        elif self.method == 'eemd':
            imf = emd.sift.mask_sift(values, max_imfs=self.max_imfs, ensemble_mode='eemd')
        else:  # emd
            imf = emd.sift.sift(values, max_imfs=self.max_imfs)

        # imf shape: (n_imfs, n_samples)
        if imf.shape[0] > self.max_imfs:
            imf = imf[:self.max_imfs]
        # Store IMFs and residual (last row is residual if return_all=True)
        # Actually, sift returns IMFs including residual as last component.
        self.imfs = [imf[i, :] for i in range(imf.shape[0])]
        # Residual is the last IMF (trend)
        self.residual = self.imfs[-1]
        # Remove residual from list if you prefer to keep it separate
        # But we'll keep all as IMFs and sum all for reconstruction.
        return self.imfs

    def build_features(self, imf_idx, returns_series, macro_df):
        """
        For a given IMF, build feature matrix X and target y.
        Features: lagged returns (original series) and macro values.
        Target: next day's IMF value.
        """
        # Create a DataFrame with original returns and macro
        data = pd.DataFrame(index=returns_series.index)
        data['ret'] = returns_series
        for col in self.macro_cols:
            data[col] = macro_df[col]
        # Create lagged return features
        for lag in range(1, self.lookback + 1):
            data[f'ret_lag_{lag}'] = data['ret'].shift(lag)
        # Target: next IMF value
        imf_series = pd.Series(self.imfs[imf_idx], index=returns_series.index)
        data['target'] = imf_series.shift(-1)   # predict next day's IMF
        # Drop rows with NaN
        data.dropna(inplace=True)
        # Features: all lagged returns + macro columns
        feature_cols = [f'ret_lag_{lag}' for lag in range(1, self.lookback+1)] + self.macro_cols
        X = data[feature_cols].values
        y = data['target'].values
        return X, y, data.index

    def train_imf_models(self, imf_idx, X_train, y_train, X_val, y_val):
        """
        Train SVR, MLP, LGBM on IMF data. Select model with lowest validation MSE.
        Returns: selected model object, its name, and the scaler used (for SVR/MLP).
        """
        # Scale features for SVR and MLP (LGBM handles scale well)
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
        mlp = MLPRegressor(**self.mlp_params)
        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = mlp
            best_name = 'mlp'
            best_scaler = scaler

        # LightGBM
        lgbm = lgb.LGBMRegressor(**self.lgbm_params)
        lgbm.fit(X_train, y_train)  # no scaling needed
        y_pred = lgbm.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = lgbm
            best_name = 'lgbm'
            best_scaler = None   # LGBM doesn't need scaler

        self.models[imf_idx] = {'model': best_model, 'name': best_name, 'scaler': best_scaler}
        return best_name

    def predict(self, returns_series, macro_latest):
        """
        returns_series: most recent LOOKBACK_DAYS of original returns (as numpy array, length lookback)
        macro_latest: latest macro values (1D array)
        Returns: predicted absolute return for next day.
        """
        if len(returns_series) != self.lookback:
            raise ValueError(f"Need exactly {self.lookback} past returns, got {len(returns_series)}")
        # Build feature vector for each IMF (same features for all IMFs)
        # Features: lagged returns (lookback days) + current macro
        features = list(returns_series) + list(macro_latest)
        X = np.array(features).reshape(1, -1)

        pred_sum = 0.0
        for imf_idx, imf_data in self.models.items():
            model = imf_data['model']
            scaler = imf_data['scaler']
            name = imf_data['name']
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            pred = model.predict(X_scaled)[0]
            pred_sum += pred
        # Add residual? The last IMF already includes residual, so summing all IMFs gives full signal.
        # But to be safe, we sum all IMFs (including residual as last IMF)
        return pred_sum

    def get_selected_models(self):
        """Return dict of IMF index -> model name."""
        return {idx: data['name'] for idx, data in self.models.items()}
