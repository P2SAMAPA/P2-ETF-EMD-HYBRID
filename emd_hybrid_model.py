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
import emd  # EMD-signal library (install via pip install EMD-signal)

class EMDHybridForecaster:
    """
    Hybrid forecaster using Empirical Mode Decomposition (CEEMDAN from EMD-signal).
    Decomposes a return series into Intrinsic Mode Functions (IMFs),
    then trains separate SVR, MLP, and LightGBM models for each IMF.
    The best model (lowest validation MSE) is selected per IMF.
    Forecasts are obtained by summing predictions from all IMFs.
    """

    def __init__(self, ticker, lookback_days, macro_cols, svr_params, mlp_params, lgbm_params,
                 max_imfs=6, method='ceemdan', random_seed=42):
        """
        Parameters:
        -----------
        ticker : str
            ETF ticker symbol (for logging)
        lookback_days : int
            Number of past returns to use as features
        macro_cols : list of str
            Names of macro feature columns
        svr_params : dict
            Parameters for sklearn.svm.SVR
        mlp_params : dict
            Parameters for sklearn.neural_network.MLPRegressor
        lgbm_params : dict
            Parameters for lightgbm.LGBMRegressor
        max_imfs : int
            Maximum number of IMFs to extract (residual included)
        method : str
            'ceemdan', 'eemd', or 'emd'
        random_seed : int
            Seed for reproducibility
        """
        self.ticker = ticker
        self.lookback = lookback_days
        self.macro_cols = macro_cols
        self.svr_params = svr_params
        self.mlp_params = mlp_params
        self.lgbm_params = lgbm_params
        self.max_imfs = max_imfs
        self.method = method.lower()
        self.random_seed = random_seed
        self.imfs = None          # list of IMF arrays (each length = n_obs)
        self.residual = None
        self.models = {}          # {imf_idx: {'model': obj, 'name': str, 'scaler': scaler, 'val_mse': float}}

    def decompose(self, series):
        """
        Apply EMD / EEMD / CEEMDAN to a 1D return series.

        Parameters:
        -----------
        series : pd.Series
            Daily returns for a single ETF (index = dates)

        Returns:
        --------
        list of np.ndarray
            Each array is an IMF (same length as series). The last IMF is the residual (trend).
        """
        values = series.values.flatten()
        # Use emd.sift.mask_sift for ensemble methods, emd.sift.sift for standard EMD
        if self.method == 'ceemdan':
            imf = emd.sift.mask_sift(values, max_imfs=self.max_imfs, ensemble_mode='ceemdan')
        elif self.method == 'eemd':
            imf = emd.sift.mask_sift(values, max_imfs=self.max_imfs, ensemble_mode='eemd')
        else:  # standard EMD
            imf = emd.sift.sift(values, max_imfs=self.max_imfs)
        
        # imf shape: (n_imfs, n_samples)
        if imf.shape[0] > self.max_imfs:
            imf = imf[:self.max_imfs]
        self.imfs = [imf[i, :] for i in range(imf.shape[0])]
        # Residual is the last IMF (trend component)
        self.residual = self.imfs[-1]
        return self.imfs

    def build_features(self, imf_idx, returns_series, macro_df):
        """
        Build feature matrix X and target y for a given IMF.

        Parameters:
        -----------
        imf_idx : int
            Index of the IMF (0 ... n_imfs-1)
        returns_series : pd.Series
            Original returns series (aligned with macro)
        macro_df : pd.DataFrame
            Macro features with datetime index

        Returns:
        --------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        index : pd.Index
            Dates corresponding to each sample
        """
        # Align macro to returns index (forward fill)
        macro_aligned = macro_df.reindex(returns_series.index, method='ffill')
        # Create feature DataFrame
        data = pd.DataFrame(index=returns_series.index)
        data['ret'] = returns_series
        for col in self.macro_cols:
            data[col] = macro_aligned[col]
        # Lagged returns
        for lag in range(1, self.lookback + 1):
            data[f'ret_lag_{lag}'] = data['ret'].shift(lag)
        # Target: next day's IMF value
        imf_series = pd.Series(self.imfs[imf_idx], index=returns_series.index)
        data['target'] = imf_series.shift(-1)
        # Drop rows with NaN (due to lags or shift)
        data.dropna(inplace=True)
        # Feature columns
        feature_cols = [f'ret_lag_{lag}' for lag in range(1, self.lookback+1)] + self.macro_cols
        X = data[feature_cols].values
        y = data['target'].values
        return X, y, data.index

    def train_imf_models(self, imf_idx, X_train, y_train, X_val, y_val):
        """
        Train SVR, MLP, and LightGBM on the given IMF data.
        Select the model with lowest validation MSE.

        Parameters:
        -----------
        imf_idx : int
            Index of the IMF
        X_train, y_train : np.ndarray
            Training set
        X_val, y_val : np.ndarray
            Validation set

        Returns:
        --------
        best_name : str
            Name of the selected model ('svr', 'mlp', or 'lgbm')
        """
        # Scale features for SVR and MLP (LightGBM does not need scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        best_mse = np.inf
        best_model = None
        best_name = None
        best_scaler = scaler  # default

        # --- SVR ---
        svr = SVR(**self.svr_params)
        svr.fit(X_train_scaled, y_train)
        y_pred = svr.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = svr
            best_name = 'svr'
            best_scaler = scaler

        # --- MLP ---
        mlp = MLPRegressor(**self.mlp_params, random_state=self.random_seed)
        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = mlp
            best_name = 'mlp'
            best_scaler = scaler

        # --- LightGBM ---
        lgbm = lgb.LGBMRegressor(**self.lgbm_params, random_state=self.random_seed)
        lgbm.fit(X_train, y_train)
        y_pred = lgbm.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = lgbm
            best_name = 'lgbm'
            best_scaler = None   # no scaler needed for LGBM

        # Store the selected model and its metadata
        self.models[imf_idx] = {
            'model': best_model,
            'name': best_name,
            'scaler': best_scaler,
            'val_mse': best_mse
        }
        return best_name

    def predict(self, last_returns, latest_macro):
        """
        Predict the next day's absolute return for the ETF.

        Parameters:
        -----------
        last_returns : np.ndarray, shape (lookback_days,)
            Most recent daily returns (from oldest to newest).
        latest_macro : np.ndarray, shape (len(macro_cols),)
            Latest macro feature values (aligned with the date of the last return).

        Returns:
        --------
        float
            Predicted return for the next trading day.
        """
        if len(last_returns) != self.lookback:
            raise ValueError(f"Need exactly {self.lookback} past returns, got {len(last_returns)}")
        if len(latest_macro) != len(self.macro_cols):
            raise ValueError(f"Need {len(self.macro_cols)} macro values, got {len(latest_macro)}")

        # Combine features
        features = list(last_returns) + list(latest_macro)
        X = np.array(features).reshape(1, -1)

        total_pred = 0.0
        for imf_data in self.models.values():
            model = imf_data['model']
            scaler = imf_data['scaler']
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            pred = model.predict(X_scaled)[0]
            total_pred += pred
        return total_pred

    def get_selected_models(self):
        """
        Returns a dictionary mapping IMF index -> selected model name.
        """
        return {idx: data['name'] for idx, data in self.models.items()}

    def get_validation_errors(self):
        """
        Returns a dictionary mapping IMF index -> validation MSE of the selected model.
        """
        return {idx: data['val_mse'] for idx, data in self.models.items()}
