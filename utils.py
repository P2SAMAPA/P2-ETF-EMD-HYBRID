"""
Utility functions for time series splitting and evaluation.
"""

import numpy as np
import pandas as pd

def temporal_split(dates, train_ratio=0.8, val_ratio=0.1):
    """Return indices for train, validation, test based on chronological order."""
    n = len(dates)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_idx = np.arange(train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    return train_idx, val_idx, test_idx
