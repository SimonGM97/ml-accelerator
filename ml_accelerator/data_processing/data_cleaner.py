from ml_accelerator.config.params import Params
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from typing import List


class DataCleaner:

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        # Compute any necessary statistics here (e.g., median)
        self.medians = X.median()
        return self
    
    def transform(self, X, y=None):
        # Fill missing values with the median
        X_clean = X.fillna(self.medians)
        
        # Drop columns where all values are NaN
        X_clean = X_clean.dropna(axis=1, how='all')
        
        return X_clean
    
    def remove_unexpected_neg_values(
        self,
        X: pd.DataFrame,
        non_neg_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Replace Negative Values in Non-Negative columns
        """
        if non_neg_cols is None:
            # Replace Negative Values with np.nan
            for col in non_neg_cols:
                X.loc[X[col] < 0, col] = np.nan

        return X