from ml_accelerator.config.params import Params
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from typing import List

# Get logger
LOGGER = get_logger(
    name=__name__,
    level=Params.LEVEL,
    txt_fmt=Params.TXT_FMT,
    json_fmt=Params.JSON_FMT,
    filter_lvls=Params.FILTER_LVLS,
    log_file=Params.LOG_FILE,
    backup_count=Params.BACKUP_COUNT
)


class DataCleaner(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        schema: dict = None,
        z_threshold: float = 2.5
    ) -> None:
        # Set attributes
        self.schema: dict = schema
        self.z_threshold: float = z_threshold

        # Set default attributes
        self.num_cols: List[str] = None
        self.str_cols: List[str] = None
        self.datetime_cols: List[str] = None

        self.ffill_cols: List[str] = None
        self.interpolate_cols: List[str] = None
        self.simple_impute_cols: List[str] = None

        self.outliers_dict: dict = None

        # Set column attributes
        self.find_column_attrs()

    def find_column_attrs(self) -> None:
        # Fill num_cols & str_cols
        if self.schema is not None:
            # Set self.num_cols
            self.num_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['type'] in ['int', 'float']
            ]

            # Set self.str_cols
            self.str_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['type'] in ['string']
            ]

            # Set self.datetime_cols
            self.datetime_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['type'] in ['datetime']
            ]

            # Set self.ffill_cols
            self.ffill_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['fillna_method'] == 'ffill'
            ]

            # Set self.interpolate_cols
            self.interpolate_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['fillna_method'] == 'interpolate'
            ]

            # Set self.simple_impute_cols
            self.simple_impute_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['fillna_method'] == 'simple_imputer'
            ]
        
        # self.num_cols = list(df.select_dtypes(include=['number']).columns)
        # self.str_cols = list(df.select_dtypes(exclude=['number']).columns)

    def fit(self, X, y=None):
        # Run self.cleaner_pipeline with fit=True
        self.cleaner_pipeline(X=X, fit=True)

        return self
    
    def transform(self, X, y=None):
        # Run self.cleaner_pipeline with fit=False
        X: pd.DataFrame = self.cleaner_pipeline(X=X, fit=False)
        
        return X
    
    def fit_transform(self, X, y=None):
        # Run self.cleaner_pipeline with fit=True
        X: pd.DataFrame = self.cleaner_pipeline(X=X, fit=True)

        return X
    
    def cleaner_pipeline(
        self,
        X: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        # Drop dummy columns
        X: pd.DataFrame = self.drop_dummy_columns(X=X)

        # Set data types
        X: pd.DataFrame = self.set_data_types(X=X)

        # Remove unexpected values
        X: pd.DataFrame = self.remove_unexpected_values(X=X)

        # Remove & correct outliers
        if fit:
            self.fit_outliers_dict(X=X)

        X: pd.DataFrame = self.correct_outliers(X=X)

        # Cap min values
        X: pd.DataFrame = self.cap_min_values(X=X)

        # Cap max values
        X: pd.DataFrame = self.cap_max_values(X=X)

        # Fill null values
        if fit:
            self.fit_imputers(X=X)

        X: pd.DataFrame = self.fill_nulls(X=X)

        return X
    
    def drop_dummy_columns(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # Drop columns where all values are NaN
        X.dropna(axis=1, how='all', inplace=True)

        # Drop columns with all values equal to 0
        X = X.loc[:, (X!=0).any(axis=0)]

        # Drop unexpected columns
        drop_columns = [c for c in X.columns if c not in [f['name'] for f in self.schema['fields']]]
        X.drop(columns=drop_columns, inplace=True)

        return X

    def set_data_types(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        return X
    
    def cap_min_values(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        return X

    def cap_max_values(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        return X

    def remove_unexpected_values(
        self,
        X: pd.DataFrame
    ):
        return X

    def fit_outliers_dict(
        self,
        X: pd.DataFrame
    ) -> None:
        def find_threshold(col: str):
            # Calculate mean & std
            mean, std = X[col].mean(), X[col].std()

            # Return thresholds
            return mean - self.z_threshold * std, mean + self.z_threshold * std
        
        self.outliers_dict = {
            col: find_threshold(col) for col in self.num_cols
        }

    def correct_outliers(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        for col, thresholds in self.outliers_dict.items():
            # Find low outliers mask
            low_outliers_mask = X[col] < thresholds[0]

            # Correct low outliers
            X.loc[low_outliers_mask, col] = thresholds[0]

            # Find high outliers mask
            high_outliers_mask = X[col] > thresholds[1]

            # Replace high outliers
            X.loc[high_outliers_mask, col] = thresholds[1]

        return X
    
    def fit_imputers(
        self,
        X: pd.DataFrame
    ) -> None:
        # Update str_imputer
        self.str_imputer = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
        if len(self.str_cols) > 0:
            self.str_imputer.fit(X[self.str_cols])

        # Update num_imputer
        self.num_imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        if len(self.num_cols) > 0:
            self.num_imputer.fit(X[self.num_cols])
    
    def fill_nulls(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # Replace inf values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill ffill columns
        if len(self.ffill_cols) > 0 and X[self.ffill_cols].isnull().sum().sum() > 0:
            X[self.ffill_cols] = (
                X[self.ffill_cols]
                .ffill()
                .bfill()
            )
        
        # Fill interpolate columns
        if len(self.interpolate_cols) > 0 and X[self.interpolate_cols].isnull().sum().sum() > 0:
            X[self.interpolate_cols] = X[self.interpolate_cols].interpolate(method='linear')

        # Fill SimpleImputer columns
        if len(self.simple_impute_cols) > 0 and X[self.simple_impute_cols].isnull().sum().sum() > 0:
            if len(self.str_cols) > 0:
                X[self.str_cols] = self.str_imputer.transform(X[self.str_cols])

            if len(self.num_cols) > 0:
                X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])
        
        return X
