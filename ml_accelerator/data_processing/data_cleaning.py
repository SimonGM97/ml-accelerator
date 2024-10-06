from ml_accelerator.config.params import Params
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.aws.s3_helper import load_from_s3, save_to_s3
from ml_accelerator.utils.filesystem.filesystem_helper import (
    load_from_filesystem,
    save_to_filesystem
)

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import gc
import os

from typing import List, Tuple, Dict

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


class DataCleaner(DataHelper):

    # Pickled attrs
    pickled_attrs = [
        'outliers_dict',
        'str_imputer',
        'num_imputer'
    ]

    def __init__(
        self,
        target: str = Params.TARGET,
        dataset_name: str = Params.DATASET_NAME,
        z_threshold: float = Params.OUTLIER_Z_THRESHOLD
    ) -> None:
        # Instanciate parent classes
        super().__init__(
            target=target,
            dataset_name=dataset_name
        )

        # Set attributes
        self.target: str = target
        self.dataset_name: str = dataset_name
        self.z_threshold: float = z_threshold

        # Set default attributes
        self.num_cols: List[str] = None
        self.cat_cols: List[str] = None
        self.datetime_cols: List[str] = None

        self.ffill_cols: List[str] = None
        self.interpolate_cols: List[str] = None
        self.simple_impute_cols: List[str] = None

        # Load attrs
        self.outliers_dict: dict = None
        self.str_imputer: SimpleImputer = None
        self.num_imputer: SimpleImputer = None

        # Load self.schema
        self.schema: dict = self.load_schema()
    
    def transform(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.cleaner_pipeline with fit=False
        X, y = self.cleaner_pipeline(X=X, y=y, fit=False)
        
        return X, y
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.cleaner_pipeline with fit=True
        X, y = self.cleaner_pipeline(X=X, y=y, fit=True)

        return X, y
    
    def cleaner_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        fit: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Merge datasets
        df = self.merge_datasets(X=X, y=y)

        # Delete X & y
        del X
        if y is not None:
            del y
        gc.collect()

        # Drop dummy columns
        df = self.drop_dummy_columns(df=df)

        # Find column attributes
        self.find_column_attrs(df=df)

        # Drop duplicates
        df = self.drop_duplicates(df=df) 

        # Set data types
        df = self.set_data_types(df=df)

        # Remove unexpected values
        df = self.remove_unexpected_values(df=df)
        
        # Fit self.outliers_dict
        if fit:
            self.fit_outliers_dict(df=df)
        
        # Remove & correct outliers
        df = self.correct_outliers(df=df)

        # Cap min values
        df = self.cap_min_values(df=df)

        # Cap max values
        df = self.cap_max_values(df=df)

        # Fit SimpleImputers
        if fit:
            self.fit_imputers(df=df)

        # Fill null values
        df = self.fill_nulls(df=df)

        # Divide X & y
        X, y = self.divide_datasets(df=df)

        # Delete df
        del df
        gc.collect()

        return X, y
    
    def merge_datasets(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> pd.DataFrame:
        if y is not None:
            # Concatenate datasets
            df: pd.DataFrame = pd.concat([y, X], axis=1)
        else:
            df: pd.DataFrame = X.copy(deep=True)

        return df
    
    def divide_datasets(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.target is not None and self.target in df.columns:
            # Divide df into X & y
            X, y = df.drop(columns=[self.target]), df[[self.target]]

            return X, y
        else:
            return df, None

    def drop_dummy_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Drop columns where all values are NaN
        df.dropna(axis=1, how='all', inplace=True)

        # Drop columns with all values equal to 0
        df = df.loc[:, (df!=0).any(axis=0)]

        # Drop unexpected columns
        drop_columns = [c for c in df.columns if c not in [f['name'] for f in self.schema['fields']]]
        df.drop(columns=drop_columns, inplace=True)

        return df

    def find_column_attrs(
        self,
        df: pd.DataFrame
    ) -> None:
        # Fill num_cols & str_cols
        if self.schema is not None:
            # Set self.num_cols
            self.num_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['type'] in ['int', 'float'] and field['name'] in df.columns
            ]

            # Set self.cat_cols
            self.cat_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['type'] in ['string'] and field['name'] in df.columns
            ]

            # Set self.datetime_cols
            self.datetime_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['type'] in ['datetime'] and field['name'] in df.columns
            ]

            # Set self.ffill_cols
            self.ffill_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['fillna_method'] == 'ffill' and field['name'] in df.columns
            ]

            # Set self.interpolate_cols
            self.interpolate_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['fillna_method'] == 'interpolate' and field['name'] in df.columns
            ]

            # Set self.simple_impute_cols
            self.simple_impute_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['fillna_method'] == 'simple_imputer' and field['name'] in df.columns
            ]
        
        # self.num_cols = list(df.select_dtypes(include=['number']).columns)
        # self.cat_cols = list(df.select_dtypes(exclude=['number']).columns)

    def drop_duplicates(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        df: pd.DataFrame = (
            df.loc[:, ~df.columns.duplicated(keep='first')]
            .drop_duplicates()
        )

        return df

    def set_data_types(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Extract expected data dypes
        map_types: dict = {
            'string': str,
            'object': object,
            'float': float,
            'float32': np.float32,
            'float64': np.float64,
            'int': int,
            'int32': np.int32,
            'int64': np.int64
        }

        expected_dtypes: dict = {
            field['name']: map_types[field['type']] for field in self.schema['fields']
            if field['name'] in df.columns
        }

        # Transform data types
        df = df.astype(expected_dtypes)

        return df
    
    def cap_min_values(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Extract expected minimum values
        min_vals: dict = {
            field['name']: field['min_value'] for field in self.schema['fields']
            if field['min_value'] is not None
        }

        # Clip minimum values
        clip_cols = list(min_vals.keys())
        df[clip_cols] = df[clip_cols].clip(lower=min_vals)

        return df

    def cap_max_values(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Extract expected maximum values
        max_vals: dict = {
            field['name']: field['max_value'] for field in self.schema['fields']
            if field['max_value'] is not None
        }

        # Clip minimum values
        clip_cols = list(max_vals.keys())
        df[clip_cols] = df[clip_cols].clip(upper=max_vals)

        return df

    def remove_unexpected_values(
        self,
        df: pd.DataFrame
    ):
        # Extract allowed values
        allowed_vals: Dict[str, List[str]] = {
            field['name']: field['allowed_values'] for field in self.schema['fields']
            if field['allowed_values'] is not None and field['name'] in df.columns
        }

        # Apply filtering for each column based on allowed values
        def filter_allowed_values(column: pd.Series, allowed: list):
            return column.where(column.isin(allowed), np.nan)

        for col, allowed in allowed_vals.items():
            df[col] = filter_allowed_values(df[col], allowed)

        return df

    def fit_outliers_dict(
        self,
        df: pd.DataFrame
    ) -> None:
        def find_threshold(col: str):
            # Calculate mean & std
            mean, std = df[col].mean(), df[col].std()

            # Return thresholds
            return mean - self.z_threshold * std, mean + self.z_threshold * std
        
        # Populate self.outliers_dict
        self.outliers_dict = {
            col: find_threshold(col) for col in self.num_cols
        }

    def correct_outliers(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        for col, thresholds in self.outliers_dict.items():
            # Find low outliers mask
            low_outliers_mask = df[col] < thresholds[0]

            # Correct low outliers
            df.loc[low_outliers_mask, col] = thresholds[0]

            # Find high outliers mask
            high_outliers_mask = df[col] > thresholds[1]

            # Replace high outliers
            df.loc[high_outliers_mask, col] = thresholds[1]

        return df
    
    def fit_imputers(
        self,
        df: pd.DataFrame
    ) -> None:
        # Update str_imputer
        self.str_imputer = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
        if len(self.cat_cols) > 0:
            self.str_imputer.fit(df[self.cat_cols])

        # Update num_imputer
        self.num_imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        if len(self.num_cols) > 0:
            self.num_imputer.fit(df[self.num_cols])
    
    def fill_nulls(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Replace inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill ffill columns
        if len(self.ffill_cols) > 0 and df[self.ffill_cols].isnull().sum().sum() > 0:
            df[self.ffill_cols] = (
                df[self.ffill_cols]
                .ffill()
                .bfill()
            )
        
        # Fill interpolate columns
        if len(self.interpolate_cols) > 0 and df[self.interpolate_cols].isnull().sum().sum() > 0:
            df[self.interpolate_cols] = df[self.interpolate_cols].interpolate(method='linear')

        # Fill SimpleImputer columns
        if len(self.simple_impute_cols) > 0 and df[self.simple_impute_cols].isnull().sum().sum() > 0:
            if len(self.cat_cols) > 0:
                df[self.cat_cols] = self.str_imputer.transform(df[self.cat_cols])

            if len(self.num_cols) > 0:
                df[self.num_cols] = self.num_imputer.transform(df[self.num_cols])
        
        return df

    def save(self) -> None:
        # Run self.save_transformer
        self.save_transformer(transformer_name='data_cleaner')

    def load(self) -> None:
        # Run self.load_transformer
        self.load_transformer(transformer_name='data_cleaner')
