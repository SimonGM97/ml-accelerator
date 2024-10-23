from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.transformers.transformer import Transformer
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import gc
from pprint import pformat
from typing import List, Tuple, Dict

# Get logger
LOGGER = get_logger(name=__name__)


class DataCleaner(Transformer):

    # Pickled attrs
    pickled_attrs = [
        'model_id',
        'outliers_dict',
        'str_imputer',
        'num_imputer'
    ]

    def __init__(
        self,
        transformer_id: str = None,
        target: str = Params.TARGET,
        dataset_name: str = Params.DATASET_NAME,
        z_threshold: float = Params.OUTLIER_Z_THRESHOLD
    ) -> None:
        # Instanciate parent classes
        super().__init__(transformer_id=transformer_id)

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
        self.outliers_dict: Dict[str, Tuple[float]] = None
        self.str_imputer: SimpleImputer = None
        self.num_imputer: SimpleImputer = None

        # Load self.schema
        self.schema: dict = self.load_schema()
    
    """
    Required methods (from Transformer abstract methods)
    """

    def transform(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame = None,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.cleaner_pipeline with fit=False
        X, y = self.cleaner_pipeline(
            X=X, y=y, 
            fit=False,
            debug=debug
        )
        
        return X, y
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame = None,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.cleaner_pipeline with fit=True
        X, y = self.cleaner_pipeline(
            X=X, y=y, 
            fit=True,
            debug=debug
        )

        return X, y
    
    def diagnose(self) -> None:
        return None
    
    """
    Non-required methods
    """

    def cleaner_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        fit: bool = False,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Merge datasets
        df = self.merge_datasets(X=X, y=y)
        
        # Delete X & y
        del X
        if y is not None:
            del y
        gc.collect()

        # Rename columns
        df = self.rename_columns(df=df)

        # Drop dummy columns
        df = self.drop_dummy_columns(df=df)
        
        # Find column attributes
        self.find_column_attrs(df=df)

        # Drop duplicates
        df = self.drop_duplicates(df=df) 
        
        # Add missing rows
        df = self.add_missing_rows(df=df)
        
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

    def rename_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Replace empty spaces with '_'
        df.columns = df.columns.str.replace(' ', '_')

        return df

    def drop_dummy_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        if df.shape[0] > 0.05 * self.schema['length']:
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
                if ('float' in field['type'] or 'int' in field['type']) and field['name'] in df.columns
            ]

            # Set self.cat_cols
            self.cat_cols: List[str] = [
                field['name'] for field in self.schema['fields']
                if field['type'] in ['string', 'object'] and field['name'] in df.columns
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
        # Drop duplicate columns & rows
        df: pd.DataFrame = (
            df.loc[:, ~df.columns.duplicated(keep='first')]
            .drop_duplicates()
        )

        return df

    def add_missing_rows(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Add expected idx
        # expected_idx = pd.RangeIndex(df.index.min(), df.index.max(), step=1)

        # Fill missing rows with nan values
        # df = df.reindex(expected_idx).fillna(np.nan)

        return df

    def set_data_types(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Extract expected data dypes
        # map_types: dict = {
        #     'string': np.dtypes.StrDType,
        #     'object': np.dtypes.ObjectDType,
        #     'float64': np.dtypes.Float64DType,
        #     'float32': np.dtypes.Float32DType,
        #     'float16': np.dtypes.Float16DType,
        #     'int': np.dtypes.IntDType,
        #     'int16': np.dtypes.Int16DType,
        #     'int32': np.dtypes.Int32DType,
        #     'int64': np.dtypes.Int64DType
        # }

        expected_dtypes: dict = {
            field['name']: field['type'] for field in self.schema['fields']
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
    ) -> pd.DataFrame:
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
        self.outliers_dict: Dict[str, Tuple[float]] = {
            col: find_threshold(col) for col in self.num_cols
        }

        LOGGER.info('New self.outliers_dict was populated.') # pformat(self.outliers_dict))

    def correct_outliers(
        self,
        df: pd.DataFrame,
        debug: bool = False
    ) -> pd.DataFrame:
        for col, thresholds in self.outliers_dict.items():
            # Find low outliers mask
            low_outliers_mask = df[col] < thresholds[0]

            if debug and low_outliers_mask.sum() > 0:
                LOGGER.debug(
                    'Low outliers found in %s (min value allowed - %s):\n%s',
                    col, thresholds[0], df.loc[low_outliers_mask, col]
                )

            # Correct low outliers
            df.loc[low_outliers_mask, col] = thresholds[0]

            if debug and low_outliers_mask.sum() > 0:
                LOGGER.debug(
                    'Low outliers corrected:\n%s',
                    df.loc[low_outliers_mask, col]
                )

            # Find high outliers mask
            high_outliers_mask = df[col] > thresholds[1]

            if debug and high_outliers_mask.sum() > 0:
                LOGGER.debug(
                    'High outliers found in %s (max value allowed - %s):\n%s',
                    col, thresholds[1], df.loc[high_outliers_mask, col]
                )

            # Replace high outliers
            df.loc[high_outliers_mask, col] = thresholds[1]

            if debug and high_outliers_mask.sum() > 0:
                LOGGER.debug(
                    'High outliers corrected:\n%s',
                    df.loc[high_outliers_mask, col]
                )

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

        LOGGER.info('New imputers have been fitted.')
    
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


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python ml_accelerator/data_processing/transformers/data_cleaner.py
if __name__ == "__main__":
    # Instanciate DataCleaner
    DC: DataCleaner = DataCleaner()

    # Load DataCleaner
    DC.load()