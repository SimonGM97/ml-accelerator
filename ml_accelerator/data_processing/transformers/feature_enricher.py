from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.transformers.feature_selector import FeatureSelector
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from holidays import CountryHoliday
from typing import List, Tuple, Dict
from pprint import pformat


# Get logger
LOGGER = get_logger(name=__name__)


class FeatureEnricher(FeatureSelector):

    # Pickled attrs
    pickled_attrs = [
        'engineer_cols',
        'outliers_dict',
        'derivatives_dict'
    ]

    def __init__(
        self,
        transformer_id: str = None,
        add_outlier_features: bool = Params.ADD_OUTLIER_FEATURES,
        outlier_features_z: float = Params.OUTLIER_FEATURES_Z,
        add_fibonachi_features: bool = Params.ADD_FIBONACHI_FEATURES,
        add_derivative_features: bool = Params.ADD_DERIVATIVE_FEATURES,
        add_lag_features: bool = Params.ADD_LAG_FEATURES,
        lags: List[int] = Params.LAGS,
        add_rolling_features: bool = Params.ADD_ROLLING_FEATURES,
        add_ema_features: bool = Params.ADD_EMA_FEATURES,
        rolling_windows: List[int] = Params.ROLLING_WINDOWS,
        add_temporal_embedding_features: bool = Params.ADD_TEMPORAL_EMBEDDING_FEATURES,
        datetime_col: bool = Params.DATETIME_COL,
        holiday_country: str = Params.HOLIDAY_COUNTRY
    ) -> None:
        # Instanciate parent classes
        super().__init__(transformer_id=transformer_id)

        # Set non-load attributes
        self.add_outlier_features: bool = add_outlier_features
        self.outlier_features_z: float = outlier_features_z
        self.add_fibonachi_features: bool = add_fibonachi_features
        self.add_derivative_features: bool = add_derivative_features
        self.add_lag_features: bool = add_lag_features
        self.lags: List[int] = lags
        self.add_rolling_features: bool = add_rolling_features
        self.add_ema_features: bool = add_ema_features
        self.rolling_windows: List[int] = rolling_windows
        self.add_temporal_embedding_features: bool = add_temporal_embedding_features
        self.datetime_col: bool = datetime_col
        self.holiday_country: str = holiday_country

        # Set attributes to load
        self.engineer_cols: List[str] = None
        self.outliers_dict: Dict[str, Tuple[float]] = None
        self.derivatives_dict: Dict[str, List[float]] = None

    """
    Required methods (from Transformer abstract methods)
    """

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.enricher_pipeline
        X, y = self.enricher_pipeline(X=X, y=y, fit=False)
        
        return X, y

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.enricher_pipeline
        X, y = self.enricher_pipeline(X=X, y=y, fit=True)

        return X, y

    def diagnose(self) -> None:
        return None
    
    """
    Non-required methods
    """

    def enricher_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        fit: bool = False,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Find features to engineer
        if fit or self.engineer_cols is None:
            self.find_engineer_cols(
                X=X.copy(), 
                y=y.copy(),
                debug=debug
            )

        # Find outlier features
        if self.add_outlier_features:
            if fit or self.outliers_dict is None:
                # Fit self.outliers_dict
                self.fit_outliers_dict(X=X)

            # Find outlier features
            outlier_features: pd.DataFrame = self.find_outlier_features(
                X=X[self.engineer_cols].copy()
            )
        else:
            outlier_features: pd.DataFrame = None

        # Find fibonachi features
        fib_features: pd.DataFrame = None

        # Find derivative features
        if self.add_derivative_features:
            if fit or self.derivatives_dict is None:
                # Fit self.derivatives_dict
                self.fit_derivatives_dict(X=X)

            # Find derivartive features
            deriv_features: pd.DataFrame = self.find_derivative_features(
                X=X[self.engineer_cols].copy()
            )
        else:
            deriv_features: pd.DataFrame = None

        # Find lag features
        if self.add_lag_features:
            # Find lag features
            lag_features: pd.DataFrame = self.find_lag_features(
                X=X[self.engineer_cols].copy()
            )
        else:
            lag_features: pd.DataFrame = None

        # Find rolling features
        if self.add_rolling_features:
            rolling_features: pd.DataFrame = self.find_rolling_features(
                X=X[self.engineer_cols].copy()
            )
        else:
            rolling_features: pd.DataFrame = None

        # Find EMA features
        if self.add_ema_features:
            ema_features: pd.DataFrame = self.find_ema_features(
                X=X[self.engineer_cols].copy()
            )
        else:
            ema_features: pd.DataFrame = None

        # Find temporal embedding features
        if self.add_temporal_embedding_features:
            te_features: pd.DataFrame = self.find_temporal_embedding_features(
                X=X[self.engineer_cols].copy()
            )
        else:
            te_features: pd.DataFrame = None

        # Concatenate DataFrames
        X = (
            pd.concat([
                X, 
                outlier_features,
                fib_features,
                deriv_features,
                lag_features,
                rolling_features,
                ema_features,
                te_features
            ], axis=1)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0)
        )

        return X, y
    
    def find_engineer_cols(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        debug: bool = False
    ) -> None:
        # Encode target
        if 'classification' in self.task:
            from sklearn.preprocessing import LabelEncoder

            # Encode target
            encoder: LabelEncoder = LabelEncoder()
            y[self.target] = encoder.fit_transform(y=y[self.target].values)

        if self.task in ['regression', 'binary_classification']:
            # Find correlation features
            correlation_features: List[str] = self.find_correlation_features(
                X=X, y=y,
                tf_quantile_threshold=None,
                ff_correl_threshold=None,
                debug=debug
            )
            
            # Find categorical features
            cat_features: List[str] = self.find_categorical_features(
                X=X, y=y,
                keep_percentage=None,
                debug=debug
            )

            # Define features to keep
            self.engineer_cols: List[str] = correlation_features + cat_features
        else:
            raise NotImplementedError('')

        LOGGER.info('New self.engineer_cols was found (%s):\n%s', len(self.engineer_cols), pformat(self.engineer_cols))
    
    def fit_outliers_dict(
        self,
        X: pd.DataFrame
    ) -> None:
        def find_threshold(col: str):
            # Calculate mean & std
            mean, std = X[col].mean(), X[col].std()

            # Return thresholds
            return mean - self.outlier_features_z * std, mean + self.outlier_features_z * std
        
        # Define outliers_cols
        outlier_cols: List[str] = list(X.select_dtypes(include=['number']))

        # Populate self.outliers_dict
        self.outliers_dict: Dict[str, Tuple[float]] = {
            col: find_threshold(col) for col in outlier_cols
        }

        LOGGER.info('New self.outliers_dict was populated:\n%s', pformat(self.outliers_dict))

    def find_outlier_features(
        self,
        X: pd.DataFrame,
        debug: bool = False
    ) -> pd.DataFrame:
        # Add feature columns
        for col_name in X.columns:
            bins = [
                -np.inf, 
                self.outliers_dict[col_name][0],
                self.outliers_dict[col_name][1],
                np.inf
            ]

            new_col_name = f'{col_name}__outlier'

            X[new_col_name] = pd.cut(
                X[col_name],
                bins=bins,
                labels=[f'{col_name}__low_outlier',
                        f'{col_name}__no_outlier',
                        f'{col_name}__top_outlier']
            )

            X[new_col_name] = X[new_col_name].fillna(f'{col_name}__no_outlier')
            # df[f'{col_name.split("_")[1]}_outlier'].fillna(f'{col_name.split("_")[1]}_no_outlier', inplace=True)
            X[new_col_name] = X[new_col_name].astype(str)

            if debug:
                LOGGER.debug('col_name %s: bins %s', col_name, bins)
                LOGGER.debug(
                    'X[[%s, %s]]:\n%s', 
                    col_name, new_col_name, X[[col_name, new_col_name]]
                )

        # Filter outlier features
        X = X.filter(like='__outlier', axis=1)

        if debug:
            LOGGER.debug('Outlier features:\n%s', X)

        return X

    def fit_derivatives_dict(
        self,
        X: pd.DataFrame
    ) -> None:
        def series_is_erratic(col_name: str) -> bool:
            # Calculate coefficient of variation
            cv = X[col_name].std() / X[col_name].mean()

            if cv > 1:
                return True
            return False
        
        # Populate self.derivative_dict with erratic & non-erratic series
        self.derivatives_dict: Dict[str, List[float]] = {
            'pct_change_cols': [],
            'diff_cols': []
        }
        
        for col_name in X.columns:
            if series_is_erratic(col_name):
                self.derivatives_dict['diff_cols'].append(col_name)
            else:
                self.derivatives_dict['pct_change_cols'].append(col_name)

    def find_derivative_features(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # Calculate first derivatives
        pct_change_df: pd.DataFrame = X[self.derivatives_dict['pct_change_cols']].pct_change().fillna(0)
        diff_df: pd.DataFrame = X[self.derivatives_dict['diff_cols']].diff().fillna(0)
        
        deriv1_df: pd.DataFrame = pd.concat([pct_change_df, diff_df], axis=1)

        # Calculate second derivatives
        deriv2_df: pd.DataFrame = deriv1_df.diff().fillna(0)

        # Calculate third derivatives
        deriv3_df: pd.DataFrame = deriv2_df.diff().fillna(0)

        # Rename columns
        deriv1_df: pd.DataFrame = deriv1_df.add_suffix('__deriv1')
        deriv2_df: pd.DataFrame = deriv2_df.add_suffix('__deriv2')
        deriv3_df: pd.DataFrame = deriv3_df.add_suffix('__deriv3')

        # Concatenate DataFrames
        deriv_df: pd.DataFrame = pd.concat([deriv1_df, deriv2_df, deriv3_df], axis=1)

        return deriv_df

    def find_lag_features(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # Calculate lag features
        lag_df: pd.DataFrame = pd.concat(
            [X.shift(periods=lag).add_suffix(f'__lag_{lag}').bfill() for lag in self.lags],
            axis=1
        )

        return lag_df
    
    def find_rolling_features(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        def helper_func(df: pd.DataFrame, window: int) -> pd.Series:
            # Find rolling transformations
            df = (
                df
                .rolling(window, min_periods=1)
                .agg({
                    column: ['mean', 'std', 'min', 'max'] 
                    for column in df.columns
                })
                .bfill()
            )

            # Rename columns
            df.columns = [f'__rolling_{window}_'.join(c) for c in df.columns]

            # Add min-max DataFrame
            max_cols = [c for c in df.columns if c.endswith('_max')]
            min_cols = [c for c in df.columns if c.endswith('_min')]
            min_max_df = (
                df
                .filter(items=max_cols) 
                - df
                .filter(items=min_cols)
                .rename(columns=lambda c: c.replace('_min', '_max'))
            ).rename(columns=lambda c: c.replace('_max', '_range'))
            
            return pd.concat([df, min_max_df], axis=1)
        
        # Calculate rolling features
        rolling_df: pd.DataFrame = pd.concat(
            [rolling_df(X.copy(), window=window) for window in self.rolling_windows],
            axis=1
        )

        return rolling_df

    def find_ema_features(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        def helper_func(df: pd.DataFrame, span: int) -> pd.Series:
            # Find EMA transformations
            df = (
                df
                .ewm(span=span, adjust=False, min_periods=1)
                .agg({
                    column: ['mean', 'std', 'min', 'max'] 
                    for column in df.columns
                })
                .bfill()
            )

            # Rename columns
            df.columns = [f'__ema_{span}_'.join(c) for c in df.columns]

            # Add min-max DataFrame
            max_cols = [c for c in df.columns if c.endswith('_max')]
            min_cols = [c for c in df.columns if c.endswith('_min')]
            min_max_df = (
                df
                .filter(items=max_cols) 
                - df
                .filter(items=min_cols)
                .rename(columns=lambda c: c.replace('_min', '_max'))
            ).rename(columns=lambda c: c.replace('_max', '_range'))
            
            return pd.concat([df, min_max_df], axis=1)
        
        # Calculate EMA features
        ema_df: pd.DataFrame = pd.concat(
            [helper_func(X.copy(), span=span) for span in self.rolling_windows],
            axis=1
        )

        return ema_df

    def find_temporal_embedding_features(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # Define initial temporal embedding features DataFrame
        te_features: pd.DataFrame = X[[self.datetime_col]].copy()

        # Add hour related features
        te_features['hour'] = X[self.datetime_col].dt.hour
        te_features['hour_sin'] = np.sin(2 * np.pi * te_features['hour'] / 24)
        te_features['hour_cos'] = np.cos(2 * np.pi * te_features['hour'] / 24)

        # Add day related features
        te_features['day'] = X[self.datetime_col].dt.day
        te_features['day_sin'] = np.sin(2 * np.pi * te_features['day'] / 31)
        te_features['day_cos'] = np.cos(2 * np.pi * te_features['day'] / 31)

        # Add day_of_week related features
        te_features['day_of_week'] = X[self.datetime_col].dt.day_of_week
        te_features['day_of_week_sin'] = np.sin(2 * np.pi * te_features['day_of_week'] / 7)
        te_features['day_of_week_cos'] = np.cos(2 * np.pi * te_features['day_of_week'] / 7)

        # Add week related features
        te_features['week'] = X[self.datetime_col].dt.week
        te_features['week_sin'] = np.sin(2 * np.pi * te_features['week'] / 52)
        te_features['week_cos'] = np.cos(2 * np.pi * te_features['week'] / 52)

        # Add holidays related features
        holidays = CountryHoliday(self.holiday_country, observed=True)
        te_features['is_holiday'] = te_features[self.datetime_col].apply(lambda x: x.date() in holidays)

        return te_features
