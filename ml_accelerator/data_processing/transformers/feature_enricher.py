from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.transformers.transformer import Transformer
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pprint import pformat


# Get logger
LOGGER = get_logger(name=__name__)


class FeatureEnricher(Transformer):

    # Pickled attrs
    pickled_attrs = [
        'outlier_cols',
        'outliers_dict'
    ]

    def __init__(
        self,
        transformer_id: str = None,
        add_outlier_features: bool = Params.ADD_OUTLIER_FEATURES,
        outlier_features_z: float = Params.OUTLIER_FEATURES_Z,
        fibonachi_features: List[str] = Params.FIBONACHI_FEATURES,
        derivative_features: List[str] = Params.DERIVATIVE_FEATURES,
        lag_features: List[str] = Params.LAG_FEATURES,
        rolling_features: List[str] = Params.ROLLING_FEATURES,
        ema_features: List[str] = Params.EMA_FEATURES,
        temporal_embedding_features: List[str] = Params.TEMPORAL_EMBEDDING_FEATURES,
        temporal_based_features: List[str] = Params.TEMPORAL_BASED_FEATURES,
        holiday_features: List[str] = Params.HOLIDAY_FEATURES,
        holiday_country: List[str] = Params.HOLIDAY_COUNTRY
    ) -> None:
        # Instanciate parent classes
        super().__init__(transformer_id=transformer_id)

        # Set non-load attributes
        self.add_outlier_features: bool = add_outlier_features
        self.outlier_features_z: float = outlier_features_z
        self.fibonachi_features: List[str] = fibonachi_features
        self.derivative_features: List[str] = derivative_features
        self.lag_features: List[str] = lag_features
        self.rolling_features: List[str] = rolling_features
        self.ema_features: List[str] = ema_features
        self.temporal_embedding_features: List[str] = temporal_embedding_features
        self.temporal_based_features: List[str] = temporal_based_features
        self.holiday_features: List[str] = holiday_features
        self.holiday_country: List[str] = holiday_country

        # Set attributes to load
        self.outlier_cols: List[str] = None
        self.outliers_dict: Dict[str, Tuple[float]] = None

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
        fit: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Find outlier features
        if self.add_outlier_features:
            if fit:
                # Find self.outlier_cols
                self.find_outlier_cols(X=X)

                # Fit self.outliers_dict
                self.fit_outliers_dict(X=X)

            # Find outlier features
            outlier_features: pd.DataFrame = self.find_outlier_features(X=X.copy())
        else:
            outlier_features: pd.DataFrame = None

        # Find fibonachi features
        fib_features: pd.DataFrame = None

        # Find derivative features
        deriv_features: pd.DataFrame = None

        # Find lag features
        lag_features: pd.DataFrame = None

        # Find rolling features
        rolling_features: pd.DataFrame = None

        # Find EMA features
        ema_features: pd.DataFrame = None

        # Find temporal embedding features
        te_features: pd.DataFrame = None

        # Find holiday features
        holiday_features: pd.DataFrame = None

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
                te_features,
                holiday_features
            ], axis=1)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0)
        )

        return X, y
    
    def find_outlier_cols(
        self,
        X: pd.DataFrame
    ) -> None:
        # Find outlier_cols
        self.outlier_cols: List[str] = list(X.select_dtypes(include=['number']))
    
    def fit_outliers_dict(
        self,
        X: pd.DataFrame
    ) -> None:
        def find_threshold(col: str):
            # Calculate mean & std
            mean, std = X[col].mean(), X[col].std()

            # Return thresholds
            return mean - self.outlier_features_z * std, mean + self.outlier_features_z * std
        
        # Populate self.outliers_dict
        self.outliers_dict: Dict[str, Tuple[float]] = {
            col: find_threshold(col) for col in self.outlier_cols
        }

        LOGGER.info('New self.outliers_dict was populated:\n%s', pformat(self.outliers_dict))

    def find_outlier_features(
        self,
        X: pd.DataFrame,
        debug: bool = False
    ) -> pd.DataFrame:
        # Add feature columns
        for col_name in self.outlier_cols:
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
