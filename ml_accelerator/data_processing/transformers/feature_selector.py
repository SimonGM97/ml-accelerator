from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.transformers.transformer import Transformer
from ml_accelerator.utils.transformers.boruta_py import BorutaPy
from ml_accelerator.utils.timing.timing_helper import timing
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from typing import List, Tuple, Set


# Get logger
LOGGER = get_logger(name=__name__)


class FeatureSelector(Transformer):

    # Pickled attrs
    pickled_attrs = [
        'selected_features'
    ]

    def __init__(
        self,
        transformer_id: str = None,
        forced_features: List[str] = Params.FORCED_FEATURES,
        target_feature_quantile: float = Params.TARGET_FEATURE_QUANTILE,
        feature_feature_quantile: float = Params.FEATURE_FEATURE_QUANTILE,
        boruta_algorithm: str = Params.BORUTA_ALGORITHM,
        rfe_n: int = Params.RFE_N,
        b_best: int = Params.K_BEST,
        tsfresh_p_value: float = Params.TSFRESH_P_VALUE,
        tsfresh_n: int = Params.TSFRESH_N,
        max_features: int = Params.MAX_FEATURES
    ) -> None:
        # Instanciate parent classes
        super().__init__(transformer_id=transformer_id)

        # Set non-load attributes
        self.forced_features: List[str] = forced_features
        self.target_feature_quantile: float = target_feature_quantile
        self.feature_feature_quantile: float = feature_feature_quantile
        self.boruta_algorithm: str = boruta_algorithm
        self.rfe_n: int = rfe_n
        self.b_best: int = b_best
        self.tsfresh_p_value: float = tsfresh_p_value
        self.tsfresh_n: int = tsfresh_n
        self.max_features: int = max_features

        # Set load attributes
        self.selected_features: List[str] = None

    """
    Required methods (from Transformer abstract methods)
    """

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.selector_pipeline
        X, y = self.selector_pipeline(X=X, y=y, fit=False)
        
        return X, y

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.selector_pipeline
        X, y = self.selector_pipeline(X=X, y=y, fit=True)

        return X, y

    def diagnose(self) -> None:
        return None
    
    """
    Non-required methods
    """

    def selector_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        fit: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Find selected features
        if fit:
            # Find boruta features
            boruta_features: List[str] = self.find_boruta_features(X=X, y=y)

            # Concatenate selected features
            self.selected_features: List[str] = self.concatenate_selected_features(
                X=X,
                boruta_features=boruta_features
            )

        # Filter features
        X = X.filter(items=self.selected_features)

        return X, y
    
    def find_base_model(self):
        if self.boruta_algorithm == 'random_forest':
            if self.task in ['binary_classification', 'multiclass_classification']:
                from sklearn.ensemble import RandomForestClassifier
                # Return vanilla Random Forest Classifier
                return RandomForestClassifier()
            elif self.task in ['regression']:
                from sklearn.ensemble import RandomForestRegressor
                # Return vanilla Random Forest Regressor
                return RandomForestRegressor()
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented yet.\n')
        elif self.boruta_algorithm == 'lightgbm':
            if self.task in ['binary_classification', 'multiclass_classification']:
                from lightgbm import LGBMClassifier
                # Return vanilla LGBM Classifier
                return LGBMClassifier(verbose=-1)
            elif self.task in ['regression']:
                from lightgbm import LGBMRegressor
                # Return vanilla LGBM Regressor
                return LGBMRegressor(verbose=-1)
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented yet.\n')
        elif self.boruta_algorithm == 'xgboost':
            if self.task in ['binary_classification', 'multiclass_classification']:
                from xgboost import XGBClassifier
                # Return vanilla XGB Classifier
                return XGBClassifier()
            elif self.task in ['regression']:
                from xgboost import XGBRegressor
                # Return vanilla XGB Regressor
                return XGBRegressor()
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented yet.\n')
        else:
            raise NotImplementedError(f'Boruta algorithm "{self.boruta_algorithm}" has not been implemented yet.\n')
    
    @timing
    def find_boruta_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> List[str]:
        # Instanciate dummy model
        model = self.find_base_model()

        # Instanciate and fit the BorutaPy estimator
        boruta_estimator = BorutaPy(
            model,
            max_iter=70,
            verbose=-1, 
            random_state=23111997
        )

        boruta_estimator.fit(
            np.array(X), 
            np.array(y)
        )

        # Extract features selected by Boruta
        boruta_features: List[str] = X.columns[boruta_estimator.support_].tolist()

        return boruta_features
    
    def concatenate_selected_features(
        self,
        X: pd.DataFrame,
        boruta_features: List[str]
    ) -> List[str]:
        # Concatenate features
        concat_features: Set[str] = set(
            boruta_features
        )

        # Order concatenated features
        concat_features: List[str] = [
            col for col in X.columns if col in concat_features
        ]

        # Filter features
        if len(concat_features) > self.max_features:
            LOGGER.warning(
                'Selected features are larger than allowed: %s > %s', 
                len(concat_features), self.max_features
            )

            # Keep first n features
            concat_features = concat_features[:self.max_features]
            ignored_features = concat_features[self.max_features:]
            
            LOGGER.info(
                'Selected features ignored (%s): %s', 
                len(ignored_features), ignored_features
            )

        return concat_features