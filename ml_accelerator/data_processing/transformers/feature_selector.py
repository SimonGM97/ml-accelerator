from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.transformers.transformer import Transformer
from ml_accelerator.utils.transformers.boruta_py import BorutaPy
from ml_accelerator.utils.timing.timing_helper import timing
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest
import tsfresh as tsf
import os
from pprint import pformat
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
        k_best: int = Params.K_BEST,
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
        self.k_best: int = k_best
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

            # Find RFE features
            rfe_features: List[str] = self.find_rfe_features(X=X, y=y)

            # Find K-Best features
            k_best_features: List[str] = self.find_k_best_features(X=X, y=y)

            # Find TSFresh features
            tsf_features: List[str] = self.find_tsfresh_features(X=X, y=y)

            # Find KXY features
            # kxy_features: List[str] = self.find_kxy_features(X=X, y=y)

            # Concatenate selected features
            self.selected_features: List[str] = self.concatenate_selected_features(
                X,
                boruta_features,
                rfe_features,
                k_best_features,
                tsf_features,
                # kxy_features
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
        y: pd.DataFrame,
        debug: bool = False
    ) -> List[str]:
        # Instanciate dummy model
        model = self.find_base_model()

        # Instanciate and fit the BorutaPy selector
        selector = BorutaPy(
            model,
            max_iter=70,
            verbose=-1, 
            random_state=int(os.environ.get("SEED"))
        )

        selector.fit(X.values, y.values)

        # Extract features selected by Boruta
        boruta_features: List[str] = X.columns[selector.support_].tolist()

        if debug:
            LOGGER.debug(
                'Features selected with Boruta (%s):\n%s', 
                len(boruta_features), pformat(boruta_features)
            )

        return boruta_features
    
    @timing
    def find_rfe_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        rfe_n: int = None,
        debug: bool = False
    ) -> List[str]:
        # Validate rfe_n
        if rfe_n is None:
            rfe_n = self.rfe_n
        
        # Instanciate dummy model
        model = self.find_base_model()
        
        # Instanciate and fit the Recursive Feature Eliminator
        selector = RFE(
            model, 
            n_features_to_select=rfe_n, 
            step=0.01
        )

        selector.fit(X.values, y.values)

        # Extract features selected by RFE
        rfe_features: List[str] = X.columns[selector.support_].tolist()

        if debug:
            LOGGER.debug(
                'Features selected with RFE (%s):\n%s', 
                len(rfe_features), pformat(rfe_features)
            )

        return rfe_features
    
    @timing
    def find_k_best_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        k_best: int = None,
        debug: bool = False
    ) -> List[str]:
        def find_score_func():
            """
            `score_func`: una función que devuelve algún score entre X e y:
                - Para regresión: f_regression, mutual_info_regression
                - Para clasificación: f_classif, mutual_info_classif

            `F-Test`: tanto f_regression como f_classif se basan en un test estadístico 
            llamado F-Test, en donde se compara la performance de un modelo linear que incluye 
            a X cómo variable regresora, respecto de un modelo que sólo tiene intercept.

            `Mutual Information`: la información mutua es una métrica que captura relaciones no
            lineales entre variables.
            """
            if self.task in ['binary_classification', 'multiclass_classification']:
                from sklearn.feature_selection import mutual_info_regression
                return mutual_info_regression
            elif self.task in ['regression']:
                from sklearn.feature_extraction import mutual_info_classif
                return mutual_info_classif
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented yet.\n')
        
        # Validate k_best
        if k_best is None:
            k_best = self.k_best
        
        # Instanciate and fit the Recursive Feature Eliminator
        selector = SelectKBest(
            score_func=find_score_func(), 
            k=k_best
        )

        selector.fit(X.values, y.values)

        # Extract features selected by RFE
        k_best_features: List[str] = X.columns[selector.get_support()].tolist()

        if debug:
            LOGGER.debug(
                'Features selected with SelectKBest (%s):\n%s', 
                len(k_best_features), pformat(k_best_features)
            )

        return k_best_features
    
    @timing
    def find_tsfresh_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        tsfresh_p_value: float = None,
        tsfresh_n: int = None,
        debug: bool = False
    ) -> List[str]:
        debug=True
        # Validate tsfresh_p_value & tsfresh_n
        if tsfresh_p_value is None:
            tsfresh_p_value = self.tsfresh_p_value
        if tsfresh_n is None:
            tsfresh_n = self.tsfresh_n

        # Run the TSFresh relevance table calculation
        relevance_table: pd.DataFrame = tsf.feature_selection.relevance.calculate_relevance_table(
            X=X, y=y.squeeze(), ml_task='auto', fdr_level=self.tsfresh_p_value, n_jobs=Params.GPUS
        )

        # Sort & filter relevance_table 
        # relevance_table = relevance_table[relevance_table['relevant']].sort_values("p_value")
        # relevance_table = relevance_table.loc[relevance_table['p_value'] < tsfresh_p_value]

        # Select relevant features based on the TSFresh relevance table
        # tsf_features: List[str] = list(relevance_table["feature"].values)
        tsf_features: List[str] = relevance_table[relevance_table.relevant].feature.tolist()

        # Reduce features (if needed)
        if tsfresh_n is not None:
            # top_k = int(perc * len(tsf_features))
            return tsf_features[:tsfresh_n]
        
        if debug:
            LOGGER.debug(
                'Features selected with TSFresh (%s):\n%s', 
                len(tsf_features), pformat(tsf_features)
            )
        
        return tsf_features

    @timing
    def find_kxy_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        debug: bool = False
    ) -> List[str]:
        def find_problem_type():
            if self.task in ['binary_classification', 'multiclass_classification']:
                return 'classification'
            elif self.task in ['regression']:
                return 'regression'
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented yet.\n')

        # Concatenate datasets
        full_df = pd.concat([X, y], axis=1)

        # Run KXY variable selection
        data_val_df: pd.DataFrame = full_df.kxy.variable_selection(
            self.target,
            problem_type=find_problem_type(),
            anonymize=True
        )

        # Extract features selected by KXY
        kxy_features: List[str] = data_val_df['Variable'].tolist()[1:]

        if debug:
            LOGGER.debug(
                'Features selected with KXY (%s):\n%s', 
                len(kxy_features), pformat(kxy_features)
            )

        return kxy_features

    def concatenate_selected_features(
        self,
        X: pd.DataFrame,
        *feature_lists: List[str]
    ) -> List[str]:
        # Concatenate features
        selected_features: Set[str] = set()
        for feature_list in feature_lists:
            selected_features.update(set(feature_list))

        # Order concatenated features
        selected_features: List[str] = [
            col for col in X.columns if col in selected_features
        ]

        # Filter features
        if len(selected_features) > self.max_features:
            LOGGER.warning(
                'Selected features are larger than allowed: %s > %s', 
                len(selected_features), self.max_features
            )

            # Keep first n features
            ignored_features = selected_features[self.max_features:]
            selected_features = selected_features[:self.max_features]
            
            LOGGER.info(
                'Selected features ignored (%s): %s', 
                len(ignored_features), ignored_features
            )
        
        LOGGER.info('Selected Features (%s):\n%s', len(selected_features), pformat(selected_features))

        return selected_features