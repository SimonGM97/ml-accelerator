from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.transformers.transformer import Transformer
from ml_accelerator.utils.transformers.boruta_py import BorutaPy
from ml_accelerator.utils.timing.timing_helper import timing
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.config.env import Env

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.preprocessing import OneHotEncoder
import tsfresh as tsf
from nancorrmp.nancorrmp import NaNCorrMp
from scipy.stats import pearsonr, ttest_ind, chi2_contingency, f_oneway
import time
import gc
from copy import deepcopy
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
        ignore_features_p_value: float = Params.IGNORE_FEATURES_P_VALUE,
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
        self.ignore_features_p_value: float = ignore_features_p_value
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
        y: pd.DataFrame = None,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.selector_pipeline
        X, y = self.selector_pipeline(
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
        # Run self.selector_pipeline
        X, y = self.selector_pipeline(
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

    def selector_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        fit: bool = False,
        debug: bool = False
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
                debug=debug
            )

        # Filter features
        X = X.filter(items=self.selected_features)

        return X, y
    
    def find_base_model(self):
        if self.boruta_algorithm == 'random_forest':
            # Define hyper_parameters
            hyper_parameters = {
                'n_estimators': 100,
                'max_depth': 50,
                'n_jobs': Params.CPUS,
                'random_state': Env.get("SEED"),
                'verbose': -1
            }

            if self.task in ['binary_classification', 'multiclass_classification']:
                from sklearn.ensemble import RandomForestClassifier
                # Return vanilla Random Forest Classifier
                return RandomForestClassifier(**hyper_parameters)
            elif self.task in ['regression']:
                from sklearn.ensemble import RandomForestRegressor
                # Return vanilla Random Forest Regressor
                return RandomForestRegressor(**hyper_parameters)
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented.')
            
        elif self.boruta_algorithm == 'lightgbm':
            # Define hyper_parameters
            hyper_parameters = {
                'max_depth': 50,
                'n_estimators': 100,
                'random_state': Env.get("SEED"),
                'n_jobs': Params.CPUS,
                'verbose': -1
            }

            if self.task in ['binary_classification', 'multiclass_classification']:
                from lightgbm import LGBMClassifier
                # Return vanilla LGBM Classifier
                return LGBMClassifier(**hyper_parameters)
            elif self.task in ['regression']:
                from lightgbm import LGBMRegressor
                # Return vanilla LGBM Regressor
                return LGBMRegressor(**hyper_parameters)
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented.\n')
            
        elif self.boruta_algorithm == 'xgboost':
            # Define hyper_parameters
            hyper_parameters = {
                'max_depth': 50,
                'n_estimators': 100,
                'random_state': Env.get("SEED"),
                'nthread': Params.CPUS,
                'verbosity': -1
            }

            if self.task in ['binary_classification', 'multiclass_classification']:
                from xgboost import XGBClassifier
                # Return vanilla XGB Classifier
                return XGBClassifier(**hyper_parameters)
            elif self.task in ['regression']:
                from xgboost import XGBRegressor
                # Return vanilla XGB Regressor
                return XGBRegressor(**hyper_parameters)
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented.')
        else:
            raise NotImplementedError(f'Boruta algorithm "{self.boruta_algorithm}" has not been implemented.')
    
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
            verbose=-1, 
            random_state=int(Env.get("SEED"))
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
    
    def find_score_func(self):
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
            from sklearn.feature_selection import mutual_info_classif
            return mutual_info_classif
        else:
            raise NotImplementedError(f'Task "{self.task}" has not been implemented.')

    @timing
    def find_k_best_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        k_best: int = None,
        debug: bool = False
    ) -> List[str]:
        # Validate k_best
        if k_best is None:
            k_best = self.k_best
        
        # Instanciate and fit the Recursive Feature Eliminator
        selector = SelectKBest(
            score_func=self.find_score_func(), 
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
                raise NotImplementedError(f'Task "{self.task}" has not been implemented.')

        # Concatenate datasets
        full_df = pd.concat([X, y], axis=1)

        # Run KXY variable selection
        data_val_df: pd.DataFrame = full_df.kxy.variable_selection(
            self.target_column,
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
        *feature_lists: List[str],
        debug: bool = False
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
        
        LOGGER.info('Selected Features (%s).', len(selected_features)) #, pformat(selected_features))
        if debug:
            LOGGER.debug('Selected Features:\n%s', pformat(selected_features))

        return selected_features

    def find_ignore_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        p_value_threshold: float = None,
        debug: bool = False
    ) -> List[str]:
        # Validate p_value_threshold
        if p_value_threshold is None:
            p_value_threshold = self.ignore_features_p_value

        # Find intersection between X & y
        intersection = y.index.intersection(X.index)

        y = y.loc[intersection]
        X = X.loc[intersection]

        # Define empty ignore_features
        ignore_features: List[str] = []
    
        for feature in X.columns:
            if self.task == 'regression':
                if X[feature].dtype in ['string', 'object'] or len(X[feature].unique()) < 10:
                    # Perform ANOVA for categorical feature with continuous target
                    groups = [y[X[feature] == category] for category in np.unique(X[feature])]
                    _, p_value = f_oneway(*groups)
                else:
                    # Perform Pearson correlation for continuous target
                    corr, p_value = pearsonr(X[feature].values, y[self.target_column].values)
            
            elif self.task == 'binary_classification':
                if X[feature].dtype in ['string', 'object'] or len(X[feature].unique()) < 10:
                    # Chi-squared test for categorical features
                    contingency_table = pd.crosstab(X[feature], y)
                    _, p_value, _, _ = chi2_contingency(contingency_table)
                else:
                    # T-test for continuous features in binary classification
                    full_df: pd.DataFrame = pd.concat([X[[feature]], y[[self.target_column]]], axis=1)
                    y_unique = np.unique(y[self.target_column])
                    group1 = full_df.loc[full_df[self.target_column] == y_unique[0], feature]
                    group2 = full_df.loc[full_df[self.target_column] == y_unique[1], feature]
                    _, p_value = ttest_ind(group1, group2)
            
            elif self.task == 'multiclass_classification':
                if X[feature].dtype == 'object' or len(X[feature].unique()) < 10:
                    # Chi-squared test for categorical features
                    contingency_table = pd.crosstab(X[feature], y)
                    _, p_value, _, _ = chi2_contingency(contingency_table)
                else:
                    # ANOVA test for continuous features in multiclass classification
                    groups = [X[feature].values[y == cls] for cls in np.unique(y)]
                    _, p_value = f_oneway(*groups)
            
            else:
                raise ValueError("Invalid problem_type. Choose from 'regression', 'binary', or 'multiclass'.")
            
            if np.isnan(p_value) or p_value > p_value_threshold:
                ignore_features.append(feature)
            
            if debug:
                LOGGER.debug(
                    'p_value between %s (%s) and %s (%s): %s',
                    self.target_column, y[self.target_column].values.tolist()[:5],
                    feature, X[feature].values.tolist()[:5], p_value
                )
            
            if np.isnan(p_value):
                LOGGER.warning(
                    'p_value was nan (%s), when comparing %s (%s) with %s (%s)', 
                    p_value, self.target_column, y[self.target_column].values.tolist()[:5],
                    feature, X[feature].values.tolist()[:5]
                )
        
        return ignore_features

    """
    Deprecated methods
    """

    @staticmethod
    def prepare_correlation_datasets(
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> pd.DataFrame:
        # Find intersection between X & y
        intersection = y.index.intersection(X.index)

        y = y.loc[intersection]
        X = X.loc[intersection]

        # Filter non-numeric columns
        X = X.select_dtypes(include=['number'])

        return X, y
    
    @staticmethod
    def prepare_categorical_datasets(
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> pd.DataFrame:
        # Find intersection between X & y
        intersection = y.index.intersection(X.index)

        y = y.loc[intersection]
        X = X.loc[intersection]

        # Filter non-numeric columns
        X = X.select_dtypes(include=['category', 'object'])

        return X, y

    def find_target_feature_correlation_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        quantile_threshold: float = None,
        debug: bool = False
    ) -> List[str]:
        # Validate quantile_threshold
        if quantile_threshold is None:
            quantile_threshold = self.tf_quantile_threshold

        # Prepare X & y
        X, y = self.prepare_correlation_datasets(X=X, y=y)

        # Calculate Correlations with target
        tf_corr_df: pd.DataFrame = pd.DataFrame(columns=[self.target_column])
        for c in X.columns:
            tf_corr_df.loc[c] = [abs(y[self.target_column].corr(X[c]))]

        # Delete X & y from memory
        del X
        del y
        gc.collect()

        # Filter features
        threshold = np.quantile(tf_corr_df[self.target_column].dropna(), quantile_threshold)
        tf_corr_df = tf_corr_df.loc[tf_corr_df[self.target_column] >= threshold]

        # Filter features
        selected_features: List[str] = (
            tf_corr_df
            .sort_values(by=[self.target_column], ascending=False)
            .index
            .tolist()
        )

        if debug:
            LOGGER.debug("threshold: %s", threshold)
            LOGGER.debug("tf_corr_df.tail():\n%s\n", tf_corr_df.tail())
            LOGGER.debug(
                'Features selected with Target Feature Correlation Filter (%s):\n%s', 
                len(selected_features), pformat(selected_features)
            )

        return selected_features
    
    def find_feature_feature_correlation_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        correl_threshold: float = None,
        debug: bool = False
    ) -> List[str]:
        # Validate quantile
        if correl_threshold is None:
            correl_threshold = self.ff_correl_threshold

        # Prepare X & y
        X, _ = self.prepare_correlation_datasets(X=X, y=y)

        # Calculate Correlations among features
        t0 = time.time()
        n_jobs = Params.CPUS

        ff_corr_df: pd.DataFrame = (
            NaNCorrMp
            .calculate(X, n_jobs=n_jobs)
            .abs()
            * 100
        ).fillna(100)

        if debug:
            LOGGER.debug("ff_corr_df took %s sec to be created.", int(time.time()-t0))

        # Delete X & y from memory
        del X
        del y
        gc.collect()

        # Define initial selected_features
        selected_features: List[str] = deepcopy(ff_corr_df.columns)

        # Filter selected_features
        i = 0
        while i < len(selected_features):
            # Define feature to keep
            keep_feature: str = selected_features[i]

            # Define features to drop
            drop_features: List[str] = ff_corr_df.loc[
                (ff_corr_df[keep_feature] < 100) &
                (ff_corr_df[keep_feature] >= correl_threshold * 100)
            ][keep_feature].index.tolist()
            
            # Drop features
            if len(drop_features) > 0:
                selected_features = [c for c in selected_features if c not in drop_features]
            i += 1
        
        if debug:
            LOGGER.debug(
                'Features selected with Feature Feature Correlation Filter (%s):\n%s', 
                len(selected_features), pformat(selected_features)
            )

        # Remove ff_corr_df from memort
        del ff_corr_df
        gc.collect()

        return selected_features
    
    @timing
    def find_correlation_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        tf_quantile_threshold: float = None,
        ff_correl_threshold: float = None,
        debug: bool = False
    ) -> List[str]:
        # Find target-features correlation features
        tf_correl_features: List[str] = self.find_target_feature_correlation_features(
            X=X, y=y, 
            quantile_threshold=tf_quantile_threshold,
            debug=debug
        )

        # Apply target-features correlation filter
        X = X[tf_correl_features]

        # Find feature-feature correlation features
        ff_correl_features: List[str] = self.find_feature_feature_correlation_features(
            X=X, y=y,
            correl_threshold=ff_correl_threshold,
            debug=debug
        )

        return ff_correl_features
    
    @timing
    def find_categorical_features(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        keep_percentage: float = None,
        debug: bool = False
    ) -> List[str]:
        # Prepare X & y
        X, _ = self.prepare_categorical_datasets(X=X, y=y)

        if X.shape[1] > 1:
            # Define initial columns
            initial_cols = X.columns.tolist().copy()

            # Instanciate OneHotEncoder
            OHE: OneHotEncoder = OneHotEncoder(handle_unknown='ignore')

            # Fit OHE
            OHE.fit(X[initial_cols])

            # Apply OHE
            X = pd.DataFrame(
                OHE.transform(X[initial_cols]).toarray(),
                columns=OHE.get_feature_names_out(initial_cols),
                index=X.index
            )

            # Apply SeleckKBest
            k = max([1, int(keep_percentage * X.shape[1])])
            selector = SelectKBest(score_func=self.find_score_func(), k=k)
            selector.fit(X, y)
            
            # Get the selected features
            selected_features: List[str] = X.columns[selector.get_support()].tolist()

            # Delete X & y from memory
            del X
            del y
            gc.collect()

            selected_features: List[str] = [c for c in initial_cols if any(c in c2 for c2 in selected_features)]
        else:
            selected_features: List[str] = []

        if debug:
            LOGGER.debug(
                'Features selected with Categorical Filter (%s):\n%s', 
                len(selected_features), pformat(selected_features)
            )
        
        return selected_features
