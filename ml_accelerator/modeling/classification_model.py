from config.params import Params
from ml_accelerator.modeling.model import Model
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.special import expit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    make_scorer, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve
)

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


class ClassificationModel(Model):
    """
    Class designed to homogenize the methods for building, evaluating, tracking & registering multiple types
    of ML classification models with different flavors/algorithms & hyperparameters, in a unified fashion. 
    """

    # Pickled attrs
    pickled_attrs = [
        # Register Parameters
        'model_id',
        'version',
        'stage',

        # General Parameters
        'algorithm',
        'hyper_parameters',
        'fitted',

        # Feature Importance
        'shap_values',
        'importance_method',

        # Classification Parameters
        'cutoff',
        
        'f1_score',
        'precision_score',
        'recall_score',
        'roc_auc_score',
        'accuracy_score',

        'confusion_matrix',
        'fpr', 
        'tpr',
        'thresholds',

        'cv_scores',
        'test_score'
    ]

    # csv attrs
    csv_attrs = [
        'feature_importance_df'
    ]

    # Parquet attrs
    parquet_attrs = []

    # Metrics
    metric_names = [
        'f1_score',
        'precision_score',
        'recall_score',
        'roc_auc_score',
        'accuracy_score'
    ]

    def __init__(
        self,
        model_id: str = None,
        version: int = 1,
        stage: str = 'staging',
        algorithm: str = None,
        hyper_parameters: dict = {},
        target: str = None,
        selected_features: List[str] = None,
        importance_method: str = None,
        cutoff: float = None
    ) -> None:
        # Instanciate parent class to inherit attrs & methods
        super().__init__(
            model_id=model_id,
            version=version,
            stage=stage,
            algorithm=algorithm,
            hyper_parameters=hyper_parameters,
            target=target,
            selected_features=selected_features,
            importance_method=importance_method
        )

        # Correct self.hyperparameters
        self.hyper_parameters: dict = self.correct_hyper_parameters(
            hyper_parameters=hyper_parameters,
            debug=False
        )

        # Classification parameters
        self.cutoff: float = cutoff

        self.f1_score: float = 0
        self.precision_score: float = 0
        self.recall_score: float = 0
        self.roc_auc_score: float = 0
        self.accuracy_score: float = 0

        self.confusion_matrix: np.ndarray = None
        self.fpr: np.ndarray = None 
        self.tpr: np.ndarray = None 
        self.thresholds: np.ndarray = None 
        
        self.cv_scores: np.ndarray = np.array([])
        self.test_score: float = 0

    def correct_hyper_parameters(
        self,
        hyper_parameters: dict,
        debug: bool = False
    ) -> dict:
        """
        Method that completes pre-defined hyperparameters.

        :param `hyper_parameters`: (dict) hyper_parameters that might not contain pre-defined hyperparameters.
        :param `debug`: (bool) Wether or not to show output hyper_parameters for debugging purposes.

        :return: (dict) hyper_parameters containing pre-defined hyperparameters.
        """
        if self.algorithm == 'random_forest':
            hyper_parameters.update(**{
                'class_weight': Params.CLASS_WEIGHT,
                'oob_score': False,
                'n_jobs': -1,
                'random_state': 23111997
            })

        elif self.algorithm == 'lightgbm':
            hyper_parameters.update(**{
                "scale_pos_weight": Params.CLASS_WEIGHT[1] / Params.CLASS_WEIGHT[0],
                "importance_type": 'split',
                "random_state": 23111997,
                "verbose": -1,
                "n_jobs": -1
            })

        elif self.algorithm == 'xgboost':
            hyper_parameters.update(**{
                "scale_pos_weight": Params.CLASS_WEIGHT[1] / Params.CLASS_WEIGHT[0],
                "verbosity": 0,
                "use_rmm": True,
                "device": 'cuda', # 'cpu', 'cuda' # cuda -> GPU
                "nthread": -1,
                "n_gpus": -1,
                "max_delta_step": 0,
                "gamma": 0,
                "subsample": 1, # hp.uniform('xgboost.subsample', 0.6, 1)
                "sampling_method": 'uniform',
                "random_state": 23111997,
                "n_jobs": -1
            })

        # Show hyper-parameters
        if debug and hyper_parameters is not None:
            show_params: str = "hyper_parameters:\n{"

            for key in hyper_parameters.keys():
                show_params += f"    '{key}': {hyper_parameters[key]} ({type(hyper_parameters[key])})"
            
            show_params += "}\n"

            LOGGER.debug(show_params)

        return hyper_parameters
    
    def build(
        self,
        debug: bool = False
    ) -> None:
        """
        Method to instanciate the specified ML classification model, based on the model flavor/alrorithm
        & hyper-parameters.

        :param `debug`: (bool) Wether or not to show output hyper_parameters for debugging purposes.
        """
        if self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(**self.hyper_parameters)
        
        elif self.algorithm == 'lightgbm':
            self.model = LGBMClassifier(**self.hyper_parameters)

        elif self.algorithm == 'xgboost':
            self.model = XGBClassifier(**self.hyper_parameters)

        else:
            raise Exception(f'Invalid algorithm: {self.algorithm}!')
        
        self.fitted = False
        
        if debug:
            LOGGER.debug('self.model: %s', self.model)

    def fit(
        self,
        y_train: pd.DataFrame = None,
        X_train: pd.DataFrame = None
    ) -> None:
        """
        Method to fit self.model.

        :param `y_train`: (pd.DataFrame) Binary & balanced train target.
        :param `X_train`: (pd.DataFrame) Train features.
        """
        self.model.fit(
            X_train.values.astype(float), 
            y_train[self.target].values.astype(int).ravel()
        )
        
        # Update Version
        if not self.fitted:
            self.fitted = True
            self.version = 1
        else:
            self.version += 1

    def predict(
        self,
        X: pd.DataFrame,
        cutoff: float = None
    ) -> np.ndarray:
        """
        Method for realizing new category inferences, based on the cutoff and the predicted
        probability.

        :param `X`: (pd.DataFrame) New features to make inferences on.
        :param `cutoff`: (float) Probability threshold at which to infer a class 1.
            - Note: if None, cutoff is set to the self.cutoff value.

        :return: (np.ndarray) New category inferences.
        """
        # Set up cutoff
        if cutoff is None:
            cutoff = self.cutoff

        # Predict probabilities
        y_score = self.predict_proba(X=X)

        # Define class based on the self.cutoff
        y_pred = np.where(y_score > cutoff, 1, 0)

        # Return predictions
        return y_pred
    
    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Method for realizing new probabilistic inferences.

        :param `X`: (pd.DataFrame) New features to make inferences on.

        :return: (np.ndarray) New probabilistic inferences.
        """
        y_score = self.model.predict_proba(X.values.astype(float))[:, 1]

        if (
            self.algorithm == 'xgboost' 
            and 'objective' in self.hyper_parameters.keys() 
            and self.hyper_parameters['objective'] == 'binary:logitraw'
        ):
            # Transform logits into probabilities
            return expit(y_score)
        return y_score

    def evaluate_val(
        self,
        y_train: pd.DataFrame,
        X_train: pd.DataFrame,
        eval_metric: str,
        splits: int,
        debug: bool = False
    ) -> None:
        """
        Method that will define a score metric (based on the eval_metric parameter) and will leverage
        the cross validation technique to obtain the validation scores.

        :param `y_train`: (pd.DataFrame) binary & balanced train target.
        :param `X_train`: (pd.DataFrame) Train features.
        :param `eval_metric`: (str) Metric to measure on each split of the cross validation.
        :param `splits`: (int) Number of splits to perform in the cross validation.
        :param `debug`: (bool) Wether or not to show self.cv_scores, for debugging purposes.
        """
        # Define scorer
        if eval_metric == 'f1_score':
            scorer = make_scorer(f1_score)
        elif eval_metric == 'precision':
            scorer = make_scorer(precision_score)
        elif eval_metric == 'recall':
            scorer = make_scorer(recall_score)
        elif eval_metric == 'roc_auc':
            scorer = make_scorer(roc_auc_score)
        elif eval_metric == 'accuracy':
            scorer = make_scorer(accuracy_score)
        else:
            raise Exception(f'Invalid "eval_metric": {eval_metric}.\n\n')

        # Evaluate Model using Cross Validation
        self.cv_scores = cross_val_score(
            self.model, 
            X_train.values.astype(float),
            y_train[self.target].values.astype(int).ravel(),
            cv=splits, 
            scoring=scorer
        )
        
        if debug:
            LOGGER.debug('self.cv_scores: %s\n', self.cv_scores)

    def evaluate_test(
        self,
        y_test: pd.DataFrame,
        X_test: pd.DataFrame,
        eval_metric: str,
        debug: bool = False
    ) -> None:
        """
        Method that will predict test set values and define the following test metrics:
            - self.f1_score
            - self.precision_score
            - self.recall_score
            - self.roc_auc_score
            - self.accuracy_score
            - self.test_score (utilized to define champion model)

        :param `y_test`: (pd.DataFrame) Binary & un-balanced test target.
        :param `X_test`: (pd.DataFrame) Test features.
        :param `eval_metric`: (str) Metric utilized to define the self.test_score attribute.
        :param `debug`: (bool) Wether or not to show self.test_score, for debugging purposes.
        """
        # Prepare y_test
        y_test = y_test.values.astype(int)

        # Predict test values
        y_pred = self.predict(X=X_test)

        # Predict probability
        y_score = self.predict_proba(X=X_test)

        # Evaluate F1 Score
        self.f1_score = f1_score(y_test, y_pred)

        # Evaluate Precision Score
        self.precision_score = precision_score(y_test, y_pred)

        # Evaluate Recall Score
        self.recall_score = recall_score(y_test, y_pred)

        # ROC AUC Score
        self.roc_auc_score = roc_auc_score(y_test, y_score)

        # Accuracy Score
        self.accuracy_score = accuracy_score(y_test, y_pred)

        # Confusion Matrix
        self.confusion_matrix = confusion_matrix(y_test, y_pred)

        # ROC Curve
        self.fpr, self.tpr, self.thresholds = roc_curve(y_test, y_score)

        # Define test score
        if eval_metric == 'f1_score':
            self.test_score = self.f1_score
        elif eval_metric == 'precision':
            self.test_score = self.precision_score
        elif eval_metric == 'recall':
            self.test_score = self.recall_score
        elif eval_metric == 'roc_auc':
            self.test_score = self.roc_auc_score
        elif eval_metric == 'accuracy':
            self.test_score = self.accuracy_score
        else:
            raise Exception(f'Invalid "eval_metric": {eval_metric}.\n\n')

        if debug:
            LOGGER.debug('self.test_score (%s): %s', eval_metric, self.test_score)

    def optimize_cutoff(
        self,
        y_test: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> None:
        """
        Method that will iteratively loop over different cutoff configurations in order to find the most 
        optimal one.

        :param `y_test`: (pd.DataFrame) Binary & un-balanced test target.
        :param `X_test`: (pd.DataFrame) Test features.
        """
        # Define performances
        performances = {}
        
        # Loop through different cutoffs  
        for cutoff in np.arange(0.3, 0.61, 0.01):
            # Find new predictions (dependant on the cutoff)
            y_pred = self.predict(X=X_test, cutoff=cutoff)

            # Calculate F1 Score
            score = f1_score(y_test, y_pred)

            # Assign performances
            performances[cutoff] = score
        
        # Find optimal cutoff
        optimal_cutoff, optimal_f1_score = max(performances.items(), key=lambda x: x[1])

        # Assign cutoff
        self.cutoff = optimal_cutoff

        LOGGER.info('Optimal cutoff for %s: %s (F1 Score %s).\n', 
                    self.model_id, round(self.cutoff, 3), optimal_f1_score)