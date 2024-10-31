from ml_accelerator.config.params import Params
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.config.env import Env

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
LOGGER = get_logger(name=__name__)


class ClassificationModel(Model):
    """
    Class designed to homogenize the methods for building, evaluating, tracking & registering multiple types
    of ML classification models with different flavors/algorithms & hyperparameters, in a unified fashion. 
    """

    # Pickled attrs
    pickled_attrs: List[str] = [
        # Register Parameters
        'model_id',
        'version',
        'stage',

        # General Parameters
        'task',
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
    csv_attrs: List[str] = [
        'feature_importance_df'
    ]

    # Parquet attrs
    parquet_attrs: List[str] = []

    # Metrics
    metric_names: List[str] = [
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
        stage: str = 'development',
        task: str = Params.TASK,
        algorithm: str = None,
        hyper_parameters: dict = None,
        target_column: str = Params.TARGET_COLUMN,
        selected_features: List[str] = None,
        optimization_metric: str = Params.OPTIMIZATION_METRIC,
        importance_method: str = Params.IMPORTANCE_METHOD,
        cutoff: float = None
    ) -> None:
        # Instanciate parent class to inherit attrs & methods
        super().__init__(
            model_id=model_id,
            version=version,
            stage=stage,
            task=task,
            algorithm=algorithm,
            hyper_parameters=hyper_parameters,
            target_column=target_column,
            selected_features=selected_features,
            optimization_metric=optimization_metric,
            importance_method=importance_method
        )

        # Correct self.hyperparameters
        if self.hyper_parameters is not None:
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
        # Define default parameters
        default_params: dict = {
            "class_weight": 'balanced' if Params.CLASS_WEIGHT is None else Params.CLASS_WEIGHT,
            "scale_pos_weight": None if Params.CLASS_WEIGHT is None else Params.CLASS_WEIGHT[1] / Params.CLASS_WEIGHT[0],
            "importance_type": 'gain',
            "verbose": -1,
            "random_state": int(Env.get("SEED")),
            "n_jobs": -1
        }

        # Filter default parameters
        def extract_available_params():
            if self.algorithm == 'random_forest':
                return list(RandomForestClassifier().get_params().keys())
            elif self.algorithm == 'lightgbm':
                return list(LGBMClassifier().get_params().keys())
            elif self.algorithm == 'xgboost':
                return list(XGBClassifier().get_params().keys())
            
            raise NotImplementedError(f'Algorithm "{self.algorithm}" has not yet been implemented.\n')

        available_params: List[str] = extract_available_params()

        default_params: dict = {
            param_name: default_params[param_name] for param_name in default_params
            if param_name in available_params
        }

        # Update default hyperparameters
        hyper_parameters.update(**default_params)

        if self.algorithm == 'random_forest':
            hyper_parameters.update(**{
                "verbose": 0,
                'oob_score': False
            })

        elif self.algorithm == 'lightgbm':
            if self.task == 'binary_classification':
                hyper_parameters.update(**{
                    "objective": 'binary',
                    "metric": 'binary_logloss' # auc
                })
            elif self.task == 'multiclass_classification':
                hyper_parameters.update(**{
                    "objective": 'multiclass',
                    "metric": 'multi_logloss',
                    # "num_class": n_classes,
                })

            hyper_parameters.update(**{"verbose": -1})

        elif self.algorithm == 'xgboost':
            if self.task == 'binary_classification':
                hyper_parameters.update(**{
                    "objective": 'binary:logistic',
                    "eval_metric": 'logloss' # auc
                })
            elif self.task == 'multiclass_classification':
                hyper_parameters.update(**{
                    "objective": 'multi:softmax', # multi:softprob
                    "eval_metric": 'mlogloss',
                    # "num_class": n_classes,
                })
            
            hyper_parameters.update(**{
                "use_rmm": True,
                "device": 'cpu', # cpu, cuda, gpu
                "nthread": -1,
                "max_delta_step": 0,
                "gamma": 0,
                "subsample": 1, # hp.uniform('xgboost.subsample', 0.6, 1)
                "sampling_method": 'uniform'
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
            raise NotImplementedError(f'Algorithm "{self.algorithm}" has not been implemented yet.\n')
        
        self.fitted = False
        
        if debug:
            LOGGER.debug('self.model: %s', self.model)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> None:
        """
        Method to fit self.model.

        :param `y`: (pd.DataFrame) Binary & balanced train target.
        :param `X`: (pd.DataFrame) Train features.
        """
        # Build self.model, if required
        if self.model is None:
            self.build()
        
        # Fit self.model
        self.model.fit(
            X.values, 
            y.values.ravel()
        )
        
        # Update Version
        if not self.fitted:
            self.fitted = True
            self.version = 1
        else:
            self.version += 1

    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Method for realizing new probabilistic inferences.

        :param `X`: (pd.DataFrame) New features to make inferences on.

        :return: (np.ndarray) New probabilistic inferences.
        """
        y_score: np.ndarray = self.model.predict_proba(X.values)[:, 1]

        if (
            self.algorithm == 'xgboost' 
            and 'objective' in self.hyper_parameters.keys() 
            and self.hyper_parameters['objective'] == 'binary:logitraw'
        ):
            # Transform logits into probabilities
            return expit(y_score)
        return y_score
    
    def interpret_score(
        self,
        y_score: np.ndarray,
        cutoff: float = None
    ) -> np.ndarray:
        # Set up cutoff
        if cutoff is None:
            cutoff = self.cutoff

        # Define class based on the self.cutoff
        y_pred = np.where(y_score > cutoff, 1, 0)

        return y_pred
    
    def evaluate_val(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        splits: int,
        debug: bool = False
    ) -> None:
        """
        Method that will define a score metric (based on the eval_metric parameter) and will leverage
        the cross validation technique to obtain the validation scores.

        :param `X_train`: (pd.DataFrame) Train features.
        :param `y_train`: (pd.DataFrame) binary & balanced train target.
        :param `splits`: (int) Number of splits to perform in the cross validation.
        :param `debug`: (bool) Wether or not to show self.cv_scores, for debugging purposes.
        """
        # # Define scorer
        # if self.optimization_metric == 'f1_score':
        #     scorer = make_scorer(f1_score)
        # elif self.optimization_metric == 'f1_weighted':
        #     scorer = 'f1_weighted'
        # elif self.optimization_metric == 'precision':
        #     scorer = make_scorer(precision_score)
        # elif self.optimization_metric == 'recall':
        #     scorer = make_scorer(recall_score)
        # elif self.optimization_metric == 'roc_auc':
        #     scorer = make_scorer(roc_auc_score)
        # elif self.optimization_metric == 'accuracy':
        #     scorer = make_scorer(accuracy_score)
        # else:
        #     raise Exception(f'Invalid "self.optimization_metric": {self.optimization_metric}.\n\n')

        # Evaluate Model using Cross Validation
        self.cv_scores = cross_val_score(
            estimator=self.model, 
            X=X_train.values,
            y=y_train.values.ravel(),
            cv=splits, 
            scoring=self.optimization_metric,
            n_jobs=-1
        )
        
        if debug:
            LOGGER.debug('self.cv_scores: %s\n', self.cv_scores)

    def evaluate_test(
        self,
        y_pred: np.ndarray,
        y_test: pd.DataFrame,
        cutoff: float = None,
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
        # Set up cutoff
        if cutoff is None:
            cutoff = self.cutoff

        # Define y_score
        y_score: np.ndarray = y_pred.copy()

        # Interpret probabilities
        y_pred: np.ndarray = self.interpret_score(
            y_score=y_score,
            cutoff=cutoff
        )

        # Prepare y_test
        y_test = y_test.values

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
        if self.optimization_metric == 'f1':
            self.test_score = self.f1_score
        elif self.optimization_metric == 'precision':
            self.test_score = self.precision_score
        elif self.optimization_metric == 'recall':
            self.test_score = self.recall_score
        elif self.optimization_metric == 'roc_auc':
            self.test_score = self.roc_auc_score
        elif self.optimization_metric == 'accuracy':
            self.test_score = self.accuracy_score
        else:
            raise NotImplementedError(f'Optimization metric "{self.optimization_metric}" has not yet been implemented.\n')

        if debug:
            LOGGER.debug('self.test_score (%s): %s', self.optimization_metric, self.test_score)

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

        LOGGER.info(
            'Optimal cutoff for %s: %s (F1 Score %s).\n', 
            self.model_id, round(self.cutoff, 3), optimal_f1_score
        )

    def diagnose(self) -> dict:
        return {
            'needs_repair': False
        }