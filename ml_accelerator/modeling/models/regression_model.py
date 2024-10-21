from ml_accelerator.config.params import Params
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from typing import List


# Get logger
LOGGER = get_logger(name=__name__)


class RegressionModel(Model):
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

        # Regression Parameters
        'mape',

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
        'mape'
    ]

    def __init__(
        self,
        model_id: str = None,
        version: int = 1,
        stage: str = 'development',
        algorithm: str = None,
        hyper_parameters: dict = {},
        target: str = Params.TARGET,
        selected_features: List[str] = None,
        optimization_metric: str = Params.OPTIMIZATION_METRIC,
        importance_method: str = Params.IMPORTANCE_METHOD
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
            optimization_metric=optimization_metric,
            importance_method=importance_method
        )

        # Correct self.hyperparameters
        self.hyper_parameters: dict = self.correct_hyper_parameters(
            hyper_parameters=hyper_parameters,
            debug=False
        )

        # Regression parameters
        self.mape: float = None

    def correct_hyper_parameters(
        self,
        hyper_parameters: dict,
        debug: bool = False
    ):
        pass