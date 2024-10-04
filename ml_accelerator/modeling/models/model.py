from ml_accelerator.config.params import Params
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.aws.s3_helper import load_from_s3, save_to_s3
from ml_accelerator.utils.filesystem.filesystem_helper import (
    load_from_filesystem,
    save_to_filesystem
)

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import mlflow
import shap
import secrets
import string
import os
from pprint import pformat
from copy import deepcopy
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


class Model(ABC):

    # Pickled attrs
    pickled_attrs = []

    # csv attrs
    csv_attrs = []

    # Parquet attrs
    parquet_attrs = []
    partition_cols = {}

    # Metrics
    metric_names = []

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
        importance_method: str = Params.IMPORTANCE_METHOD,
        storage_env: str = Params.MODEL_STORAGE_ENV,
        bucket: str = Params.BUCKET,
        models_path: List[str] = Params.MODELS_PATH
    ) -> None:
        # Register Parameters
        if model_id is not None:
            self.model_id: str = model_id
        else:
            self.model_id: str = ''.join(secrets.choice(string.ascii_letters) for i in range(10))
        
        self.version: int = version
        self.stage: str = stage

        # Storage Parameters
        self.storage_env: str = storage_env
        self.bucket: str = bucket
        self.models_path: List[str] = models_path

        # Model Parameters
        self.model = None
        self.algorithm: str = algorithm
        self.hyper_parameters: dict = deepcopy(hyper_parameters)
        self.fitted: bool = False

        # Data Parameters
        self.target: str = target
        self.selected_features: List[str] = deepcopy(selected_features)

        # Performance Parameters
        self.optimization_metric: str = optimization_metric
        self.cv_scores: np.ndarray = np.array([])
        self.test_score: float = None

        # Feature importance Parameters
        self.feature_importance_df: pd.DataFrame = pd.DataFrame(columns=['feature', 'importance'])
        self.importance_method: str = importance_method
        self.shap_values: np.ndarray = None

    """
    Properties
    """

    @property
    def warm_start_params(self) -> dict:
        """
        Defines the parameters required for a warm start on the ModelTuner.run() method.
        Can be accesses as an attribute.

        :return: (dict) warm parameters.
        """
        algorithms: List[str] = Params.ALGORITHMS

        params = {
            # General Parameters
            'algorithm': self.algorithm,

            # Register Parameters            
            'model_id': self.model_id,
            'version': self.version,
            'stage': self.stage,

            # Others
            'model_type': algorithms.index(self.algorithm)
        }

        # Hyper-Parameters
        params.update(**{
            f'{self.algorithm}.{k}': v for k, v in self.hyper_parameters.items()
        })

        return params

    @property
    def tags(self) -> dict:
        """
        Defines the tags to be saved in the mlflow tracking server.
        Can be accessed as an attribute.

        :return: (dict) Dictionary of tags.
        """
        return {
            'algorithm': self.algorithm,
            'stage': self.stage,
            'version': self.version
        }
    
    @property
    def metrics(self) -> dict:
        """
        Defines the test and validation metrics to be logged in the mlflow tracking server.
        Can be accessed as an attribute.

        :return: (dict) validation & test metrics.
        """

        return {
            metric_name: getattr(metric_name) for metric_name in self.metric_names
        }

    @property
    def val_score(self) -> float:
        """
        Defines the validation score as the mean value of the cross validation results.
        Can be accessed as an attribute.

        :return: (float) mean cross validation score.
        """
        if self.cv_scores is not None:
            # return (np.abs(self.cv_scores - self.cv_scores.mean()) / self.cv_scores.mean()).mean()
            return self.cv_scores.mean()
        return None

    """
    Abstract methods
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> None:
        pass
    
    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_val(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        splits: int,
        debug: bool = False
    ) -> None:
        pass

    @abstractmethod
    def evaluate_test(self):
        pass

    @abstractmethod
    def diagnose(self) -> dict:
        pass

    """
    Feature Importance
    """

    def find_feature_importance(
        self,
        X_test: pd.DataFrame,
        find_new_shap_values: bool = False,
        debug: bool = False
    ) -> None:
        """
        Method that utilizes the shap library to calculate feature impotances on the test dataset 
        (whenever possible).

        :param `test_features`: (pd.DataFrame) Test features.
        :param `find_new_shap_values`: (bool) Wether or not to calculate new shaply values.
        :param `debug`: (bool) Wether or not to show top feature importances, for debugging purposes.
        """
        def find_shap_feature_importance() -> pd.DataFrame:
            if find_new_shap_values or self.shap_values is None:
                LOGGER.info('Calculating new shaply values for %s.', self.model_id)

                # Fits the explainer
                # explainer = shap.Explainer(self.model.predict_proba, X_test)

                # Calculates the SHAP values
                # self.shap_values: np.ndarray = explainer(X_test)

                # Instanciate explainer
                explainer = shap.TreeExplainer(self.model)

                # Calculate shap values
                self.shap_values = explainer.shap_values(X_test)

            # Find the sum of feature values
            shap_sum = np.abs(self.shap_values).mean(0).sum(0)

            # Find shap feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': X_test.columns.tolist(),
                'importance': shap_sum
            })

            self.importance_method = 'shap'

            return importance_df
        
        def find_native_feature_importance() -> pd.DataFrame:
            # Define DataFrame to describe importances on (utilizing native feature importance calculation method)
            importance_df = pd.DataFrame({
                'feature': X_test.columns.tolist(),
                'importance': self.model.feature_importances_.tolist()
            })

            self.shap_values = None
            self.importance_method = f'native_{self.algorithm}'

            return importance_df

        if self.importance_method == 'shap':
            try:
                importance_df: pd.DataFrame = find_shap_feature_importance()
            except Exception as e:
                LOGGER.warning(
                    "Unable to calculate shap feature importance on %s (%s).\n"
                    "Exception: %s\n"
                    "Re-trying with native approach.\n",
                    self.model_id, self.algorithm, e
                )
            
                importance_df: pd.DataFrame = find_native_feature_importance()
        else:
            importance_df: pd.DataFrame = find_native_feature_importance()

        # Sort DataFrame by shap_value
        importance_df.sort_values(by=['importance'], ascending=False, ignore_index=True, inplace=True)

        # Find shap cumulative percentage importance
        importance_df['cum_perc'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

        # Assign result to the self.feature_importance_df attribute
        self.feature_importance_df = importance_df
        
        if debug:
            LOGGER.debug('Shap importance df (top 20): \n%s\n', importance_df.iloc[:20].to_string())

    """
    Save Methods
    """

    def save(self) -> None:
        if self.storage_env == 'filesystem':
            # Save to filesystem
            self.save_to_filesystem()

        elif self.storage_env == 'S3':
            # Save to S3
            self.save_to_s3()
        
        elif self.storage_env == 'ml_flow':
            # Register model
            self.register_model()

    def save_to_filesystem(self) -> None:
        # Define model_attrs
        model_attrs: dict = {
            key: value for (key, value) in self.__dict__.items() if key in self.pickled_attrs
        }

        # Define base_path
        base_path = os.path.join(self.bucket, "models", self.model_id)

        # Save self.model
        if self.model is not None:
            save_to_filesystem(
                asset=self.model,
                path=os.path.join(base_path, f"{self.model_id}_model.pickle"),
                partition_column=None,
                overwrite=True
            )
        
        # Save model_attrs
        save_to_filesystem(
            asset=model_attrs,
            path=os.path.join(base_path, f"{self.model_id}_model_attrs.pickle"),
            partition_column=None,
            overwrite=True
        )

        # Save csv attrs
        for attr_name in self.csv_attrs:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                save_to_filesystem(
                    asset=df,
                    path=os.path.join(base_path, f"{self.model_id}_{attr_name}.csv"),
                    partition_column=None,
                    overwrite=True
                )

        # Save parquet attrs
        for attr_name in self.parquet_attrs:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                save_to_s3(
                    asset=df,
                    path=os.path.join(base_path, f"{self.model_id}_{attr_name}.parquet"),
                    partition_column=self.partition_cols.get("attr_name"),
                    overwrite=True
                )

    def save_to_s3(self) -> None:
        # Define model_attrs
        model_attrs: dict = {key: value for (key, value) in self.__dict__.items() if key in self.pickled_attrs}

        # Define base_path
        base_path = f"{self.bucket}/models/{self.model_id}"

        # Save self.model
        if self.model is not None:
            save_to_s3(
                asset=self.model,
                path=f"{base_path}/{self.model_id}_model.pickle",
                partition_column=None,
                overwrite=True
            )

        # Save model_attrs
        save_to_s3(
            asset=model_attrs,
            path=f"{base_path}/{self.model_id}_model_attrs.pickle",
            partition_column=None,
            overwrite=True
        )

        # Save csv attrs
        for attr_name in self.csv_attrs:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                save_to_s3(
                    asset=df,
                    path=f"{base_path}/{self.model_id}_{attr_name}.csv",
                    partition_column=None,
                    overwrite=True
                )

        # Save parquet attrs
        for attr_name in self.parquet_attrs:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                save_to_s3(
                    asset=df,
                    path=f"{base_path}/{self.model_id}_{attr_name}.parquet",
                    partition_column=self.partition_cols.get("attr_name"),
                    overwrite=True
                )

    
    """
    Load Methods
    """

    def load(self) -> None:
        if self.storage_env == 'filesystem':
            # Load from filesystem
            self.load_from_filesystem()            
        
        elif self.storage_env == 'S3':
            # Load from S3
            self.load_from_s3()
        
        elif self.storage_env == 'mlflow':
            # Load registered model from ML Flow
            self.load_registered_model()

    def load_from_filesystem(self) -> None:
        # Define base_path
        base_path = os.path.join(self.bucket, *self.models_path, self.model_id)

        # Load self.model
        self.model = load_from_filesystem(
            path=os.path.join(base_path, f"{self.model_id}_model.pickle"),
            partition_cols=None,
            filters=None
        )
        
        # Load model_attrs
        model_attrs: dict = load_from_filesystem(
            path=os.path.join(base_path, f"{self.model_id}_model_attrs.pickle"),
            partition_cols=None,
            filters=None
        )

        # Assign pickled attrs
        for attr_name, attr_value in model_attrs.items():
            if attr_name in self.pickled_attrs:
                setattr(self, attr_name, attr_value)

        # Load csv attrs
        for attr_name in self.csv_attrs:
            # Load attribute
            df: pd.DataFrame = load_from_filesystem(
                path=os.path.join(base_path, f"{self.model_id}_{attr_name}.csv"),
                partition_cols=None,
                filters=None
            )

            # Assign attribute
            setattr(self, attr_name, df)

        # Load parquet attrs
        for attr_name in self.parquet_attrs:
            # Load attribute
            df: pd.DataFrame = load_from_filesystem(
                path=os.path.join(base_path, f"{self.model_id}_{attr_name}.parquet"),
                partition_column=self.partition_cols.get("attr_name"),
                filters=None
            )

            # Assign attribute
            setattr(self, attr_name, df)

    def load_from_s3(self) -> None:
        # Define base_path
        base_path = f"{self.bucket}/{'/'.join(self.models_path)}/{self.model_id}"

        # Load self.model
        self.model = load_from_s3(
            path=f"{base_path}/{self.model_id}_model.pickle",
            partition_cols=None,
            filters=None
        )

        # Load pickled attrs
        model_attrs: dict = load_from_s3(
            path=f"{base_path}/{self.model_id}_model_attrs.pickle",
            partition_cols=None,
            filters=None
        )

        # Assign pickled attrs
        for attr_name, attr_value in model_attrs.items():
            if attr_name in self.pickled_attrs:
                setattr(self, attr_name, attr_value)

        # Load csv files
        for attr_name in self.csv_attrs:
            # Load attribute
            df: pd.DataFrame = load_from_s3(
                path=f"{base_path}/{self.model_id}_{attr_name}.csv",
                partition_cols=None,
                filters=None
            )

            # Assign attribute
            setattr(self, attr_name, df)

        # Load parquet attrs
        for attr_name in self.parquet_attrs:
            # Load attribute
            df: pd.DataFrame = load_from_s3(
                path=f"{base_path}/{self.model_id}_{attr_name}.parquet",
                partition_column=self.partition_cols.get("attr_name"),
                filters=None
            )

            # Assign attribute
            setattr(self, attr_name, df)

    """
    MLFlow Methods
    """

    def register_model(self):
        pass

    def log_model(self):
        # Save Model to tracking system
        pass

    def load_registered_model(self) -> None:
        pass
    
    """
    Other methods
    """

    def __repr__(self) -> str:
        # Define register attributes
        reg_attrs = {
            'Model ID': self.model_id,
            'Version': self.version,
            'Stage': self.stage
        }

        # ML Attributes
        n_features = 0 if self.selected_features is None else len(self.selected_features)
        ml_attrs = {
            'Algorithm': self.algorithm,
            'Hyper Parameters': self.hyper_parameters,
            'Selected Features (len)': n_features
        }

        # Prepare output
        output = "Model:\n"
        output += f"Register Attributes:\n{pformat(reg_attrs)}\n\n"
        output += f"ML Attributes:\n{pformat(ml_attrs)}\n\n"

        return output
        

        


