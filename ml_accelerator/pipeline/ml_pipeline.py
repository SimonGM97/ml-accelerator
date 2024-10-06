from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.data_cleaning import DataCleaner
from ml_accelerator.data_processing.data_transforming import DataTransformer
from ml_accelerator.modeling.models.classification_model import ClassificationModel
from ml_accelerator.modeling.models.regression_model import RegressionModel
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
import numpy as np

from typing import List, Tuple
import warnings

# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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


class MLPipeline:

    def __init__(
        self,
        DC: DataCleaner,
        DT: DataTransformer,
        model: ClassificationModel | RegressionModel = None
    ) -> None:
        # Data Processing attributes
        self.DC: DataCleaner = DC
        self.DT: DataTransformer = DT

        # Modeling attributes
        self.model: ClassificationModel | RegressionModel = model

    @timing
    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        persist_datasets: bool = False,
        overwrite: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Clean X & y
        X, y = self.DC.transform(X=X, y=y)

        # Persist datasets
        if persist_datasets:
            # Persist X_clean
            self.DC.persist_dataset(
                df=X, 
                df_name='X_clean',
                overwrite=overwrite
            )

            # Persist y_clean
            self.DC.persist_dataset(
                df=y,
                df_name='y_clean',
                overwrite=overwrite
            )
        
        # Transform X & y
        X, y = self.DT.transform(X=X, y=y)

        # Persist datasets
        if persist_datasets:
            # Persist X_trans
            self.DC.persist_dataset(
                df=X, 
                df_name='X_trans',
                overwrite=overwrite
            )

            # Persist y_trans
            self.DC.persist_dataset(
                df=y,
                df_name='y_trans',
                overwrite=overwrite
            )

        return X, y

    @timing
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        persist_datasets: bool = False,
        overwrite: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        # Run DC.fit_transform method
        X, y = self.DC.fit_transform(X=X, y=y)

        # Persist datasets
        if persist_datasets:
            # Persist X_clean
            self.DC.persist_dataset(
                df=X, 
                df_name='X_clean',
                overwrite=overwrite
            )

            # Persist y_clean
            self.DC.persist_dataset(
                df=y,
                df_name='y_clean',
                overwrite=overwrite
            )
        
        # Run DT.fit_transform method
        X, y = self.DT.fit_tranform(X=X, y=y)

        # Persist datasets
        if persist_datasets:
            # Persist X_trans
            self.DC.persist_dataset(
                df=X, 
                df_name='X_trans',
                overwrite=overwrite
            )

            # Persist y_trans
            self.DC.persist_dataset(
                df=y,
                df_name='y_trans',
                overwrite=overwrite
            )

        return X, y

    @timing
    def predict(
        self,
        X: pd.DataFrame,
        cutoff: float = None
    ) -> np.ndarray:
        # Apply data transformation pipeline
        X, _ = self.transform(X=X, y=None)

        # Predict new y
        y_pred = self.model.predict(X=X, cutoff=cutoff)

        return y_pred
    
    @timing
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> None:
        # Run fit_transform method
        X_train, y_train = self.fit_transform(X=X_train, y=y_train)

        # Fit the model
        if self.model is not None:
            self.model.fit(X=X_train, y=y_train)

    @timing
    def fit_predict(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        cutoff: float = None
    ) -> np.ndarray:
        # Fit pipeline with X_train & y_train
        self.fit(X_train=X_train, y_train=y_train)

        # Predict with y_test
        y_pred: np.ndarray = self.predict(X_test=X_test, cutoff=cutoff)

        return y_pred
    
    def save(self) -> None:
        # Save DataCleaner
        self.DC.save()

        # Save DataTransformer
        self.DT.save()

        # Save Model
        if self.model is not None:
            self.model.save()

    def load(self) -> None:
        # Load DataCleaner
        self.DC.load()

        # Load Datatransformer
        self.DT.load()
        
        # Load Model
        if self.model is not None:
            self.model.load(light=False)

