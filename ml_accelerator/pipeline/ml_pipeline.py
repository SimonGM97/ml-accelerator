from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.data_cleaning import DataCleaner
from ml_accelerator.data_processing.data_transforming import DataTransformer
from ml_accelerator.modeling.classification_model import ClassificationModel
from ml_accelerator.modeling.regression_model import RegressionModel
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

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
        target: str,
        task: str,
        DC: DataCleaner,
        DT: DataTransformer,
        model: ClassificationModel | RegressionModel
    ) -> None:
        # General attributes
        self.target: str = target
        self.task: str = task

        # Data Processing attributes
        self.DC: DataCleaner = DC
        self.DT: DataTransformer = DT

        # Modeling attributes
        self.model: ClassificationModel | RegressionModel = model

    def divide_datasets(
        self,
        df: pd.DataFrame,
        test_size: float,
        balance_train: bool = False,
        balance_method: str = None,
        debug: bool = False
    ) -> Tuple[
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame
    ]:
        # Divide df into X & y
        X, y = df.drop(columns=[self.target]), df[self.target]

        # Delete df from memory
        del df

        # Divide into X_train, y_train, X_test, y_test
        if self.task == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=23111997,
                stratify=y
            )
        elif self.task == 'regression':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=23111997
            )
        elif self.task == 'forecasting':
            train_periods: int = int(test_size * X.shape[0])

            X_train: pd.DataFrame = X.iloc[:train_periods]
            X_test: pd.DataFrame = X.iloc[train_periods:]
            y_train: pd.DataFrame = y.iloc[:train_periods]
            y_test: pd.DataFrame = y.iloc[train_periods:]
        else:
            raise Exception(f'Invalid self.task parameter: "{self.task}".\n')
        
        # Balance Train Dataset
        if self.task == 'classification' and balance_train:
            if balance_method == 'RandomOverSampler':
                # Utilize over-sampling methodology
                RO = RandomOverSampler(random_state=0)
                X_train, y_train = RO.fit_resample(X_train, y_train)
            elif balance_method == 'RandomUnderSampler':
                # Utilize under-sampling methodology
                RU = RandomUnderSampler(return_indices=False, random_state=0)
                X_train, y_train = RU.fit_resample(X_train, y_train)
            elif balance_method == 'SMOTE':
                # Utilize Synthetic minority over-sampling technique (SMOTE) methodology
                smote = SMOTE(sampling_strategy='minority', random_state=0, n_jobs=-1)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            else:
                raise Exception(f'Invalid "balance_method" parameter was chosen: {balance_method}.\n')
        else:
            LOGGER.warning('balance_train is False, therefore test datasets will not be balanced.')

        if debug:
            LOGGER.debug("train balance: \n%s\n"
                         "test balance: \n%s\n",
                         y_train.groupby(self.target)[self.target].count() / y_train.shape[0],
                         y_test.groupby(self.target)[self.target].count() / y_test.shape[0])
        
        # Delte X & y from memory
        del X
        del y
        
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> None:
        # Run DC.fit_transform method
        X_train, y_train = self.DC.fit_transform(X=X_train, y=y_train)
        
        # Run DT.fit_transform method
        X_train, y_train = self.DT.fit_tranform(X=X_train, y=y_train)

        # Fit the model
        self.model.fit(X=X_train, y=y_train)

    def predict(
        self,
        X_test: pd.DataFrame,
        cutoff: float = None
    ) -> np.ndarray:
        # Clean X
        X_test, _ = self.DC.transform(X=X_test, y=None)
        
        # Transform X
        X_test, _ = self.DT.transform(X=X_test, y=None)

        # Predict new y
        y_pred = self.model.predict(X=X_test, cutoff=cutoff)

        return y_pred

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

    def load(
        self
    ):
        pass

    def save(
        self,
        to_s3: bool = True,
        log_model: bool = False,
        register_model: bool = False
    ) -> None:
        # Save DataCleaner
        self.DC.save()

        # Save DataTransformer
        self.DT.save()

        # Save Model
        self.model.save(
            to_s3=to_s3,
            log_model=log_model,
            register_model=register_model
        )

