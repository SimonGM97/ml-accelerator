from ml_accelerator.config.env import Env
from ml_accelerator.config.params import Params
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import gc
import requests
from typing import List, Tuple, Dict


# Get logger
LOGGER = get_logger(name=__name__)


class ExtractTransformLoad(DataHelper):

    def __init__(
        self,        
        source: str = Params.ETL_SOURCE,
        target_column: str = Params.TARGET_COLUMN,
        dataset_name: str = Params.DATASET_NAME,
    ) -> None:
        # Instanciate parent class to inherit attrs & methods
        super().__init__(
            dataset_name=dataset_name
        )

        # Define ETL specific attributes
        self.source: str = source
        self.target_column: str = target_column 

    def extract(
        self,
        pred_id = None,
        persist: bool = False,
        write_mode: str = None,
        mock_datasets: bool = False,
        debug: bool = False
    ) -> List[pd.DataFrame]:
        """
        Extract input datasets from various sources
        """
        if self.source == 'sklearn':
            # Extract data
            if self.dataset_name == 'iris':
                data = load_iris() # Multiclass classification
            elif self.dataset_name == 'wine':
                data = load_wine()
            elif self.dataset_name == 'breast_cancer':
                data = load_breast_cancer() # Binary classification
            elif self.dataset_name == 'diabetes':
                data = load_diabetes() # Regression
            elif self.dataset_name == 'california_housing':
                data = fetch_california_housing() # Regression
            else:
                raise NotImplementedError(f'Dataset "{self.dataset_name}" has not yet been implemented.\n')
            
            # Extract X
            X: pd.DataFrame = pd.DataFrame(
                data=data['data'],
                columns=data['feature_names']
            )
            
            # Extract y
            # target_col = {
            #     'iris': 'species',
            #     'wine': 'wine_class',
            #     'breast_cancer': 'diagnosis',
            #     'diabetes': 'disease_progression',
            #     'california_housing': 'average_price'
            # }.get(self.dataset_name)

            y: pd.DataFrame = pd.DataFrame(
                data=data['target'],
                columns=[self.target_column]
            )

            # Map target names
            if 'target_names' in data and len(data['target_names']) > 1:
                target_names: List[str] = data['target_names']
                mapping_dict = {i: target_names[i] for i in range(len(target_names))}
                y[self.target_column] = y[self.target_column].map(mapping_dict)

            # Concatenate datasets
            df: pd.DataFrame = pd.concat([y, X], axis=1)

            # Delete df from memory
            del X
            del y
            gc.collect()
        else:
            raise NotImplementedError(f'Source {self.source} has not yet been implemented.\n')
        
        # Extract pred_id
        if pred_id is not None:
            df: pd.DataFrame = df.loc[pred_id:pred_id]

        # Persist datasets
        if persist:
            self.persist_dataset(
                df=df,
                df_name='df_raw',
                write_mode=write_mode,
                mock=mock_datasets
            )

        return [df]

    def transform(
        self,
        datasets: List[pd.DataFrame],
        debug: bool = False
    ) -> pd.DataFrame:
        """
        Method designed to perform transformations required to create a tabular 
        DataFrame that can be consumed by an ML model.
        """
        # Concatenate datasets into tabular form
        df: pd.DataFrame = pd.concat(datasets, axis=1)

        return df
    
    def load(
        self,
        df: pd.DataFrame,
        persist: bool = False,
        write_mode: str = None,
        mock_datasets: bool = False,
        debug: bool = False
    ) -> pd.DataFrame:
        """
        Load datasets into training directory
        """
        # Persist datasets
        if persist:
            # Persist df
            self.persist_dataset(
                df=df, 
                df_name='df_raw',
                write_mode=write_mode,
                mock=mock_datasets
            )

        return df

    def run_pipeline(
        self,
        pred_id = None,
        persist_datasets: bool = False,
        write_mode: str = None,
        mock_datasets: bool = False,
        debug: bool = False
    ) -> pd.DataFrame:
        # Run extract method
        datasets: List[str] = self.extract(
            pred_id=pred_id,
            persist=persist_datasets,
            write_mode=write_mode,
            mock_datasets=mock_datasets,
            debug=debug
        )

        # Run transform method
        df = self.transform(
            datasets=datasets,
            debug=debug
        )

        # Run load method
        df = self.load(
            df=df,
            persist=persist_datasets,
            write_mode=write_mode,
            mock_datasets=mock_datasets,
            debug=debug
        )

        return df
    
    def divide_datasets(
        self,
        df: pd.DataFrame,
        test_size: float = 0,
        balance_train: bool = False,
        balance_method: str = None,
        persist_datasets: bool = False,
        write_mode: str = None,
        mock_datasets: bool = False,
        debug: bool = False
    ) -> Tuple[
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame
    ]:
        # Divide df into X & y
        X, y = df.drop(columns=[self.target_column]), df[[self.target_column]]
        
        # Delete df_raw from memory
        del df
        gc.collect()

        # Divide into X_train, X_test, y_train, y_test
        if test_size > 0:
            if self.task in ['binary_classification', 'multiclass_classification']:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=int(Env.get("SEED")),
                    stratify=y
                )
            elif self.task == 'regression':
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=int(Env.get("SEED"))
                )
            elif self.task == 'forecasting':
                train_periods: int = int(test_size * X.shape[0])

                X_train: pd.DataFrame = X.iloc[:train_periods]
                X_test: pd.DataFrame = X.iloc[train_periods:]
                y_train: pd.DataFrame = y.iloc[:train_periods]
                y_test: pd.DataFrame = y.iloc[train_periods:]
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented.')
        else:
            X_train, X_test, y_train, y_test = X, None, y, None
        
        # Delete X & y
        del X
        del y
        gc.collect()
        
        # Balance Train Dataset
        if self.task in ['binary_classification', 'multiclass_classification'] and balance_train:
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
                smote = SMOTE(sampling_strategy='minority', random_state=0) # , n_jobs=-1)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            else:
                raise ValueError(f'Invalid "balance_method" parameter was chosen: {balance_method}.\n')
        else:
            LOGGER.warning('balance_train is False, therefore test datasets will not be balanced.')

        if debug and 'classification' in self.task:
            LOGGER.debug(
                "train balance: \n%s\n\n"
                "test balance: \n%s\n",
                y_train.groupby(self.target_column)[self.target_column].count() / y_train.shape[0],
                y_test.groupby(self.target_column)[self.target_column].count() / y_test.shape[0] if y_test is not None else None
            )

        # Persist datasets
        if persist_datasets:
            # Persist X_train
            self.persist_dataset(
                df=X_train, 
                df_name='X_train_raw',
                write_mode=write_mode,
                mock=mock_datasets
            )

            # Persist X_test
            self.persist_dataset(
                df=X_test, 
                df_name='X_test_raw',
                write_mode=write_mode,
                mock=mock_datasets
            )

            # Persist y_train
            self.persist_dataset(
                df=y_train, 
                df_name='y_train_raw',
                write_mode=write_mode,
                mock=mock_datasets
            )

            # Persist y_test
            self.persist_dataset(
                df=y_test, 
                df_name='y_test_raw',
                write_mode=write_mode,
                mock=mock_datasets
            )
        
        return X_train, X_test, y_train, y_test
