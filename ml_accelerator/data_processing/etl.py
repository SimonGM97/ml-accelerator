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
import gc
import os
import requests
from typing import List, Tuple, Dict


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


class ExtractTransformLoad(DataHelper):

    def __init__(
        self,        
        source: str = Params.ETL_SOURCE,
        target: str = Params.TARGET,
        dataset_name: str = Params.DATASET_NAME,
    ) -> None:
        # Instanciate parent class to inherit attrs & methods
        super().__init__(
            dataset_name=dataset_name
        )

        # Define ETL specific attributes
        self.source: str = source
        self.target: str = target 

    def extract(self) -> List[pd.DataFrame]:
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
            target_col = {
                'iris': 'species',
                'wine': 'wine_class',
                'breast_cancer': 'diagnosis',
                'diabetes': 'disease_progression',
                'california_housing': 'average_price'
            }.get(self.dataset_name)

            y: pd.DataFrame = pd.DataFrame(
                data=data['target'],
                columns=[target_col]
            )

            # Map target names
            if 'target_names' in data and len(data['target_names']) > 1:
                target_names: List[str] = data['target_names']
                mapping_dict = {i: target_names[i] for i in range(len(target_names))}
                y[target_col] = y[target_col].map(mapping_dict)

            # Concatenate datasets
            df: pd.DataFrame = pd.concat([y, X], axis=1)
        else:
            raise NotImplementedError(f'Source {self.source} has not yet been implemented.\n')
        
        return [df]

    def transform(
        self,
        datasets: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method designed to perform transformations required to create a tabular 
        DataFrame that can be consumed by an ML model.
        """
        # Concatenate datasets into tabular form
        df: pd.DataFrame = pd.concat(datasets, axis=1)

        # Divide df into X & y
        X, y = df.drop(columns=[self.target]), df[[self.target]]

        # Delete df from memory
        del df
        gc.collect()

        return X, y
    
    def load(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        persist: bool = False,
        overwrite: bool = True,
        mock_datasets: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load datasets into training directory
        """
        if persist:
            # Persist X
            self.persist_dataset(
                df=X, 
                df_name='X_raw',
                overwrite=overwrite,
                mock=mock_datasets
            )

            # Persist y
            self.persist_dataset(
                df=y, 
                df_name='y_raw',
                overwrite=overwrite,
                mock=mock_datasets
            )

        return X, y

    def run_pipeline(
        self,
        persist_datasets: bool = False,
        overwrite: bool = True,
        mock_datasets: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run extract method
        datasets: List[str] = self.extract()

        # Run transform method
        X, y = self.transform(datasets=datasets)

        # Run load method
        X, y = self.load(
            X=X, y=y,
            persist=persist_datasets,
            overwrite=overwrite,
            mock_datasets=mock_datasets
        )

        return X, y