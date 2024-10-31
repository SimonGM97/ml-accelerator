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
        mock_datasets: bool = False
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
        datasets: List[pd.DataFrame]
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
        mock_datasets: bool = False
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
        mock_datasets: bool = False
    ) -> pd.DataFrame:
        # Run extract method
        datasets: List[str] = self.extract(
            pred_id=pred_id,
            persist=persist_datasets,
            write_mode=write_mode,
            mock_datasets=mock_datasets
        )

        # Run transform method
        df = self.transform(datasets=datasets)

        # Run load method
        df = self.load(
            df=df,
            persist=persist_datasets,
            write_mode=write_mode,
            mock_datasets=mock_datasets
        )

        return df