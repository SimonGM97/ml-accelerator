from ml_accelerator.config.params import Params
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.filesystem.filesystem_helper import (
    load_from_filesystem,
    save_to_filesystem
)
from ml_accelerator.utils.aws.s3_helper import (
    load_from_s3,
    save_to_s3
)

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import seaborn as sns
import yaml
import os
import gc
from typing import List, Tuple, Dict, Any


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


class DataHelper:

    # Pickled attrs
    pickled_attrs = []

    def __init__(
        self,
        target: str = Params.TARGET,
        task: str = Params.TASK,
        dataset_name: str = Params.DATASET_NAME,
        bucket: str = Params.BUCKET,
        cwd: str = Params.CWD,
        storage_env: str = Params.DATA_STORAGE_ENV,
        training_path: List[str] = Params.TRAINING_PATH,
        inference_path: List[str] = Params.INFERENCE_PATH,
        transformers_path: List[str] = Params.TRANSFORMERS_PATH,
        data_extention: str = Params.DATA_EXTENTION,
        partition_cols: str = Params.PARTITION_COLUMNS
    ) -> None:
        # Set attributes
        self.target: str = target
        self.task: str = task

        self.dataset_name: str = dataset_name
        self.bucket: str = bucket
        self.cwd: str = cwd
        self.storage_env: str = storage_env

        self.training_path: List[str] = training_path
        self.inference_path: List[str] = inference_path
        self.transformers_path: List[str] = transformers_path

        self.data_extention: str = data_extention
        self.partition_cols: str = partition_cols

    def divide_datasets(
        self,
        X: pd.DataFrame,
        y: pd.Series,
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
        # Divide into X_train, y_train, X_test, y_test
        if self.task in ['binary_classification', 'multiclass_classification']:
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
        
        return X_train, X_test, y_train, y_test

    def load_schema(self) -> dict:
        try:
            # Load schema
            with open(os.path.join("schemas", f"{self.dataset_name}_schema.yaml")) as file:
                schema: dict = yaml.load(file, Loader=yaml.FullLoader)
        except Exception as e:
            LOGGER.warning(
                'Unable to load schema from %s.\n'
                'Schema will be re-created.\n'
                'Exception: %s.',
                self.dataset_name, e
            )

            # Infer schema
            schema: dict = self.infer_schema()

        return schema
    
    def infer_schema(self) -> dict:
        # Load df
        df: pd.DataFrame = pd.concat([
            self.load_dataset(df_name="y_raw", filters=None),
            self.load_dataset(df_name="X_raw", filters=None)
        ], axis=1)

        # Extract dtypes
        dtypes: pd.Series = df.dtypes
        dtypes = dtypes.apply(lambda x: str(x))

        # Define schema
        schema: dict = {
            "name": self.dataset_name,
            "path": '/'.join(self.training_path),
            "fields": [
                {
                    "name": col_name,
                    "type": dtypes[col_name],
                    "mandatory": True,
                    "nullable": True if df[col_name].isnull().sum() > 0 else False,
                    "min_value": float(df[col_name].min()) if dtypes[col_name] not in ['string', 'object'] else None,
                    "max_value": float(df[col_name].max()) if dtypes[col_name] not in ['string', 'object'] else None,
                    "allowed_values": df[col_name].unique().tolist() if dtypes[col_name] in ['string', 'object'] else None,
                    "fillna_method": 'simple_imputer'

                } for col_name in dtypes.index
            ]
        }
        
        # Save schema
        with open(os.path.join('schemas', f'{self.dataset_name}_schema.yaml'), 'w') as file:
            yaml.dump(schema, file)

        return schema

    def persist_dataset(
        self,
        df: pd.DataFrame,
        df_name: str,
        overwrite: bool = True
    ) -> None:
        if self.storage_env == 'filesystem':
            # Persist to filesystem
            save_to_filesystem(
                asset=df,
                path=os.path.join(self.bucket, *self.training_path, f"{df_name}.{self.data_extention}"),
                partition_cols=self.partition_cols,
                overwrite=overwrite
            )
        elif self.storage_env == 'S3':
            # Persist to S3
            save_to_s3(
                asset=df,
                path=f"{self.bucket}/{'/'.join(self.training_path)}/{df_name}.{self.data_extention}",
                partition_cols=self.partition_cols,
                overwrite=overwrite
            )
        else:
            raise Exception(f'Invalid self.storage_env was received: "{self.storage_env}".\n')

    def load_dataset(
        self,
        df_name: str,
        filters: List[Tuple[str, str, List[str]]] = None
    ) -> pd.DataFrame:
        if self.storage_env == 'filesystem':
            # Load from filesystem
            df: pd.DataFrame = load_from_filesystem(
                path=os.path.join(self.bucket, *self.training_path, f"{df_name}.{self.data_extention}"),
                partition_cols=self.partition_cols,
                filters=filters
            )
        elif self.storage_env == 'S3':
            # Load from S3
            df: pd.DataFrame = load_from_s3(
                path=f"{self.bucket}/{'/'.join(self.training_path)}/{df_name}.{self.data_extention}",
                partition_cols=self.partition_cols,
                filters=filters
            )
        else:
            raise Exception(f'Invalid self.storage_env was received: "{self.storage_env}".\n')
        
        return df

    def save_transformer(
        self,
        transformer_name: str
    ) -> None:
        # Define attrs to save
        attrs: dict = {key: value for (key, value) in self.__dict__.items() if key in self.pickled_attrs}

        if self.storage_env == 'filesystem':
            # Define base_path
            base_path = os.path.join(self.bucket, *self.transformers_path, transformer_name)

            # Save attrs to filesystem
            save_to_filesystem(
                asset=attrs,
                path=os.path.join(base_path, f"{transformer_name}_attrs.pickle"),
                partition_cols=None,
                overwrite=True
            )

        elif self.storage_env == 'S3':
            # Define base_path
            base_path = f"{self.bucket}/{'/'.join(self.transformers_path)}/{transformer_name}"

            # Save attrs to S3
            save_to_s3(
                asset=attrs,
                path=f"{base_path}/{transformer_name}_attrs.pickle",
                partition_cols=None,
                overwrite=True
            )

    def load_transformer(
        self,
        transformer_name: str
    ) -> None:
        if self.storage_env == 'filesystem':
            # Define base_path
            base_path = os.path.join(self.bucket, *self.transformers_path, transformer_name)

            # Load from filesystem
            attrs: dict = load_from_filesystem(
                path=os.path.join(base_path, f"{transformer_name}_attrs.pickle"),
                partition_cols=None,
                filters=None
            )
        
        elif self.storage_env == 'S3':
            # Define base_path
            base_path = f"{self.bucket}/{'/'.join(self.transformers_path)}/{transformer_name}"

            # Load from S3
            attrs: dict = load_from_s3(
                path=f"{base_path}/{transformer_name}_attrs.pickle",
                partition_cols=None,
                filters=None
            )

        # Assign pickled attrs
        for attr_name, attr_value in attrs.items():
            if attr_name in self.pickled_attrs:
                setattr(self, attr_name, attr_value)


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python ml_accelerator/data_processing/data_extractor.py
if __name__ == "__main__":
    # Instanciate DataHelper
    DH: DataHelper = DataHelper(
        bucket=Params.BUCKET,
        cwd=Params.CWD,
        storage_env=Params.DATA_STORAGE_ENV,
        training_path=Params.TRAINING_PATH,
        inference_path=Params.INFERENCE_PATH
    )

    # Load dataset
    df: pd.DataFrame = sns.load_dataset(Params.DATASET_NAME)

    # Persist dataset
    DH.persist_dataset(df=df, df_name=f"{Params.DATASET_NAME}_raw_data")