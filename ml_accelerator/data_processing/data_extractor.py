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
import seaborn as sns
import yaml
import os

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


class DataExtractor:

    def __init__(
        self,
        bucket: str = Params.BUCKET,
        cwd: str = Params.CWD,
        storage_env: str = Params.DATA_STORAGE_ENV,
        training_path: List[str] = Params.TRAINING_PATH,
        inference_path: List[str] = Params.INFERENCE_PATH,
        data_extention: str = Params.DATA_EXTENTION,
        partition_column: str = Params.PARTITION_COLUMN
    ) -> None:
        # Set attributes
        self.bucket: str = bucket
        self.cwd: str = cwd
        self.storage_env: str = storage_env
        self.training_path: List[str] = training_path
        self.inference_path: List[str] = inference_path
        self.data_extention: str = data_extention
        self.partition_column: str = partition_column

    def load_schema(
        self,
        schema_name: str
    ) -> dict:
        # Load schema
        with open(os.path.join("schemas", f"{schema_name}.yaml")) as file:
            schema: dict = yaml.load(file, Loader=yaml.FullLoader)

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
                partition_column=self.partition_column,
                overwrite=overwrite
            )
        elif self.storage_env == 'S3':
            # Persist to S3
            save_to_s3(
                asset=df,
                path=f"{self.bucket}/{'/'.join(self.training_path)}/{df_name}.{self.data_extention}",
                partition_column=self.partition_column,
                overwrite=overwrite
            )
        else:
            raise Exception(f'Invalid self.storage_env was received: "{self.storage_env}".\n')

    def load_dataset(
        self,
        df_name: str,
        partition_cols: List[str] = None,
        filters: List[Tuple[str, str, List[str]]] = None
    ) -> pd.DataFrame:
        if self.storage_env == 'filesystem':
            # Load from filesystem
            df: pd.DataFrame = load_from_filesystem(
                path=os.path.join(self.bucket, *self.training_path, f"{df_name}.{self.data_extention}"),
                partition_cols=partition_cols,
                filters=filters
            )
        elif self.storage_env == 'S3':
            # Load from S3
            df: pd.DataFrame = load_from_s3(
                path=f"{self.bucket}/{'/'.join(self.training_path)}/{df_name}.{self.data_extention}",
                partition_cols=partition_cols,
                filters=filters
            )
        else:
            raise Exception(f'Invalid self.storage_env was received: "{self.storage_env}".\n')
        
        return df


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python ml_accelerator/data_processing/data_extractor.py
if __name__ == "__main__":
    # Instanciate DataExtractor
    DE: DataExtractor = DataExtractor(
        bucket=Params.BUCKET,
        cwd=Params.CWD,
        storage_env=Params.DATA_STORAGE_ENV,
        training_path=Params.TRAINING_PATH,
        inference_path=Params.INFERENCE_PATH
    )

    # Load dataset
    df: pd.DataFrame = sns.load_dataset(Params.DATASET_NAME)

    # Persist dataset
    DE.persist_dataset(df=df, df_name=f"{Params.DATASET_NAME}_raw_data")