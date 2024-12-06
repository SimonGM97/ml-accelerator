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
from ml_accelerator.config.env import Env

import pandas as pd
import os
import gc
from typing import List, Tuple


# Get logger
LOGGER = get_logger(name=__name__)


class DataHelper:

    # Pickled attrs
    pickled_attrs = []

    def __init__(
        self,
        target_column: str = Params.TARGET_COLUMN,
        task: str = Params.TASK,
        dataset_name: str = Params.DATASET_NAME,
        data_extention: str = Params.DATA_EXTENTION,
        partition_cols: str = Params.PARTITION_COLUMNS
    ) -> None:
        # Set attributes
        self.target_column: str = target_column
        self.task: str = task
        self.dataset_name: str = dataset_name
        self.data_extention: str = data_extention
        self.partition_cols: str = partition_cols

        # Environment parameters
        self.bucket: str = Env.get("BUCKET_NAME")
        self.storage_env: str = Env.get("DATA_STORAGE_ENV")

        self.raw_datasets_path: str = Env.get("RAW_DATASETS_PATH")
        self.processing_datasets_path: str = Env.get("PROCESSING_DATASETS_PATH")
        self.inference_path: str = Env.get("INFERENCE_PATH")
        self.transformers_path: str = Env.get("TRANSFORMERS_PATH")
        self.schemas_path: str = Env.get("SCHEMAS_PATH")

    def load_schema(self) -> dict:
        # Extract schema path
        path: str = self.find_path(f"{self.dataset_name}_schema")

        # Load schema
        try:
            if self.storage_env == 'filesystem':
                # Load from filesystem
                schema: dict = load_from_filesystem(
                    path=path,
                    partition_cols=None,
                    filters=None
                )
            elif self.storage_env == 'S3':
                # Load from S3
                schema: dict = load_from_s3(
                    path=path,
                    partition_cols=None,
                    filters=None
                )
            else:
                raise Exception(f'Invalid self.storage_env was received: "{self.storage_env}".\n')
        except Exception as e:
            LOGGER.warning(
                'Unable to load %s schema from %s.\n'
                'Exception: %s',
                self.dataset_name, self.storage_env, e
            )
            schema: dict = None
            
        # Verify it is not an empty schema
        if schema is None:
            LOGGER.warning('%s schema is None, thus it will be re-created.', self.dataset_name)

            # Infer schema
            schema: dict = self.infer_schema()

        return schema
    
    def save_schema(
        self,
        schema: dict
    ) -> None:
        # Extract schema path
        path: str = self.find_path(f"{self.dataset_name}_schema")

        # Load schema
        if self.storage_env == 'filesystem':
            # Save to filesystem
            save_to_filesystem(
                asset=schema,
                path=path,
                partition_cols=None,
                write_mode=None
            )
        elif self.storage_env == 'S3':
            # Save to S3
            save_to_s3(
                asset=schema,
                path=path,
                partition_cols=None,
                write_mode=None
            )
        else:
            raise Exception(f'Invalid self.storage_env was received: "{self.storage_env}".\n')
    
    def infer_schema(self) -> dict:
        def extract_min_value(col_name: str):
            if dtypes[col_name] in ['string', 'object']:
                return None
            else:
                return float(df[col_name].min())
            
        def extract_max_value(col_name: str):
            if dtypes[col_name] in ['string', 'object']:
                return None
            else:
                return float(df[col_name].max()) 

        def extract_allowed_values(col_name: str) -> list:
            # Extract seen values
            if dtypes[col_name] in ['string', 'object']:
                seen_values: list = df[col_name].unique().tolist()
            else:
                seen_values: list = None
            
            # Add allowed classification values
            if 'classification' in self.task and col_name == self.target_column:
                seen_values.extend(list(range(len(seen_values))))

            return seen_values
        
        LOGGER.info('Infering new schema for %s', self.dataset_name)

        # Load df
        df: pd.DataFrame = self.load_dataset(df_name="df_raw", filters=None)

        assert df is not None, "Loaded df_raw is None."

        # Extract dtypes
        dtypes: pd.Series = df.dtypes
        dtypes = dtypes.apply(lambda x: str(x))

        # Define schema
        schema: dict = {
            "name": self.dataset_name,
            "path": self.processing_datasets_path,
            "length": df.shape[0],
            "fields": [
                {
                    "name": col_name.replace(' ', '_'),
                    "type": dtypes[col_name],
                    "mandatory": True,
                    "nullable": True if df[col_name].isnull().sum() > 0 else False,
                    "inf_allowed": False,
                    "min_value": extract_min_value(col_name=col_name),
                    "max_value": extract_max_value(col_name=col_name),
                    "allowed_values": extract_allowed_values(col_name=col_name),
                    "fillna_method": 'simple_imputer'
                } for col_name in dtypes.index
            ]
        }

        # Save schema
        self.save_schema(schema=schema)

        return schema

    def load_inference_df(
        self,
        pipeline_id: str = None
    ) -> pd.DataFrame:
        # Define empty inference_df
        columns = ['pipeline_id', 'pred_id', 'prediction', 'year', 'month', 'day']
        inference_df: pd.DataFrame = pd.DataFrame(columns=columns)

        if self.storage_env == 'filesystem':
            # Define search dir
            search_dir = os.path.join(self.bucket, *self.inference_path.split('/'))

            for root, directories, files in os.walk(search_dir):
                for file in files:
                    if file != '.DS_Store':
                        # Load inference
                        inference_path = os.path.join(*root.split('/'), *'/'.join(directories), file)
                        inference: dict = load_from_filesystem(path=inference_path)

                        # Filter keys
                        inference: dict = {k: v for k, v in inference.items() if k in columns}

                        # Append to inference_df
                        inference_df: pd.DataFrame = pd.concat([
                            inference_df, pd.DataFrame(inference, index=list(range(len(inference['prediction']))))
                        ], axis=0)
        else:
            raise NotImplementedError(f'Storage environment "{self.storage_env}" has not been implemented yet.')
        
        # Filter inference_df
        if pipeline_id is not None:
            inference_df = inference_df.loc[inference_df['pipeline_id'] == pipeline_id]
        
        return inference_df

    def find_path(
        self,
        asset_name: str,
        mock: bool = False
    ) -> str:
        # Define base_path
        if self.storage_env == 'filesystem':
            if mock:
                base_path: str = os.path.join(self.bucket, "mock")
            else:
                base_path: str = self.bucket
        elif self.storage_env == 'S3':
            if mock:
                base_path: str = f"{self.bucket}/mock"
            else:
                base_path: str = self.bucket
        else:
            raise ValueError(f'Invalid self.storage_env was received: {self.storage_env}.')

        # Define path
        if self.storage_env == 'filesystem':
            if 'raw' in asset_name:
                path: str = os.path.join(base_path, *self.raw_datasets_path.split('/'), f"{asset_name}.{self.data_extention}")
            elif 'schema' in asset_name:
                path: str = os.path.join(base_path, *self.schemas_path.split('/'), f"{asset_name}.yaml")
            elif 'inference' in asset_name:
                path: str = os.path.join(base_path, *self.inference_path.split('/'), f"{asset_name}.json")
            elif 'transformer' in asset_name:
                transformer_name: str = self.__class__.__name__
                transformer_id: str = getattr(self, 'transformer_id')

                if transformer_id is None:
                    path: str = os.path.join(base_path, *self.transformers_path.split('/'), transformer_name, f"{asset_name}_attrs.pickle")
                else:
                    path: str = os.path.join(base_path, *self.transformers_path.split('/'), transformer_name, transformer_id, f"{asset_name}_attrs.pickle")
            else:
                path: str = os.path.join(base_path, *self.processing_datasets_path.split('/'), f"{asset_name}.{self.data_extention}")
                    
        elif self.storage_env == 'S3':
            if 'raw' in asset_name:
                path: str = f"{base_path}/{self.raw_datasets_path}/{asset_name}.{self.data_extention}"
            elif 'schema' in asset_name:
                path: str = f"{base_path}/{self.schemas_path}/{asset_name}.yaml"
            elif 'inference' in asset_name:
                path: str = f"{base_path}/{self.inference_path}/{asset_name}.json"
            elif 'transformer' in asset_name:
                transformer_name: str = self.__class__.__name__
                transformer_id: str = getattr(self, 'transformer_id')

                if transformer_id is None:
                    path: str = f"{base_path}/{self.transformers_path}/{transformer_name}/{asset_name}_attrs.pickle"
                else:
                    path: str = f"{base_path}/{self.transformers_path}/{transformer_name}/{transformer_id}/{asset_name}_attrs.pickle"
            else:
                path: str = f"{base_path}/{self.processing_datasets_path}/{asset_name}.{self.data_extention}"

        else:
            raise NotImplementedError(f'Storage environment "{self.storage_env}" has not been implemented yet.')
        
        return path

    def persist_dataset(
        self,
        df: pd.DataFrame,
        df_name: str,
        write_mode: str = None,
        mock: bool = False
    ) -> None:
        # Define path
        path: str = self.find_path(df_name, mock)

        if self.storage_env == 'filesystem':
            # Persist to filesystem
            save_to_filesystem(
                asset=df,
                path=path,
                partition_cols=self.partition_cols,
                write_mode=write_mode
            )
        elif self.storage_env == 'S3':
            # Persist to S3
            save_to_s3(
                asset=df,
                path=path,
                partition_cols=self.partition_cols,
                write_mode=write_mode
            )
        else:
            raise NotImplementedError(f'Storage environment "{self.storage_env}" has not been implemented yet.')

    def load_dataset(
        self,
        df_name: str,
        filters: List[Tuple[str, str, List[str]]] = None,
        mock: bool = False
    ) -> pd.DataFrame:        
        # Define path
        path: str = self.find_path(df_name, mock)
        
        if self.storage_env == 'filesystem':
            # Load from filesystem
            df: pd.DataFrame = load_from_filesystem(
                path=path,
                partition_cols=self.partition_cols,
                filters=filters
            )
        elif self.storage_env == 'S3':
            # Load from S3
            df: pd.DataFrame = load_from_s3(
                path=path,
                partition_cols=self.partition_cols,
                filters=filters
            )
        else:
            raise NotImplementedError(f'Storage environment "{self.storage_env}" has not been implemented yet.')
        
        return df
    
    def load_datasets(
        self,
        df_names: List[str],
        filters: List[Tuple[str, str, List[str]]] = None,
        mock: bool = False
    ) -> List[pd.DataFrame]:
        # Load datasets
        datasets: List[pd.DataFrame] = [
            self.load_dataset(df_name, filters, mock)
            for df_name in df_names
        ]
        
        return datasets

    def save_transformer(self) -> None:
        # Extract path
        transformer_id: str = getattr(self, 'transformer_id')
        path: str = self.find_path(f'{transformer_id}_transformer')

        # Define attrs to save
        attrs: dict = {key: value for (key, value) in self.__dict__.items() if key in self.pickled_attrs}

        if self.storage_env == 'filesystem':
            # Save attrs to filesystem
            save_to_filesystem(
                asset=attrs,
                path=path,
                partition_cols=None,
                write_mode=None
            )

        elif self.storage_env == 'S3':
            # Save attrs to S3
            save_to_s3(
                asset=attrs,
                path=path,
                partition_cols=None,
                write_mode=None
            )

        else:
            raise NotImplementedError(f'Storage environment "{self.storage_env}" has not been implemented yet.')

    def load_transformer(self) -> None:
        # Extract path
        transformer_id: str = getattr(self, 'transformer_id')
        path: str = self.find_path(f'{transformer_id}_transformer')
    
        if self.storage_env == 'filesystem':
            # Load from filesystem
            attrs: dict = load_from_filesystem(path=path, partition_cols=None, filters=None)
        
        elif self.storage_env == 'S3':
            # Load from S3
            attrs: dict = load_from_s3(path=path, partition_cols=None, filters=None)

        else:
            raise NotImplementedError(f'Storage environment "{self.storage_env}" has not been implemented yet.')

        # Assign pickled attrs
        for attr_name, attr_value in attrs.items():
            if attr_name in self.pickled_attrs:
                setattr(self, attr_name, attr_value)
