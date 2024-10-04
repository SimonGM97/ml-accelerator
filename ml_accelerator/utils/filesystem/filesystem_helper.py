import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
import pickle
import json
import yaml
from typing import List, Set, Tuple, Any


def load_from_filesystem(
    path: str,
    partition_cols: List[str] = None,
    filters: List[Tuple[str, str, List[str]]] = None
) -> pd.DataFrame | dict:
    # Extract format
    read_format = path.split('.')[-1]

    if read_format == 'csv':
        # Load csv file
        asset: pd.DataFrame = pd.read_csv(path)

    elif read_format == 'parquet':
        # Create a Parquet dataset
        dataset = pq.ParquetDataset(
            path_or_paths=path, # files,
            # filesystem=FS,
            filters=filters
        )

        # Read the dataset into a Pandas DataFrame
        asset: pd.DataFrame = dataset.read_pandas().to_pandas()

    elif read_format == 'pickle':
        # Load pickle file
        with open(path, 'rb') as handle:
            asset: dict = pickle.load(handle)

    elif read_format == 'json':
        # Load json file
        asset: dict = json.load(open(path))

    elif read_format == 'yaml':
        # Load yaml file
        with open(path) as file:
            asset: dict = yaml.load(file, Loader=yaml.FullLoader)

    else:
        raise Exception(f'Invalid read_format was received: "{read_format}".\n')

    return asset


def save_to_filesystem(
    asset,
    path: str,
    partition_cols: List[str] = None,
    overwrite: bool = True
) -> None:
    # Extract format
    save_format = path.split('.')[-1]

    if save_format == 'csv':
        # Copy asset
        asset: pd.DataFrame = asset

        # Save csv file        
        asset.to_csv(path)

    elif save_format == 'parquet':
        # Copy asset
        asset: pd.DataFrame = asset.copy(deep=True)

        # Remove extention
        prefix = path.replace(".parquet", "")

        # Extract existing_data_behavior
        if overwrite:
            # Delete all found files before writing a new one
            existing_data_behavior = 'delete_matching'
        else:
            # (Append) Overwrite new partitions while leaving old ones
            existing_data_behavior = 'overwrite_or_ignore'

        # Write PyArrow Table as a parquet file, partitioned by year_quarter
        if overwrite:
            if partition_cols is None:
                os.remove(os.path.join(prefix, 'dataset-0.parquet'))
            else:
                # Delete objects
                remove_directory(directory=prefix)
        
        pq.write_to_dataset(
            pa.Table.from_pandas(asset),
            root_path=prefix,
            partition_cols=partition_cols,
            # filesystem=FS,
            schema=pa.Schema.from_pandas(asset),
            basename_template='dataset-{i}.parquet',
            use_threads=True,
            compression='snappy',
            existing_data_behavior=existing_data_behavior
        )

    elif save_format == 'pickle':
        # Save pickle file
        with open(os.path.join(path), 'wb') as handle:
            pickle.dump(asset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif save_format == 'json':
        # Save json file
        with open(path, "w") as f:
            json.dump(asset, f, indent=4)

    elif save_format == 'yaml':
        # Save yaml file
        with open(path, 'w') as file:
            yaml.dump(asset, file)

    else:
        raise Exception(f'Invalid save_format was received: "{save_format}".\n')
    

def remove_directory(directory: str) -> None:
    for root, directories, files in os.walk(directory):
        for file in files:
            delete_path = os.path.join(directory, file)
            # print(f"Deleting {delete_path}.")
            os.remove(delete_path)