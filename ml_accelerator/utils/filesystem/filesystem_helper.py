from ml_accelerator.utils.logging.logger_helper import get_logger
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
import pickle
import json
import yaml
from typing import List, Set, Tuple, Any


# Get logger
LOGGER = get_logger(name=__name__)


def find_paths(
    bucket: str,
    directory: str
) -> List[str]:
    # Define empty paths
    found_paths: List[str] = []

    # Define search dir
    search_dir = os.path.join(bucket, *directory.split('/'))

    # Append paths
    for root, subdirs, files in os.walk(search_dir):
        for file in files:
            new_path = os.path.join(*root.split('/'), *'/'.join(subdirs), file)
            found_paths.append(new_path)

    return found_paths


def find_subdirs(
    bucket: str,
    directory: str
) -> List[str]:
    # Define empty paths
    found_subdirs: List[str] = []

    # Define search dir
    search_dir = os.path.join(bucket, *directory.split('/'))

    # Append paths
    for root, subdirs, _ in os.walk(search_dir):
        for subdir in subdirs:
            new_subdir = os.path.join(*root.split('/'), subdir)
            found_subdirs.append(new_subdir)

    return subdirs


def load_from_filesystem(
    path: str,
    partition_cols: List[str] = None,
    filters: List[Tuple[str, str, List[str]]] = None
) -> pd.DataFrame | dict:
    # Extract bucket, key & read_format
    bucket: str = path.split('/')[0]
    key: str = '/'.join(path.split('/')[1:])
    read_format = path.split('.')[-1]

    try:
        if read_format == 'csv':
            # Load csv file
            asset: pd.DataFrame = pd.read_csv(path, index_col=0)

        elif read_format == 'parquet':
            # Remove extention
            prefix = key.replace(".parquet", "")

            # Find paths
            paths = find_paths(bucket=bucket, directory=prefix)

            # Create a Parquet dataset
            dataset = pq.ParquetDataset(
                path_or_paths=paths,
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
    except Exception as e:
        LOGGER.warning(
            'Unable to load %s from filesystem.\n'
            'Exception: %s',
            path, e
        )
        asset = None

    return asset


def remove_directory(
    bucket: str,
    directory: str
) -> None:
    # Remove files
    paths: List[str] = find_paths(bucket=bucket, directory=directory)
    for path in paths:
        try:
            os.remove(path)
        except Exception as e:
            LOGGER.warning(
                'Unable to remove %s.\n'
                'Exception: %s', path, e
            )
    
    # Remove subdirs
    subdirs: List[str] = find_subdirs(bucket=bucket, directory=directory)
    for subdir in subdirs:
        try:
            os.removedirs(subdir)
        except Exception as e:
            LOGGER.warning(
                'Unable to remove %s.\n'
                'Exception: %s', subdir, e
            )

def save_to_filesystem(
    asset,
    path: str,
    partition_cols: List[str] = None,
    write_mode: str = None
) -> None:
    # Make sure directory exists
    if not os.path.exists(path):
        makedir = '/'.join(path.split('/')[:-1])
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    # Validate write_mode
    if write_mode is None:
        write_mode = 'append'

    # Extract format
    save_format = path.split('.')[-1]

    if save_format == 'csv':
        # Copy asset
        asset: pd.DataFrame = asset.copy(deep=True)

        # Save csv file        
        asset.to_csv(path, columns=asset.columns.tolist(), index=True)

    elif save_format == 'parquet':
        # Copy asset
        asset: pd.DataFrame = asset.copy(deep=True)

        # Remove extention
        prefix = path.replace(".parquet", "")

        # Extract existing_data_behavior
        if write_mode == 'overwrite':
            # Delete all found files before writing a new one
            existing_data_behavior = 'delete_matching'
        elif write_mode == 'append':
            # (Append) Overwrite new partitions while leaving old ones
            existing_data_behavior = 'overwrite_or_ignore'
        else:
            raise NotImplementedError(f'Invalid "write_mode" parameter was received: {write_mode}')

        # Write PyArrow Table as a parquet file, partitioned by year_quarter
        if write_mode == 'overwrite':
            if partition_cols is None:
                try:
                    os.remove(os.path.join(prefix, 'dataset-0.parquet'))
                except:
                    pass
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
        with open(path, 'wb') as handle:
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
