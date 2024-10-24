from ml_accelerator.config.env import Env
from ml_accelerator.utils.logging.logger_helper import get_logger
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
import shutil
import pickle
import json
import yaml
from tqdm import tqdm
from pprint import pformat
from typing import List, Set, Tuple, Any


# Get logger
LOGGER = get_logger(name=__name__)


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
            paths: Set[str] = find_paths(bucket_name=bucket, directory=prefix)

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


def find_paths(
    bucket_name: str,
    directory: str
) -> Set[str]:
    # Define empty paths
    found_paths: Set[str] = set()

    # Define search dir
    search_dir = os.path.join(bucket_name, *directory.split('/'))
    
    # Append paths
    for root, subdirs, files in os.walk(search_dir):
        for file in files:
            new_path = os.path.join(*root.split('/'), *'/'.join(subdirs), file)
            found_paths.update({new_path})

    return found_paths


def find_subdirs(
    bucket_name: str,
    directory: str
) -> List[str]:
    # Define empty paths
    found_subdirs: Set[str] = set()

    # Define search dir
    search_dir = os.path.join(bucket_name, *directory.split('/'))

    # Append paths
    for root, subdirs, _ in os.walk(search_dir):
        for subdir in subdirs:
            new_subdir = os.path.join(*root.split('/'), subdir)
            found_subdirs.update({new_subdir})

    return found_subdirs


def remove_directory(
    bucket: str,
    directory: str
) -> None:
    # Remove files
    paths: Set[str] = find_paths(bucket_name=bucket, directory=directory)
    for path in paths:
        try:
            os.remove(path)
        except Exception as e:
            # LOGGER.warning(
            #     'Unable to remove %s.\n'
            #     'Exception: %s', path, e
            # )
            pass
    
    # Remove subdirs
    subdirs: Set[str] = find_subdirs(bucket_name=bucket, directory=directory)
    for subdir in subdirs:
        try:
            os.removedirs(subdir)
        except Exception as e:
            # LOGGER.warning(
            #     'Unable to remove %s.\n'
            #     'Exception: %s', subdir, e
            # )
            pass


def delete_bucket(bucket_name: str) -> None:
    LOGGER.info('Deleting bucket: %s (filesystem).', bucket_name)

    # Rmove all files & directories withing the bucket
    remove_directory(bucket=bucket_name, directory='')

    # Remove empty bucket
    try:
        os.removedirs(bucket_name)
    except Exception as e:
        LOGGER.warning(
            'Unable to remove %s.\n'
            'Exception: %s', bucket_name, e
        )

    # Check that all files & directories have been deleted
    paths: Set[str] = find_paths(bucket_name=bucket_name, directory='')
    subdirs: Set[str] = find_subdirs(bucket_name=bucket_name, directory='')

    if len(paths) == 0 and len(subdirs) == 0:
        LOGGER.info('Bucket: %s (filesystem) was successfully deleted.', bucket_name)
    else:
        LOGGER.warning(
            'Bucket: %s (filesystem) was NOT successfully deleted.\n'
            'Paths remaining (%s):\n%s\n'
            'Subdirs remaining (%s):\n%s', 
            bucket_name, len(paths), pformat(paths), len(subdirs), pformat(subdirs)
        )


def copy_bucket(
    source_bucket: str,
    destination_bucket: str,
    subdir: str = '',
    delete_destination: bool = False
) -> None:
    # Delete destination bucket
    if delete_destination:
        delete_bucket(bucket_name=destination_bucket)

    # Find source paths
    source_paths: Set[str] = find_paths(bucket_name=source_bucket, directory=subdir)

    LOGGER.info('Copying bucket %s (filesystem) into %s (filesystem).', source_bucket, destination_bucket)

    # Copy files
    for source_path in tqdm(source_paths):
        # Define destination_path
        destination_path = source_path.replace(source_bucket, destination_bucket)

        # Create the destination directory if it doesn't exist
        destination_dir = os.path.dirname(destination_path)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Copy file
        shutil.copy(source_path, destination_path)

    LOGGER.info('Bucket %s (filesystem) was successfully copied into %s (filesystem).', source_bucket, destination_bucket)


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python ml_accelerator/utils/filesystem/filesystem_helper.py
if __name__ == "__main__":
    # Copy bucket into new dummy-bucket
    copy_bucket(
        source_bucket=Env.get("BUCKET_NAME"), 
        destination_bucket='dummy-bucket', 
        subdir='',
        delete_destination=True
    )

    # Delete dummy-bucket
    delete_bucket(bucket_name='dummy-bucket')
