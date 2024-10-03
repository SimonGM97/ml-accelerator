import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
import pickle
import json
import yaml

def load_from_filesystem(
    path: str
) -> pd.DataFrame | dict:
    # Extract format
    read_format = path.split('.')[-1]

    if read_format == 'csv':
        # Read csv file
        asset: pd.DataFrame = pd.read_csv(path)
    elif read_format == 'parquet':
        pass
    elif read_format == 'pickle':
        pass
    elif read_format == 'json':
        pass
    elif read_format == 'yaml':
        pass
    else:
        raise Exception(f'Invalid read_format was received: "{read_format}".\n')

    return asset

def save_to_filesystem(
    asset,
    path: str,
    partition_column: str = None
) -> None:
    # Extract format
    save_format = path.split('.')[-1]

    if save_format == 'csv':
        # Save csv file
        asset: pd.DataFrame = asset
        asset.to_csv(path)
    elif save_format == 'parquet':
        pass
    elif save_format == 'pickle':
        pass
    elif save_format == 'json':
        pass
    elif save_format == 'yaml':
        pass
    else:
        raise Exception(f'Invalid save_format was received: "{save_format}".\n')