#!/usr/bin/env python3
from ml_accelerator.data_processing.extract_transform_load import ExtractTransformLoad
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
from typing import Tuple
import argparse


# Get logger
LOGGER = get_logger(name=__name__)


@timing
def etl_pipeline(
    persist_datasets: bool = True,
    write_mode: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Log arguments
    log_params(
        logger=LOGGER,
        **{
            'persist_datasets': persist_datasets,
            'write_mode': write_mode
        }
    )

    # Instanciate ETL
    ETL: ExtractTransformLoad = ExtractTransformLoad()

    # Load input datasets
    df = ETL.run_pipeline(
        persist_datasets=persist_datasets,
        write_mode=write_mode
    )
    
    return df


"""
source .ml_accel_venv/bin/activate
conda deactivate
.ml_accel_venv/bin/python scripts/etl/etl.py \
    --persist_datasets True \
    --write_mode overwrite
"""
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Data processing script.')

    # Add arguments
    parser.add_argument('--persist_datasets', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--write_mode', type=str, default='append', choices=['append', 'overwrite'])

    # Extract arguments from parser
    args = parser.parse_args()
    persist_datasets: bool = eval(args.persist_datasets)
    write_mode: str = args.write_mode

    # Run etl pipeline
    etl_pipeline(
        persist_datasets=persist_datasets,
        write_mode=write_mode
    )

