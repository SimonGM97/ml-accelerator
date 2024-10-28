#!/usr/bin/env python3
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.utils.transformers.transformers_utils import load_transformers_list
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
from typing import Tuple
import argparse


# Get logger
LOGGER = get_logger(name=__name__)

@timing
def data_pipeline(
    fit_transformers: bool = False,
    save_transformers: bool = False,
    persist_datasets: bool = True,
    write_mode: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Log arguments
    log_params(
        logger=LOGGER,
        **{
            'fit_transformers': fit_transformers,
            'save_transformers': save_transformers,
            'persist_datasets': persist_datasets,
            'write_mode': write_mode
        }
    )

    # Instanciate DataHelper
    DH: DataHelper = DataHelper()

    # Load input dataset
    df_raw: pd.DataFrame = DH.load_dataset(
        df_name='df_raw',
        filters=None
    )

    # Divide datasets
    X, _, y, _ = DH.divide_datasets(
        df=df_raw,
        test_size=0,
        balance_train=False,
        balance_method=None,
        debug=True
    )
    
    # Extract transformers
    transformers = load_transformers_list(transformer_id='base')

    # Instanciate ML Pipeline
    MLP: MLPipeline = MLPipeline(
        transformers=transformers,
        estimator=None
    )

    # Run ML Pipeline
    if fit_transformers:
        X, y = MLP.fit_transform(
            X=X, y=y,
            persist_datasets=persist_datasets,
            write_mode=write_mode,
            debug=True
        )
    else:
        X, y = MLP.transform(
            X=X, y=y,
            persist_datasets=persist_datasets,
            write_mode=write_mode,
            debug=False
        )

    # Save transformers
    if save_transformers:
        MLP.save()

    return X, y


"""
source .ml_accel_venv/bin/activate
conda deactivate
.ml_accel_venv/bin/python scripts/data_processing/data_processing.py \
    --fit_transformers True \
    --save_transformers True \
    --persist_datasets True \
    --write_mode overwrite
"""
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Data processing script.')

    # Add arguments
    parser.add_argument('--fit_transformers', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--save_transformers', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--persist_datasets', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--write_mode', type=str, default='append', choices=['append', 'overwrite'])

    # Extract arguments from parser
    args = parser.parse_args()
    fit_transformers: bool = eval(args.fit_transformers)
    save_transformers: bool = eval(args.save_transformers)
    persist_datasets: bool = eval(args.persist_datasets)
    write_mode: str = args.write_mode

    # Run main
    data_pipeline(
        fit_transformers=fit_transformers,
        save_transformers=save_transformers,
        persist_datasets=persist_datasets,
        write_mode=write_mode
    )

