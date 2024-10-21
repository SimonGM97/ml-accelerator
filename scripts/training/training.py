#!/usr/bin/env python3
from ml_accelerator.config.params import Params
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.modeling.model_registry import ModelRegistry
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
from tqdm import tqdm
import argparse
from typing import List


# Get logger
LOGGER = get_logger(name=__name__)


@timing
def training_pipeline(
    train_prod_pipe: bool = True,
    train_staging_pipes: bool = True,
    train_dev_pipes: bool = False,
    debug: bool = False
) -> None:
    def fit_pipeline(
        pipeline: MLPipeline,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame
    ) -> None:
        # Fit MLPipeline
        pipeline.fit(
            X_train=X_train,
            y_train=y_train,
            fit_transformers=True,
            ignore_steps=['FeatureSelector']
        )

        # Save MLPipeline
        pipeline.save()

    # Log arguments
    log_params(
        logger=LOGGER,
        **{
            'train_prod_pipe': train_prod_pipe,
            'train_staging_pipes': train_staging_pipes,
            'train_dev_pipes': train_dev_pipes,
            'debug': debug
        }
    )

    # Instanciate DataHelper
    DH: DataHelper = DataHelper()

    # Load persisted datasets
    X: pd.DataFrame = DH.load_dataset(df_name='X_raw', filters=None)
    y: pd.Series = DH.load_dataset(df_name='y_raw', filters=None)

    # Divide into X_train, X_test, y_train & y_test
    X_train, _, y_train, _ = DH.divide_datasets(
        X=X, y=y, 
        test_size=Params.TEST_SIZE, 
        balance_train=Params.BALANCE_TRAIN,
        balance_method=Params.BALANCE_METHOD,
        debug=debug
    )

    # Instanciate ModelRegistry
    MR: ModelRegistry = ModelRegistry()

    # Fit Production MLPipeline
    if train_prod_pipe:
        # Load MLPipeline
        pipeline: MLPipeline = MR.load_prod_pipe()

        # Fit MLPipeline
        LOGGER.info('Fitting %s Production MLPipeline.', pipeline.pipeline_id)
        fit_pipeline(
            pipeline=pipeline,
            X_train=X_train.copy(),
            y_train=y_train.copy()
        )

    # Fit Staging MLPipelines
    if train_staging_pipes:
        # Load MLPipelines
        pipelines: List[MLPipeline] = MR.load_staging_pipes()

        # Fit MLPipelines
        LOGGER.info('Fitting Staging MLPipelines.')
        for pipeline in tqdm(pipelines):
            fit_pipeline(
                pipeline=pipeline,
                X_train=X_train.copy(),
                y_train=y_train.copy()
            )

    # Fit Development MLPipelines
    if train_dev_pipes:
        # Load MLPipelines
        pipelines: List[MLPipeline] = MR.load_dev_pipes()

        # Fit MLPipelines
        LOGGER.info('Fitting Development MLPipelines.')
        for pipeline in tqdm(pipelines):
            fit_pipeline(
                pipeline=pipeline,
                X_train=X_train.copy(),
                y_train=y_train.copy()
            )


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/training/training.py --train_prod_pipe True --train_staging_pipes True --train_dev_pipes True
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Model training script.')

    # Add arguments
    parser.add_argument('--train_prod_pipe', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--train_staging_pipes', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--train_dev_pipes', type=str, default='False', choices=['True', 'False'])

    # Extract arguments from parser
    args = parser.parse_args()
    train_prod_pipe: bool = eval(args.train_prod_pipe)
    train_staging_pipes: bool = eval(args.train_staging_pipes)
    train_dev_pipes: bool = eval(args.train_dev_pipes)

    # Run main
    training_pipeline(
        train_prod_pipe=train_prod_pipe,
        train_staging_pipes=train_staging_pipes,
        train_dev_pipes=train_dev_pipes,
        debug=False
    )