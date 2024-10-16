from ml_accelerator.config.params import Params
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.data_processing.data_cleaning import DataCleaner
from ml_accelerator.data_processing.data_transforming import DataTransformer
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.modeling.model_registry import ModelRegistry
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from typing import List


# Get logger
LOGGER = get_logger(name=__name__)


@timing
def main(
    evaluate_prod_pipe: bool = True,
    evaluate_staging_pipes: bool = True,
    evaluate_dev_pipes: bool = False,
    update_model_stages: bool = False,
    update_prod_model: bool = False,
    debug: bool = False
) -> None:
    def evaluate_pipeline(
        pipeline: MLPipeline,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame
    ) -> None:
        # Predict test
        y_pred: np.ndarray = pipeline.predict(X=X_test)

        # Encode y_test
        if 'classification' in pipeline.task:
            y_test: pd.DataFrame = pipeline.DT._encode_target(y=y_test)

        # Evaluate predictions
        pipeline.evaluate(y_pred=y_pred, y_test=y_test)

        # Save pipeline
        pipeline.save()

    # Log arguments
    log_params(
        logger=LOGGER,
        **{
            'evaluate_prod_pipe': evaluate_prod_pipe,
            'evaluate_staging_pipes': evaluate_staging_pipes,
            'evaluate_dev_pipes': evaluate_dev_pipes,
            'debug': debug
        }
    )

    # Instanciate DataHelper
    DH: DataHelper = DataHelper()

    # Load persisted raw datasets
    X: pd.DataFrame = DH.load_dataset(df_name='X_raw', filters=None)
    y: pd.Series = DH.load_dataset(df_name='y_raw', filters=None)

    # Divide into X_train, X_test, y_train & y_test
    _, X_test, _, y_test = DH.divide_datasets(
        X=X, y=y, 
        test_size=Params.TEST_SIZE, 
        balance_train=Params.BALANCE_TRAIN,
        balance_method=Params.BALANCE_METHOD,
        debug=debug
    )

    # Instanciate ModelRegistry
    MR: ModelRegistry = ModelRegistry()

    # Evaluate prod pipeline
    if evaluate_prod_pipe:
        # Load MLPipeline
        pipeline: MLPipeline = MR.load_prod_pipe()

        # Evaluate Model
        LOGGER.info('Evaluating %s Production MLPipeline.', pipeline.pipeline_id)
        evaluate_pipeline(
            pipeline=pipeline,
            X_test=X_test.copy(),
            y_test=y_test.copy()
        )

    # Fit staging pipelines
    if evaluate_staging_pipes:
        # Load MLPipelines
        pipelines: List[MLPipeline] = MR.load_staging_pipes()

        # Evaluate MLPipeline
        LOGGER.info('Evaluating Staging MLPipelines.')
        for pipeline in tqdm(pipelines):
            evaluate_pipeline(
                pipeline=pipeline,
                X_test=X_test.copy(),
                y_test=y_test.copy()
            )

    # Fit development pipelines
    if evaluate_dev_pipes:
        # Load MLPipelines
        pipelines: List[MLPipeline] = MR.load_dev_pipes()

        # Evaluate Models
        LOGGER.info('Evaluating Development MLPipelines.')
        for pipeline in tqdm(pipelines):
            evaluate_pipeline(
                pipeline=pipeline,
                X_test=X_test.copy(),
                y_test=y_test.copy()
            )

    # Update model stages
    if update_model_stages:
        MR.update_model_stages(
            update_prod_model=update_prod_model,
            debug=debug
        )

    # Show model performances
    LOGGER.info("%s", MR)


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/evaluation/evaluation.py --evaluate_prod_pipe True --evaluate_staging_pipes True --evaluate_dev_pipes True  --update_model_stages True  --update_prod_model True
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Model evaluating script.')

    # Add arguments
    parser.add_argument('--evaluate_prod_pipe', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--evaluate_staging_pipes', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--evaluate_dev_pipes', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--update_model_stages', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--update_prod_model', type=str, default='False', choices=['True', 'False'])

    # Extract arguments from parser
    args = parser.parse_args()
    evaluate_prod_pipe: bool = eval(args.evaluate_prod_pipe)
    evaluate_staging_pipes: bool = eval(args.evaluate_staging_pipes)
    evaluate_dev_pipes: bool = eval(args.evaluate_dev_pipes)
    update_model_stages: bool = eval(args.update_model_stages)
    update_prod_model: bool = eval(args.update_prod_model)

    # Run main
    main(
        evaluate_prod_pipe=evaluate_prod_pipe,
        evaluate_staging_pipes=evaluate_staging_pipes,
        evaluate_dev_pipes=evaluate_dev_pipes,
        update_model_stages=update_model_stages,
        update_prod_model=update_prod_model,
        debug=False
    )