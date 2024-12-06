#!/usr/bin/env python3
from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.extract_transform_load import ExtractTransformLoad
from ml_accelerator.modeling.model_registry import ModelRegistry
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.inference.inference_utils import save_inference
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
import numpy as np
from datetime import datetime
import argparse


# Get logger
LOGGER = get_logger(name=__name__)


@timing
def inference_pipeline(pred_id = None) -> dict:
    # Log arguments
    log_params(
        logger=LOGGER,
        **{'pred_id': pred_id}
    )
    
    # Instanciate ExtractTransformLoad
    ETL: ExtractTransformLoad = ExtractTransformLoad()

    # Extract new df
    df: pd.DataFrame = ETL.run_pipeline(pred_id=pred_id)

    # Extract X
    X: pd.DataFrame = ETL.divide_datasets(
        df=df, 
        test_size=0, 
        balance_train=False,
        balance_method=None,
        persist_datasets=False,
        write_mode=None,
        mock_datasets=False,
        debug=False
    )[0]

    # Instanciate ModelRegistry
    MR: ModelRegistry = ModelRegistry()

    # Load production MLPipeline
    pipeline: MLPipeline = MR.load_prod_pipe()

    # Predict test
    y_pred: np.ndarray = pipeline.predict(X=X)

    # Interpret probabilities
    if 'classification' in pipeline.task:
        y_pred: np.ndarray = pipeline.estimator.interpret_score(y_pred)

    # Prepare new inference
    inference: dict = {
        'pipeline_id': pipeline.pipeline_id,
        'pred_id': pred_id,
        'prediction': y_pred.tolist(),
        'features': X.columns.tolist(),
        'X': X.values.tolist(),
        'date': str(datetime.today()),
        'year': str(datetime.today().year),
        'month': ('0' + str(datetime.today().month))[-2:],
        'day': ('0' + str(datetime.today().day))[-2:]
    }

    # Save Inference
    save_inference(inference=inference)

    return inference


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/inference/inference.py --pred_id 0
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Inference script.')

    # Add arguments
    parser.add_argument('--pred_id', default=None)

    # Extract arguments from parser
    args = parser.parse_args()
    pred_id = args.pred_id

    # Run inference pipeline
    inference_pipeline(pred_id=pred_id)