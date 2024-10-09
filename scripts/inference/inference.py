from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.etl import ExtractTransformLoad
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.modeling.model_registry import ModelRegistry
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
import numpy as np
import argparse


# Get logger
LOGGER = get_logger(name=__name__)


@timing
def main(X: pd.DataFrame) -> None:
    # Instanciate ModelRegistry
    MR: ModelRegistry = ModelRegistry()

    # Load production MLPipeline
    pipeline: MLPipeline = MR.load_prod_pipe()

    # Predict test
    y_pred: np.ndarray = pipeline.predict(X=X)

    # Interpret probabilities
    if 'classification' in pipeline.task:
        y_pred: np.ndarray = pipeline.model.interpret_score(y_pred)

    return y_pred

# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/inference/inference.py --arg_name arg_value
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Inference script.')

    # Add arguments
    parser.add_argument('--arg_name', type=str, default=None)

    # Extract arguments from parser
    args = parser.parse_args()
    arg_name: str = args.arg_name

    # Instanciate ExtractTransformLoad
    ETL: ExtractTransformLoad = ExtractTransformLoad()

    # Load persisted raw datasets
    X, _ = ETL.run_pipeline(
        persist_datasets=False,
        overwrite=False
    )

    # Run main
    main(X=X)