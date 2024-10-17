#!/usr/bin/env python3
from ml_accelerator.config.params import Params
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.modeling.model_registry import ModelRegistry
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
import numpy as np
import argparse
from typing import List


# Get logger
LOGGER = get_logger(name=__name__)


@timing
def drift_pipeline(
    param_name: str = None,
    debug: bool = False
) -> None:
    # Log arguments
    log_params(
        logger=LOGGER,
        **{
            'param_name': param_name,
            'debug': debug
        }
    )

    # Instanciate ModelRegistry
    MR: ModelRegistry = ModelRegistry()

    # Extract pipeline_id
    pipeline_id: str = MR.registry_dict.get("production")[0]

    # Instanciate DataHelper
    DH: DataHelper = DataHelper()

    # Load inference_df
    inference_df: pd.DataFrame = DH.load_inference_df(
        pipeline_id=pipeline_id
    )

    LOGGER.info('inference_df:\n%s', inference_df)
    

# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/drift/drift.py --param_name param_value
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Model drift script.')

    # Add arguments
    parser.add_argument('--param_name', type=str, default=None, choices=['param_value', None])

    # Extract arguments from parser
    args = parser.parse_args()
    param_name: str = eval(args.param_name)

    # Run main
    drift_pipeline(
        param_name=param_name,
        debug=False
    )