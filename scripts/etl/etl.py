#!/usr/bin/env python3
from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.extract_transform_load import ExtractTransformLoad
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
import json
from typing import Tuple
import argparse
from pprint import pformat


# Get logger
LOGGER = get_logger(name=__name__)


@timing
def etl_pipeline(
    persist_datasets: bool = True,
    write_mode: str = None,
    debug: bool = False
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
    df_raw = ETL.run_pipeline(
        persist_datasets=persist_datasets,
        write_mode=write_mode,
        mock_datasets=False,
        debug=debug
    )

    # Divide datasets
    X_train, X_test, y_train, y_test = ETL.divide_datasets(
        df=df_raw,
        test_size=Params.TEST_SIZE,
        balance_train=Params.BALANCE_TRAIN,
        balance_method=Params.BALANCE_METHOD,
        persist_datasets=persist_datasets,
        write_mode=write_mode,
        mock_datasets=False,
        debug=debug
    )
    
    return X_train, X_test, y_train, y_test


def lambda_handler(
    event: dict, 
    context: dict = None
) -> dict:
    """
    :param `event`: (dict) Data sent during lambda function invocation.
    :param `context`: (dict) Generated by the platform and contains information about the underlying infrastructure
        and execution environment, such as allowed runtime and memory.
    """
    # Log event & context
    LOGGER.info('event:\n%s', pformat(event))
    LOGGER.info('context:\n%s', pformat(context))

    if "AWS_LAMBDA_EVENT_BODY" in event.keys():
        # Access the payload from the event parameter
        payload_str = event.get("AWS_LAMBDA_EVENT_BODY")

        # Parse the JSON payload
        payload: dict = json.loads(payload_str)

        # Extract parameters
        persist_datasets = eval(payload.get("persist_datasets", "True"))
        write_mode = payload.get("write_mode", "overwrite")
        debug = eval(payload.get("debug", "False"))
    else:
        # Extract parameters
        persist_datasets = eval(event.get("persist_datasets", "True"))
        write_mode = event.get("write_mode", "overwrite")
        debug = eval(event.get("debug", "False"))
    
    # Run ETL pipeline
    datasets: Tuple[pd.DataFrame] = etl_pipeline(
        persist_datasets=persist_datasets,
        write_mode=write_mode,
        debug=debug
    )

    return {
        'statusCode': 200,
        'body': json.dumps('ETL job ran successfully!')
    }


"""
source .ml_accel_venv/bin/activate
conda deactivate
.ml_accel_venv/bin/python scripts/etl/etl.py \
    --persist_datasets True \
    --write_mode overwrite
{
  "persist_datasets": "True",
  "write_mode": "overwrite"
}
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

    # Run lambda handler
    # lambda_handler(
    #     event={
    #         "persist_datasets": persist_datasets,
    #         "write_mode": write_mode
    #     }
    # )

