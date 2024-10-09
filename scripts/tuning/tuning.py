from ml_accelerator.config.params import Params
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.modeling.model_tuning import ModelTuner
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
import argparse


# Get logger
LOGGER = get_logger(
    name=__name__,
    level=Params.LEVEL,
    txt_fmt=Params.TXT_FMT,
    json_fmt=Params.JSON_FMT,
    filter_lvls=Params.FILTER_LVLS,
    log_file=Params.LOG_FILE,
    backup_count=Params.BACKUP_COUNT
)

@timing
def main(
    max_evals: int = Params.MAX_EVALS,
    loss_threshold: float = Params.LOSS_THRESHOLD,
    timeout_mins: float = Params.TIMEOUT_MINS,
    debug: bool = False
) -> None:
    # Log arguments
    log_params(
        logger=LOGGER,
        **{
            'max_evals': max_evals,
            'loss_threshold': loss_threshold,
            'timeout_mins': timeout_mins,
            'debug': debug
        }
    )

    # Instanciate DataHelper
    DH: DataHelper = DataHelper(
        dataset_name=Params.DATASET_NAME,
        bucket=Params.BUCKET,
        cwd=Params.CWD,
        storage_env=Params.DATA_STORAGE_ENV,
        training_path=Params.TRAINING_PATH,
        inference_path=Params.INFERENCE_PATH,
        transformers_path=Params.TRANSFORMERS_PATH,
        data_extention=Params.DATA_EXTENTION,
        partition_cols=Params.PARTITION_COLUMNS
    )

    # Load persisted datasets
    X: pd.DataFrame = DH.load_dataset(df_name='X_trans', filters=None)
    y: pd.Series = DH.load_dataset(df_name='y_trans', filters=None)

    # Divide into X_train, X_test, y_train & y_test
    X_train, X_test, y_train, y_test = DH.divide_datasets(
        X=X, y=y, 
        test_size=Params.TEST_SIZE, 
        balance_train=Params.BALANCE_TRAIN,
        balance_method=Params.BALANCE_METHOD,
        debug=debug
    )

    # Instanciate ModelTuner
    MT: ModelTuner = ModelTuner(
        algorithms=Params.ALGORITHMS,
        search_space=Params.SEARCH_SPACE,
        target=Params.TARGET,
        task=Params.TASK,
        optimization_metric=Params.OPTIMIZATION_METRIC,
        importance_method=Params.IMPORTANCE_METHOD,
        n_candidates=Params.N_CANDIDATES,
        min_performance=Params.MIN_PERFORMANCE,
        val_splits=Params.VAL_SPLITS,
        model_storage_env=Params.MODEL_STORAGE_ENV,
        data_storage_env=Params.DATA_STORAGE_ENV,
        bucket=Params.BUCKET,
        models_path=Params.MODELS_PATH
    )

    # Tune models
    MT.tune_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selected_features=X_train.columns.tolist(),
        use_warm_start=True,
        max_evals=max_evals,
        loss_threshold=loss_threshold,
        timeout_mins=timeout_mins,
        debug=debug
    )


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/tuning/tuning.py --max_evals 1000 --loss_threshold 0.995 --timeout_mins 15
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Model tuning script.')

    # Add arguments
    parser.add_argument('--max_evals', type=int, default=Params.MAX_EVALS)
    parser.add_argument('--loss_threshold', type=float, default=Params.LOSS_THRESHOLD)
    parser.add_argument('--timeout_mins', type=float, default=Params.TIMEOUT_MINS)

    # Extract arguments from parser
    args = parser.parse_args()
    max_evals: int = args.max_evals
    loss_threshold: float = args.loss_threshold
    timeout_mins: float = args.timeout_mins

    # Run main
    main(
        max_evals=max_evals,
        loss_threshold=loss_threshold,
        timeout_mins=timeout_mins
    )

