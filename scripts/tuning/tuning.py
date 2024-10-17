#!/usr/bin/env python3
from ml_accelerator.config.params import Params
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.modeling.model_tuning import ModelTuner
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd


# Get logger
LOGGER = get_logger(name=__name__)

@timing
def tuning_pipeline() -> None:
    # Instanciate DataHelper
    DH: DataHelper = DataHelper()

    # Load persisted datasets
    X: pd.DataFrame = DH.load_dataset(df_name='X_trans', filters=None)
    y: pd.Series = DH.load_dataset(df_name='y_trans', filters=None)

    # Divide into X_train, X_test, y_train & y_test
    X_train, X_test, y_train, y_test = DH.divide_datasets(
        X=X, y=y, 
        test_size=Params.TEST_SIZE, 
        balance_train=Params.BALANCE_TRAIN,
        balance_method=Params.BALANCE_METHOD,
        debug=False
    )

    # Instanciate ModelTuner
    MT: ModelTuner = ModelTuner()

    # Tune models
    MT.tune_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selected_features=X_train.columns.tolist(),
        use_warm_start=True,
        max_evals=Params.MAX_EVALS,
        loss_threshold=Params.LOSS_THRESHOLD,
        timeout_mins=Params.TIMEOUT_MINS,
        debug=False
    )


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/tuning/tuning.py
if __name__ == "__main__":
    # Run main
    tuning_pipeline()

