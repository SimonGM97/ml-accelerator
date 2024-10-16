from ml_accelerator.config.params import Params
from ml_accelerator.utils.data_helper.data_helper import DataHelper
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.modeling.model_registry import ModelRegistry
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
from tqdm import tqdm
import argparse
from typing import List


# Get logger
LOGGER = get_logger(name=__name__)


@timing
def main(
    train_prod_model: bool = True,
    train_staging_models: bool = True,
    train_dev_models: bool = False,
    debug: bool = False
) -> None:
    def fit_model(model: Model) -> None:
        # Fit Model
        model.fit(X=X_train, y=y_train)

        # Save Model
        model.save()

    # Log arguments
    log_params(
        logger=LOGGER,
        **{
            'train_prod_model': train_prod_model,
            'train_staging_models': train_staging_models,
            'train_dev_models': train_dev_models,
            'debug': debug
        }
    )

    # Instanciate DataHelper
    DH: DataHelper = DataHelper()

    # Load persisted datasets
    X: pd.DataFrame = DH.load_dataset(df_name='X_trans', filters=None)
    y: pd.Series = DH.load_dataset(df_name='y_trans', filters=None)

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

    # Fit prod model
    if train_prod_model:
        # Load Model
        model = MR.load_prod_model()

        # Fit Model
        LOGGER.info('Fitting %s prod model.', model.model_id)
        fit_model(model=model)

    # Fit staging models
    if train_staging_models:
        # Load Models
        models: List[Model] = MR.load_staging_models()

        # Fit Models
        LOGGER.info('Fitting staging models.')
        for model in tqdm(models):
            fit_model(model=model)

    # Fit development models
    if train_dev_models:
        # Load Models
        models: List[Model] = MR.load_dev_models()

        # Fit Models
        LOGGER.info('Fitting development models.')
        for model in tqdm(models):
            fit_model(model=model)


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/training/training.py --train_prod_model True --train_staging_models True --train_dev_models False
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Model training script.')

    # Add arguments
    parser.add_argument('--train_prod_model', type=bool, default=True)
    parser.add_argument('--train_staging_models', type=bool, default=True)
    parser.add_argument('--train_dev_models', type=bool, default=False)

    # Extract arguments from parser
    args = parser.parse_args()
    train_prod_model: int = args.train_prod_model
    train_staging_models: int = args.train_staging_models
    train_dev_models: int = args.train_dev_models

    # Run main
    main(
        train_prod_model=train_prod_model,
        train_staging_models=train_staging_models,
        train_dev_models=train_dev_models,
        debug=False
    )