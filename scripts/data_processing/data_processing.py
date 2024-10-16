from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.etl import ExtractTransformLoad
from ml_accelerator.data_processing.data_cleaning import DataCleaner
from ml_accelerator.data_processing.data_transforming import DataTransformer
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

import argparse


# Get logger
LOGGER = get_logger(name=__name__)

@timing
def main(
    fit_transformers: bool = False,
    save_transformers: bool = False,
    persist_datasets: bool = True,
    overwrite: bool = True
) -> None:
    # Log arguments
    log_params(
        logger=LOGGER,
        **{
            'persist_datasets': persist_datasets,
            'overwrite': overwrite,
            'fit_transformers': fit_transformers
        }
    )

    # Instanciate ETL
    ETL: ExtractTransformLoad = ExtractTransformLoad()

    # Load input datasets
    X, y = ETL.run_pipeline(
        persist_datasets=persist_datasets,
        overwrite=overwrite
    )
    
    # Instanciate DataCleaner
    DC: DataCleaner = DataCleaner()

    # Instanciate DataTransformer
    DT: DataTransformer = DataTransformer()

    # Instanciate ML Pipeline
    MLP: MLPipeline = MLPipeline(
        DC=DC,
        DT=DT,
        model=None
    )

    # Run ML Pipeline
    if fit_transformers:
        X, y = MLP.fit_transform(
            X=X, y=y,
            persist_datasets=persist_datasets,
            overwrite=overwrite
        )
    else:
        X, y = MLP.transform(
            X=X, y=y,
            persist_datasets=persist_datasets,
            overwrite=overwrite
        )

    # Save transformers
    if save_transformers:
        MLP.save()

    return X, y


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/data_processing/data_processing.py --fit_transformers True --save_transformers True --persist_datasets True --overwrite True
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Data processing script.')

    # Add arguments
    parser.add_argument('--fit_transformers', type=bool, default=False, choices=[True, False])
    parser.add_argument('--save_transformers', type=bool, default=False, choices=[True, False])
    parser.add_argument('--persist_datasets', type=bool, default=True, choices=[True, False])
    parser.add_argument('--overwrite', type=bool, default=True, choices=[True, False])

    # Extract arguments from parser
    args = parser.parse_args()
    fit_transformers: bool = args.fit_transformers
    save_transformers: bool = args.save_transformers
    persist_datasets: bool = args.persist_datasets
    overwrite: bool = args.overwrite

    # Run main
    main(
        fit_transformers=fit_transformers,
        save_transformers=save_transformers,
        persist_datasets=persist_datasets,
        overwrite=overwrite
    )

