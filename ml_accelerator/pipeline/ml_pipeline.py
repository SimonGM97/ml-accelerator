from ml_accelerator.data_processing.transformers.transformer import Transformer
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.timing.timing_helper import timing

import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings

# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Get logger
LOGGER = get_logger(name=__name__)


class MLPipeline:

    def __init__(
        self,
        transformers: List[Transformer],
        estimator: Model = None
    ) -> None:
        # Define attributes
        self.transformers: List[Transformer] = transformers
        self.estimator: Model = estimator

        # Model attributes
        if self.estimator is not None:
            self.pipeline_id: str = self.estimator.model_id
            self.task: str = self.estimator.task

    @timing
    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        persist_datasets: bool = False,
        write_mode: str = None,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run transforer steps
        for transformer in self.transformers:
            # Run transform method
            X, y = transformer.transform(
                X=X, y=y, 
                debug=debug
            )

            # Show shapes
            if debug:
                LOGGER.debug('X.shape after %s: %s', transformer.class_name, X.shape)

            # Persist datasets
            if persist_datasets:
                # Persist X
                transformer.persist_dataset(
                    df=X,
                    df_name=f'X_{transformer.__class__.__name__}',
                    write_mode=write_mode,
                    mock=False
                )

                # Persist y
                transformer.persist_dataset(
                    df=y,
                    df_name=f'y_{transformer.__class__.__name__}',
                    write_mode=write_mode,
                    mock=False
                )

        return X, y

    @timing
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ignore_steps: List[str] = None,
        persist_datasets: bool = False,
        write_mode: str = None,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        # Run transforer steps
        for idx in range(len(self.transformers)):
            # Extract transformer
            transformer: Transformer = self.transformers[idx]

            if ignore_steps is None or transformer.class_name not in ignore_steps:
                # Run fit_transform method
                X, y = transformer.fit_transform(
                    X=X, y=y,
                    debug=debug
                )
            else:
                # Run transform method
                X, y = transformer.transform(
                    X=X, y=y,
                    debug=debug
                )

            # Show shapes
            if debug:
                LOGGER.debug('X.shape after %s: %s', transformer.class_name, X.shape)

            # Re-set transformer
            self.transformers[idx] = transformer

            # Persist datasets
            if persist_datasets:
                # Persist X
                transformer.persist_dataset(
                    df=X,
                    df_name=f'X_{transformer.__class__.__name__}',
                    write_mode=write_mode,
                    mock=False
                )

                # Persist y
                transformer.persist_dataset(
                    df=y,
                    df_name=f'y_{transformer.__class__.__name__}',
                    write_mode=write_mode,
                    mock=False
                )

        return X, y

    @timing
    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        # Apply data transformation pipeline
        X, _ = self.transform(
            X=X, y=None, 
            persist_datasets=False, 
            write_mode=None
        )

        # Predict new y
        y_pred = self.estimator.predict(X=X)

        return y_pred
    
    @timing
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        fit_transformers: bool = True,
        ignore_steps: List[str] = None
    ) -> None:
        if fit_transformers:
            # Run fit_transform method
            X_train, y_train = self.fit_transform(
                X=X_train, y=y_train,
                ignore_steps=ignore_steps,
                persist_datasets=False,
                write_mode=None
            )
        else:
            # Run transform method
            X_train, y_train = self.transform(
                X=X_train, y=y_train,
                persist_datasets=False,
                write_mode=None
            )

        # Fit the model
        if self.estimator is not None:
            self.estimator.fit(X=X_train, y=y_train)
    
    @timing
    def fit_predict(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        fit_transformers: bool = True
    ) -> np.ndarray:
        # Fit pipeline with X_train & y_train
        self.fit(
            X_train=X_train, 
            y_train=y_train,
            fit_transformers=fit_transformers
        )

        # Predict with y_test
        y_pred: np.ndarray = self.predict(X=X_test)

        return y_pred
    
    def evaluate(
        self,
        y_pred: np.ndarray,
        y_test: pd.DataFrame,
        cutoff: float = None
    ) -> None:
        self.estimator.evaluate_test(
            y_pred=y_pred,
            y_test=y_test,
            cutoff=cutoff
        )

    def extract_transformer(
        self,
        transformer_name: str
    ) -> Transformer:
        # Search for requested transformer
        for transformer in self.transformers:
            if transformer.__class__.__name__ == transformer_name:
                return transformer
        
        raise Exception(f'Transformer "{transformer_name}" was not find in {self.pipeline_id} MLPipeline.')

    def save(self) -> None:
        # Save transformers
        for transformer in self.transformers:
            # Save transformer
            transformer.save()

        # Save Model
        if self.estimator is not None:
            self.estimator.save()

    def load(self) -> None:
        # Load transformers
        for idx in range(len(self.transformers)):
            # Extract transformer
            transformer: Transformer = self.transformers[idx]

            # Load transformer
            try:
                transformer.load()
            except Exception as e:
                LOGGER.warning(
                    'Unable to load %s %s.\n'
                    'A base %s will be loaded.\n'
                    'Exception: %s', 
                    transformer.transformer_id, transformer.class_name, 
                    transformer.class_name, e
                )

                transformer.transformer_id = 'base'
                transformer.load()
                transformer.transformer_id = self.pipeline_id

            # Re-set transformer
            self.transformers[idx] = transformer
        
        # Load Model
        if self.estimator is not None:
            self.estimator.load(light=False)

