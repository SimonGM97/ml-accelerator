from ml_accelerator.config.params import Params
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

import warnings

# Suppress only UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


# Get logger
LOGGER = get_logger(name=__name__)


class Transformer(ABC, DataHelper):

    # Pickled attrs
    pickled_attrs = []

    def __init__(
        self,
        transformer_id: str = None
    ) -> None:
        # Instanciate parent classes
        super().__init__()

        # Set self.transformer_id
        self.transformer_id: str = transformer_id

    """
    Properties
    """

    @property
    def class_name(self) -> float:
        return self.__class__.__name__

    """
    Abstract methods - must be implemented by subclasses
    """
    
    @abstractmethod
    def transform(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame = None,
        debug: bool = False
    ) -> np.ndarray:
        pass

    @abstractmethod
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame = None,
        debug: bool = False
    ) -> None:
        pass
    
    @abstractmethod
    def diagnose(self) -> dict:
        pass

    """
    Non-abstract methods
    """
    
    def save(self) -> None:
        # Run self.save_transformer
        self.save_transformer()
    
    def load(self) -> None:
        # Run self.load_transformer
        self.load_transformer()

    """
    Other methods
    """

    def __repr__(self) -> str:
        return ''
        
