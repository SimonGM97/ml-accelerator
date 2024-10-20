from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.transformers.transformer import Transformer
from ml_accelerator.utils.logging.logger_helper import get_logger

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from typing import List, Tuple


# Get logger
LOGGER = get_logger(name=__name__)


class DataStandardizer(Transformer):

    # Pickled attrs
    pickled_attrs = [
        'label_encoder',
        'num_scaler',
        'cat_ohe'
    ]

    def __init__(
        self,
        transformer_id: str = None,
        target: str = Params.TARGET,
        encode_target: bool = Params.ENCODE_TARGET,
        scale_num_features: bool = Params.SCALE_NUM_FEATURES,
        encode_cat_features: bool = Params.ENCODE_CAT_FEATURES
    ) -> None:
        # Instanciate parent classes
        super().__init__(transformer_id=transformer_id)

        # Set other attributes
        self.target: str = target
        self.encode_target: bool = encode_target
        self.scale_num_features: bool = scale_num_features
        self.encode_cat_features: bool = encode_cat_features

        # Set default attributes
        self.num_cols: List[str] = None
        self.cat_cols: List[str] = None
        
        self.label_encoder: LabelEncoder = None
        self.num_scaler: StandardScaler = None
        self.cat_ohe: OneHotEncoder = None

    """
    Required methods (from Transformer abstract methods)
    """

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.transformer_pipeline
        X, y = self.transformer_pipeline(X=X, y=y, fit=False)
        
        return X, y

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.transformer_pipeline
        X, y = self.transformer_pipeline(X=X, y=y, fit=True)

        return X, y

    def diagnose(self) -> None:
        return None
    
    """
    Non-required methods
    """

    def transformer_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        fit: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Find column attributes
        self.find_column_attrs(X=X)

        if self.encode_target and y is not None:
            # Fit self.label_encoder
            if fit:
                self.fit_label_encoder(y=y)

            # Encode target
            y = self._encode_target(y=y)

        if self.scale_num_features and len(self.num_cols) > 0:
            # Fit self.num_scaler
            if fit:
                self.fit_num_scaler(X=X)

            # Scale numerical features
            X = self._scale_num_features(X=X)

        if self.encode_cat_features and len(self.cat_cols) > 0:
            # Fit self.cat_ohe
            if fit:
                self.fit_ohe(X=X)
            
            # Encode categorical features
            X = self._encode_cat_features(X=X)

        return X, y

    def find_column_attrs(
        self,
        X: pd.DataFrame
    ) -> None:
        # Define attributes
        self.num_cols: List[str] = list(X.select_dtypes(include=['number']).columns)
        self.cat_cols: List[str] = list(X.select_dtypes(exclude=['number']).columns)

    def fit_label_encoder(
        self,
        y: pd.DataFrame
    ) -> None:
        # Instanciate LabelEncoder
        self.label_encoder: LabelEncoder = LabelEncoder()
        
        # Fit self.label_encoder
        self.label_encoder.fit(y=y[self.target].values)

    def _encode_target(
        self,
        y: pd.DataFrame
    ) -> pd.DataFrame:
        # Apply self.label_encoder
        y[self.target] = self.label_encoder.transform(y=y[self.target].values)

        return y

    def fit_num_scaler(
        self,
        X: pd.DataFrame
    ) -> None:
        # Instanciate StandardScaler
        self.num_scaler: StandardScaler = StandardScaler(
            with_mean=True, 
            with_std=True
        )

        # Fit self.num_scaler on numerical features
        self.num_scaler.fit(X[self.num_cols])

    def _scale_num_features(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # Apply self.num_scaler
        X[self.num_cols] = self.num_scaler.transform(X=X[self.num_cols])

        return X

    def fit_ohe(
        self,
        X: pd.DataFrame
    ) -> None:
        # Instanciate OneHotEncoder
        self.cat_ohe: OneHotEncoder = OneHotEncoder(handle_unknown='ignore')

        # Fit self.cat_ohe on categorical features
        self.cat_ohe.fit(X[self.cat_cols])

    def _encode_cat_features(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # Apply self.cat_ohe
        X_cat = pd.DataFrame(
            self.cat_ohe.transform(X[self.cat_ohe]),
            columns=self.cat_ohe.get_feature_names_out(self.cat_ohe),
            index=X.index
        )

        X = pd.concat([X[self.num_cols], X_cat], axis=1)

        return X

