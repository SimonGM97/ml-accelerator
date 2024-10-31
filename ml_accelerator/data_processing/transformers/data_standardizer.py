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
        'num_cols',
        'cat_cols',
        'label_encoder',
        'num_scaler',
        'cat_ohe'
    ]

    def __init__(
        self,
        transformer_id: str = None,
        target_column: str = Params.TARGET_COLUMN,
        encode_target: bool = Params.ENCODE_TARGET_COLUMN,
        scale_num_features: bool = Params.SCALE_NUM_FEATURES,
        encode_cat_features: bool = Params.ENCODE_CAT_FEATURES
    ) -> None:
        # Instanciate parent classes
        super().__init__(transformer_id=transformer_id)

        # Set non-load attributes
        self.target_column: str = target_column
        self.encode_target: bool = encode_target
        self.scale_num_features: bool = scale_num_features
        self.encode_cat_features: bool = encode_cat_features

        # Set attributes to load
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
        y: pd.DataFrame = None,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.standardizer_pipeline
        X, y = self.standardizer_pipeline(
            X=X, y=y, 
            fit=False,
            debug=debug
        )
        
        return X, y

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Run self.standardizer_pipeline
        X, y = self.standardizer_pipeline(
            X=X, y=y, 
            fit=True,
            debug=debug
        )

        return X, y

    def diagnose(self) -> None:
        return None
    
    """
    Non-required methods
    """

    def standardizer_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        fit: bool = False,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Find self.num_cols & self.cat_cols
        if fit:
            self.find_num_cols(X=X)
            self.find_cat_cols(X=X)

        # Encode target
        if self.encode_target and y is not None:
            # Fit self.label_encoder
            if fit:
                self.fit_label_encoder(y=y)

            # Run target encoder
            y = self._encode_target(y=y)

        # Scale numerical features
        if self.scale_num_features and len(self.num_cols) > 0:
            # Fit self.num_scaler
            if fit:
                self.fit_num_scaler(X=X)

            # Run StandardScaler
            X = self._scale_num_features(X=X)

        # Encode categorical features
        if self.encode_cat_features and len(self.cat_cols) > 0:
            # Fit self.cat_ohe
            if fit:
                self.fit_ohe(X=X)
            
            # Run OneHotEncoder
            X = self._encode_cat_features(X=X)

        return X, y

    def fit_label_encoder(
        self,
        y: pd.DataFrame
    ) -> None:
        # Instanciate LabelEncoder
        self.label_encoder: LabelEncoder = LabelEncoder()
        
        # Fit self.label_encoder
        self.label_encoder.fit(y=y[self.target_column].values)

    def _encode_target(
        self,
        y: pd.DataFrame
    ) -> pd.DataFrame:
        # Apply self.label_encoder
        y[self.target_column] = self.label_encoder.transform(y=y[self.target_column].values)

        return y

    def find_num_cols(
        self,
        X: pd.DataFrame
    ) -> None:
        # Define self.num_cols
        self.num_cols: List[str] = list(X.select_dtypes(include=['number']).columns)

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

    def find_cat_cols(
        self,
        X: pd.DataFrame
    ) -> None:
        # Define self.cat_cols
        self.cat_cols: List[str] = list(X.select_dtypes(exclude=['number']).columns)

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
            self.cat_ohe.transform(X[self.cat_cols]).toarray(),
            columns=self.cat_ohe.get_feature_names_out(self.cat_cols),
            index=X.index
        )

        X = pd.concat([X[self.num_cols], X_cat], axis=1)

        return X

