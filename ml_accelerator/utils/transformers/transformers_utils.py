from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.transformers.transformer import Transformer
from ml_accelerator.data_processing.transformers.data_cleaner import DataCleaner
from ml_accelerator.data_processing.transformers.feature_enricher import FeatureEnricher
from ml_accelerator.data_processing.transformers.data_standardizer import DataStandardizer
from ml_accelerator.data_processing.transformers.feature_selector import FeatureSelector

from typing import List


def load_transformers_list(transformer_id: str) -> List[Transformer]:
    # Define transformers
    transformers: List[Transformer] = [
        DataCleaner(transformer_id=transformer_id),
        FeatureEnricher(transformer_id=transformer_id),
        DataStandardizer(transformer_id=transformer_id),
        FeatureSelector(transformer_id=transformer_id)
    ]

    # Assert that list has expected steps
    for idx, expected_name in enumerate(Params.TRANSFORMERS_STEPS):
        # Extract transformer name
        actual_name: str = transformers[idx].class_name

        assert actual_name == expected_name

    return transformers