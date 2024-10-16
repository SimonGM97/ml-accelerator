from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.etl import ExtractTransformLoad
from unittest import TestCase
from unittest.mock import patch
from pandas.testing import assert_frame_equal
import pandas as pd


# conda deactivate
# source .ml_accel_venv/bin/activate
# python3 -m unittest test/test_data_processing/test_etl.py
class TestETL(TestCase):

    """
    Unit Tests
    """

    def test__ETL__success_data_persisted_and_loaded_correctly(self):
        # Instanciate ETL
        ETL: ExtractTransformLoad = ExtractTransformLoad()

        # Load & persist new data
        X, y = ETL.run_pipeline(
            persist_datasets=True,
            overwrite=True,
            mock_datasets=True
        )

        # Read persisted datasets
        persisted_X: pd.DataFrame = ETL.load_dataset(
            df_name='X_raw', 
            filters=None,
            mock=True
        )
        persisted_y: pd.DataFrame  = ETL.load_dataset(
            df_name='y_raw', 
            filters=None,
            mock=True
        )

        # Assert that X datasets are equal
        assert_frame_equal(X, persisted_X)

        # Assert that X datasets are equal
        assert_frame_equal(y, persisted_y)

