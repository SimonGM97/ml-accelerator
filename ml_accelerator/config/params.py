import pandas as pd
import yaml
import multiprocessing
import subprocess
import os
from typing import List, Any


pd.options.display.max_rows = 500
pd.set_option("display.max_columns", None)


def get_value(key: str, nested_dict: dict) -> Any:
    """
    Recursively search for a key in a nested dictionary and return its value.

    Args:
    key (str): The key to search for.
    nested_dict (dict): The dictionary to search.

    Returns:
    The value corresponding to the key.

    Raises:
    KeyError: If the key is not found.
    """
    # Base case: if the current level has the key
    if key in nested_dict:
        return nested_dict[key]

    # Recursive case: search in the nested dictionaries
    for k, value in nested_dict.items():
        if isinstance(value, dict):  # Check if the value is another dictionary
            try:
                return get_value(key, value)  # Recursive search
            except KeyError:
                continue  # Key not found at this level, continue searching

    # If the key is not found in the entire dictionary
    raise KeyError(f"Key '{key}' not found in the nested dictionary.")


class Params:
    initialized: bool = False

    """
    GENERAL PARAMETERS
    """
    # PROJECT PARAMETERS
    PROJECT_NAME: str
    VERSION: str
    TARGET_COLUMN: str
    TASK: str

    """
    DATA PROCESSING PARAMETERS
    """
    # EXTRACT TRANSFORM LOAD PARAMETERS
    ETL_SOURCE: str

    # STORAGE PARAMETERS
    DATASET_NAME: str    
    DATA_EXTENTION: str
    PARTITION_COLUMNS: str

    # DATA CLEANING PARAMETERS
    OUTLIER_Z_THRESHOLD: float

    # FEATURE ENRICHER PARAMETERS
    ADD_OUTLIER_FEATURES: bool
    OUTLIER_FEATURES_Z: float
    ADD_FIBONACHI_FEATURES: bool
    ADD_DERIVATIVE_FEATURES: bool
    ADD_LAG_FEATURES: bool
    LAGS: List[int]
    ADD_ROLLING_FEATURES: bool
    ADD_EMA_FEATURES: bool
    ROLLING_WINDOWS: List[int]
    ADD_TEMPORAL_EMBEDDING_FEATURES: bool
    DATETIME_COL: str
    HOLIDAY_COUNTRY: str

    # DATA TRANSFORMING PARAMETERS
    ENCODE_TARGET_COLUMN: bool
    SCALE_NUM_FEATURES: bool
    ENCODE_CAT_FEATURES: bool

    # FEATURE SELECTION PARAMETERS
    FORCED_FEATURES: List[str]
    IGNORE_FEATURES_P_VALUE: float
    BORUTA_ALGORITHM: str
    RFE_N: int
    K_BEST: int
    TSFRESH_P_VALUE: float
    TSFRESH_N: int
    MAX_FEATURES: int

    """
    MODELING PARAMETERS
    """
    # ML DATASETS PARAMETERS
    TEST_SIZE: float

    # CLASSIFICATION PARAMETERS
    BALANCE_TRAIN: bool
    BALANCE_METHOD: str
    CLASS_WEIGHT: dict
    CUTOFF: float

    # REGRESSION PARAMETERS

    # FORECASTING PARAMETERS

    # HYPER PARAMETER TUNING
    ALGORITHMS: List[str]
    SEARCH_SPACE: dict
    N_CANDIDATES: int
    OPTIMIZATION_METRIC: str
    VAL_SPLITS: int
    MAX_EVALS: int
    LOSS_THRESHOLD: float
    TIMEOUT_MINS: float
    MIN_PERFORMANCE: float

    # FEATURE IMPORTANCE PARAMETERS
    IMPORTANCE_METHOD: str

    """
    MLPIPELINE PARAMETERS
    """

    TRANSFORMERS_STEPS: List[str]

    """
    WORKFLOW PARAMETERS
    """
    # MODEL BUILDING
    MODEL_BUILDING_PARAMS: dict
    STEP_FUNCTION_STATES: List[dict]

    """
    OTHER PARAMETERS
    """
    # LOG PARAMETERS
    LEVEL: str
    TXT_FMT: str
    JSON_FMT: str
    FILTER_LVLS: str
    LOG_FILE: str
    BACKUP_COUNT: int

    # COMPUTE PARAMETERS
    GPUS: str
    CPUS: str

    @classmethod
    def load(cls) -> None:
        if cls.initialized:
            return
        
        # Load config file
        with open(os.path.join("config", "config.yaml")) as file:
            config: dict = yaml.load(file, Loader=yaml.FullLoader)

        # Load parameters
        load_parameters: List[str] = list(cls.__annotations__.keys())
        ignore_parameters: List[str] = ["load", "initialized", "GPUS", "CPUS"]

        for param_name in load_parameters:
            if param_name not in ignore_parameters and not param_name.startswith("__"):
                setattr(cls, param_name, get_value(param_name, config))
        
        # Compute parameters
        def get_gpu_count():
            cmd = "system_profiler SPDisplaysDataType | grep Chipset"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8")
            return len(output.split("\n"))

        cls.GPUS = 1 # get_gpu_count()
        cls.CPUS = multiprocessing.cpu_count()

        cls.initialized = True


# .ml_accel_venv/bin/python ml_accelerator/config/params.py
if not Params.initialized:
    Params.load()