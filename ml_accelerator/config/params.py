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
    VERSION: str
    PROJECT_NAME: str
    TARGET: str
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
    ENCODE_TARGET: bool
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
    FIT_TRANSFORMERS: bool
    SAVE_TRANSFORMERS: bool
    PERSIST_DATASETS: bool
    WRITE_MODE: str

    TRAIN_PROD_PIPE: bool
    TRAIN_STAGING_PIPES: bool
    TRAIN_DEV_PIPES: bool

    EVALUATE_PROD_PIPE: bool
    EVALUATE_STAGING_PIPES: bool
    EVALUATE_DEV_PIPES: bool
    UPDATE_MODEL_STAGES: bool
    UPDATE_PROD_MODEL: bool

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

# """
# GENERAL PARAMETERS
# """
# # PROJECT PARAMETERS
# PROJECT_PARAMS: dict = config.get("PROJECT_PARAMS")

# cls.VERSION: str = PROJECT_PARAMS.get("VERSION")
# cls.PROJECT_NAME: str = PROJECT_PARAMS.get("PROJECT_NAME")
# cls.TARGET: str = PROJECT_PARAMS.get("TARGET")
# cls.TASK: str = PROJECT_PARAMS.get("TASK")

# """
# DATA PARAMETERS
# """
# # EXTRACT TRANSFORM LOAD PARAMETERS
# ETL_PARAMS: dict = config.get("ETL_PARAMS")

# cls.ETL_SOURCE: str = ETL_PARAMS.get("ETL_SOURCE")

# # STORAGE PARAMETERS
# STORAGE_PARAMS: dict = config.get("STORAGE_PARAMS")

# cls.DATASET_NAME: str = STORAGE_PARAMS.get("DATASET_NAME")
# cls.DATA_EXTENTION: str = STORAGE_PARAMS.get("DATA_EXTENTION")
# cls.PARTITION_COLUMNS: str = STORAGE_PARAMS.get("PARTITION_COLUMNS")

# # DATA CLEANING PARAMETERS
# DATA_CLEANING_PARAMS: dict = config.get("DATA_CLEANING_PARAMS")

# cls.OUTLIER_Z_THRESHOLD: float = DATA_CLEANING_PARAMS.get("OUTLIER_Z_THRESHOLD")

# # FEATURE ENRICHER PARAMETERS
# FEATURE_ENRICHER_PARAMS: dict = config.get("FEATURE_ENRICHER_PARAMS")

# cls.ADD_OUTLIER_FEATURES: bool = FEATURE_ENRICHER_PARAMS.get("ADD_OUTLIER_FEATURES")
# cls.OUTLIER_FEATURES_Z: float = FEATURE_ENRICHER_PARAMS.get("OUTLIER_FEATURES_Z")
# cls.ADD_FIBONACHI_FEATURES: bool = FEATURE_ENRICHER_PARAMS.get("ADD_FIBONACHI_FEATURES")
# cls.ADD_DERIVATIVE_FEATURES: bool = FEATURE_ENRICHER_PARAMS.get("ADD_DERIVATIVE_FEATURES")
# cls.ADD_LAG_FEATURES: bool = FEATURE_ENRICHER_PARAMS.get("ADD_LAG_FEATURES")
# cls.LAGS: List[int] = FEATURE_ENRICHER_PARAMS.get("LAGS")
# cls.ADD_ROLLING_FEATURES: bool = FEATURE_ENRICHER_PARAMS.get("ADD_ROLLING_FEATURES")
# cls.ADD_EMA_FEATURES: bool = FEATURE_ENRICHER_PARAMS.get("ADD_EMA_FEATURES")
# cls.ROLLING_WINDOWS: List[int] = FEATURE_ENRICHER_PARAMS.get("ROLLING_WINDOWS")
# cls.ADD_TEMPORAL_EMBEDDING_FEATURES: bool = FEATURE_ENRICHER_PARAMS.get("ADD_TEMPORAL_EMBEDDING_FEATURES")
# cls.DATETIME_COL: str = FEATURE_ENRICHER_PARAMS.get("DATETIME_COL")
# cls.HOLIDAY_COUNTRY: str = FEATURE_ENRICHER_PARAMS.get("HOLIDAY_COUNTRY")

# # DATA TRANSFORMING PARAMETERS
# DATA_TRANSFORMING_PARAMS: dict = config.get("DATA_TRANSFORMING_PARAMS")

# cls.ENCODE_TARGET: bool = DATA_TRANSFORMING_PARAMS.get("ENCODE_TARGET")
# cls.SCALE_NUM_FEATURES: bool = DATA_TRANSFORMING_PARAMS.get("SCALE_NUM_FEATURES")
# cls.ENCODE_CAT_FEATURES: bool = DATA_TRANSFORMING_PARAMS.get("ENCODE_CAT_FEATURES")

# # FEATURE SELECTION PARAMETERS
# FEATURE_SELECTION_PARAMS: dict = config.get("FEATURE_SELECTION_PARAMS")

# cls.FORCED_FEATURES: List[str] = FEATURE_SELECTION_PARAMS.get("FORCED_FEATURES")
# cls.IGNORE_FEATURES_P_VALUE: float = FEATURE_SELECTION_PARAMS.get("IGNORE_FEATURES_P_VALUE")
# cls.BORUTA_ALGORITHM: str = FEATURE_SELECTION_PARAMS.get("BORUTA_ALGORITHM")
# cls.RFE_N: int = FEATURE_SELECTION_PARAMS.get("RFE_N")
# cls.K_BEST: int = FEATURE_SELECTION_PARAMS.get("K_BEST")
# cls.TSFRESH_P_VALUE: float = FEATURE_SELECTION_PARAMS.get("TSFRESH_P_VALUE")
# cls.TSFRESH_N: int = FEATURE_SELECTION_PARAMS.get("TSFRESH_N")
# cls.MAX_FEATURES: int = FEATURE_SELECTION_PARAMS.get("MAX_FEATURES")

# """
# MODELING PARAMETERS
# """
# # ML DATASETS PARAMETERS
# ML_DATASETS_PARAMS: dict = config.get("ML_DATASETS_PARAMS")

# cls.TEST_SIZE: float = ML_DATASETS_PARAMS.get("TEST_SIZE")

# # CLASSIFICATION PARAMETERS
# CLASSIFICATION_PARAMS: dict = config.get("CLASSIFICATION_PARAMS")

# cls.BALANCE_TRAIN: bool = CLASSIFICATION_PARAMS.get("BALANCE_TRAIN")
# cls.BALANCE_METHOD: str = CLASSIFICATION_PARAMS.get("BALANCE_METHOD")
# cls.CLASS_WEIGHT: dict = CLASSIFICATION_PARAMS.get("CLASS_WEIGHT")
# cls.CUTOFF: float = CLASSIFICATION_PARAMS.get("CUTOFF")

# # REGRESSION PARAMETERS
# REGRESSION_PARAMS: dict = config.get("REGRESSION_PARAMS")

# # FORECASTING PARAMETERS
# FORECASTING_PARAMS: dict = config.get("FORECASTING_PARAMS")

# # HYPER PARAMETER TUNING PARAMETERS
# HYPER_PARAMETER_TUNING_PARAMS: dict = config.get("HYPER_PARAMETER_TUNING_PARAMS")

# cls.ALGORITHMS: List[str] = HYPER_PARAMETER_TUNING_PARAMS.get("ALGORITHMS")
# cls.SEARCH_SPACE: dict = HYPER_PARAMETER_TUNING_PARAMS.get("SEARCH_SPACE")
# cls.N_CANDIDATES: int = HYPER_PARAMETER_TUNING_PARAMS.get("N_CANDIDATES")
# cls.OPTIMIZATION_METRIC: str = HYPER_PARAMETER_TUNING_PARAMS.get("OPTIMIZATION_METRIC")
# cls.VAL_SPLITS: int = HYPER_PARAMETER_TUNING_PARAMS.get("VAL_SPLITS")
# cls.MAX_EVALS: int = HYPER_PARAMETER_TUNING_PARAMS.get("MAX_EVALS")
# cls.LOSS_THRESHOLD: float = HYPER_PARAMETER_TUNING_PARAMS.get("LOSS_THRESHOLD")
# cls.TIMEOUT_MINS: float = HYPER_PARAMETER_TUNING_PARAMS.get("TIMEOUT_MINS")
# cls.MIN_PERFORMANCE: float = HYPER_PARAMETER_TUNING_PARAMS.get("MIN_PERFORMANCE")

# # FEATURE IMPORTANCE PARAMETERS
# FEATURE_IMPORTANCE_PARAMS: dict = config.get("FEATURE_IMPORTANCE_PARAMS")

# cls.IMPORTANCE_METHOD: str = FEATURE_IMPORTANCE_PARAMS.get("IMPORTANCE_METHOD")

# """
# MLPIPELINE PARAMETERS
# """
# MLPIPELINE_PARAMS: dict = config.get("MLPIPELINE_PARAMS")

# cls.TRANSFORMERS_STEPS: List[str] = MLPIPELINE_PARAMS.get("TRANSFORMERS_STEPS")

# """
# WORKFLOW PARAMETERS
# """
# # MODEL BUILDING
# MODEL_BUILDING_PARAMS: dict = config.get("MODEL_BUILDING_PARAMS")

# # Data Processing
# cls.FIT_TRANSFORMERS: bool = MODEL_BUILDING_PARAMS.get("FIT_TRANSFORMERS")
# cls.SAVE_TRANSFORMERS: bool = MODEL_BUILDING_PARAMS.get("SAVE_TRANSFORMERS")
# cls.PERSIST_DATASETS: bool = MODEL_BUILDING_PARAMS.get("PERSIST_DATASETS")
# cls.WRITE_MODE: str = MODEL_BUILDING_PARAMS.get("WRITE_MODE")

# # Training
# cls.TRAIN_PROD_PIPE: bool = MODEL_BUILDING_PARAMS.get("TRAIN_PROD_PIPE")
# cls.TRAIN_STAGING_PIPES: bool = MODEL_BUILDING_PARAMS.get("TRAIN_STAGING_PIPES")
# cls.TRAIN_DEV_PIPES: bool = MODEL_BUILDING_PARAMS.get("TRAIN_DEV_PIPES")

# # Evaluating
# cls.EVALUATE_PROD_PIPE: bool = MODEL_BUILDING_PARAMS.get("EVALUATE_PROD_PIPE")
# cls.EVALUATE_STAGING_PIPES: bool = MODEL_BUILDING_PARAMS.get("EVALUATE_STAGING_PIPES")
# cls.EVALUATE_DEV_PIPES: bool = MODEL_BUILDING_PARAMS.get("EVALUATE_DEV_PIPES")
# cls.UPDATE_MODEL_STAGES: bool = MODEL_BUILDING_PARAMS.get("UPDATE_MODEL_STAGES")
# cls.UPDATE_PROD_MODEL: bool = MODEL_BUILDING_PARAMS.get("UPDATE_PROD_MODEL")

# """
# OTHER PARAMETERS
# """
# # LOG PARAMETERS
# LOG_PARAMS: dict = config.get("LOG_PARAMS")

# cls.LEVEL: str = LOG_PARAMS.get("LEVEL")
# cls.TXT_FMT: str = LOG_PARAMS.get("TXT_FMT")
# cls.JSON_FMT: str = LOG_PARAMS.get("JSON_FMT")
# cls.FILTER_LVLS: str = LOG_PARAMS.get("FILTER_LVLS")
# cls.LOG_FILE: str = LOG_PARAMS.get("LOG_FILE")
# cls.BACKUP_COUNT: int = LOG_PARAMS.get("BACKUP_COUNT")

# # INFRASTRUCTURE PARAMETERS
# INFRASTRUCTURE_PARAMS: dict = config.get("INFRASTRUCTURE_PARAMS")