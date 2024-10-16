import pandas as pd
import yaml
from pathlib import Path
# from git.repo.base import Repo
import multiprocessing
import subprocess
import os
from typing import List


pd.options.display.max_rows = 500
pd.set_option("display.max_columns", None)


# def find_base_repo_root(project_name: str) -> Path:
#     base_path = os.path.dirname(os.path.abspath(__file__))
#     if project_name in base_path:
#         base_path = base_path[:base_path.find(project_name)]
    
#     try:
#         base_path = Path(Repo(base_path, search_parent_directories=True).working_tree_dir or base_path)
#     except Exception as e:
#         # print(f"Unable to load base_path.\n"
#         #       f"Exception: {e}\n\n")
#         # except InvalidGitRepositoryError:
#         base_path = Path(base_path)

#     return base_path / project_name


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

    # ENVIRONMENT PARAMETERS
    ENV: str
    REGION: str
    BUCKET: str
    # CWD: Path

    DATA_STORAGE_ENV: str
    MODEL_STORAGE_ENV: str
    COMPUTE_ENV: str
    DOCKER_REPOSITORY: str

    """
    DATA PROCESSING PARAMETERS
    """
    # EXTRACT TRANSFORM LOAD PARAMETERS
    ETL_SOURCE: str

    # STORAGE PARAMETERS
    DATASET_NAME: str
    TRAINING_PATH: List[str]
    INFERENCE_PATH: List[str]
    TRANSFORMERS_PATH: List[str]
    MODELS_PATH: List[str]
    SCHEMAS_PATH: List[str]
    MOCK_PATH: List[str]
    
    DATA_EXTENTION: str
    PARTITION_COLUMNS: str

    # DATA CLEANING PARAMETERS
    Z_THRESHOLD: float

    # FEATURE ENGINEERING PARAMETERS

    # DATA TRANSFORMING PARAMETERS
    ENCODE_TARGET: bool

    # FEATURE SELECTION PARAMETERS
    N_FEATURES: int

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
    WORKFLOW PARAMETERS
    """
    # MODEL BUILDING
    FIT_TRANSFORMERS: bool
    SAVE_TRANSFORMERS: bool
    PERSIST_DATASETS: bool
    WRITE_MODE: str

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

    # DEPLOYMENT PARAMETERS

    # INFRASTRUCTURE PARAMETERS

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

        """
        GENERAL PARAMETERS
        """
        # PROJECT PARAMETERS
        PROJECT_PARAMS: dict = config.get("PROJECT_PARAMS")

        cls.VERSION: str = PROJECT_PARAMS.get("VERSION")
        cls.PROJECT_NAME: str = PROJECT_PARAMS.get("PROJECT_NAME")
        cls.TARGET: str = PROJECT_PARAMS.get("TARGET")
        cls.TASK: str = PROJECT_PARAMS.get("TASK")

        # ENVIRONMENT PARAMETERS
        ENV_PARAMS: dict = config.get("ENV_PARAMS")

        cls.ENV: str = ENV_PARAMS.get("ENV")
        cls.REGION: str = ENV_PARAMS.get("REGION")
        cls.BUCKET: str = ENV_PARAMS.get("BUCKET")
        # cls.CWD = find_base_repo_root(project_name=cls.PROJECT_NAME)

        cls.DATA_STORAGE_ENV: str = ENV_PARAMS.get("DATA_STORAGE_ENV")
        cls.MODEL_STORAGE_ENV: str = ENV_PARAMS.get("MODEL_STORAGE_ENV")
        cls.COMPUTE_ENV: str = ENV_PARAMS.get("COMPUTE_ENV")
        cls.DOCKER_REPOSITORY: str = ENV_PARAMS.get("DOCKER_REPOSITORY")

        """
        DATA PARAMETERS
        """
        # EXTRACT TRANSFORM LOAD PARAMETERS
        ETL_PARAMS: dict = config.get("ETL_PARAMS")

        cls.ETL_SOURCE: str = ETL_PARAMS.get("ETL_SOURCE")

        # STORAGE PARAMETERS
        STORAGE_PARAMS: dict = config.get("STORAGE_PARAMS")
        PATHS_PARAMS: dict = STORAGE_PARAMS.get("PATHS_PARAMS")

        cls.DATASET_NAME: str = STORAGE_PARAMS.get("DATASET_NAME")
        cls.TRAINING_PATH: List[str] = PATHS_PARAMS.get("TRAINING_PATH")
        cls.INFERENCE_PATH: List[str] = PATHS_PARAMS.get("INFERENCE_PATH")
        cls.TRANSFORMERS_PATH: List[str] = PATHS_PARAMS.get("TRANSFORMERS_PATH")
        cls.MODELS_PATH: List[str] = PATHS_PARAMS.get("MODELS_PATH")
        cls.SCHEMAS_PATH: List[str] = PATHS_PARAMS.get("SCHEMAS_PATH")
        cls.MOCK_PATH: List[str] = PATHS_PARAMS.get("MOCK_PATH")
        cls.DATA_EXTENTION: str = STORAGE_PARAMS.get("DATA_EXTENTION")
        cls.PARTITION_COLUMNS: str = STORAGE_PARAMS.get("PARTITION_COLUMNS")

        # DATA CLEANING PARAMETERS
        DATA_CLEANING_PARAMS: dict = config.get("DATA_CLEANING_PARAMS")

        cls.OUTLIER_Z_THRESHOLD: float = DATA_CLEANING_PARAMS.get("OUTLIER_Z_THRESHOLD")

        # FEATURE ENGINEERING PARAMETERS
        FEATURE_ENGINEERING_PARAMS: dict = config.get("FEATURE_ENGINEERING_PARAMS")

        # DATA TRANSFORMING PARAMETERS
        DATA_TRANSFORMING_PARAMS: dict = config.get("DATA_TRANSFORMING_PARAMS")

        cls.ENCODE_TARGET: bool = DATA_TRANSFORMING_PARAMS.get("ENCODE_TARGET")
        cls.SCALE_NUM_FEATURES: bool = DATA_TRANSFORMING_PARAMS.get("SCALE_NUM_FEATURES")
        cls.ENCODE_CAT_FEATURES: bool = DATA_TRANSFORMING_PARAMS.get("ENCODE_CAT_FEATURES")
        
        # FEATURE SELECTION PARAMETERS
        FEATURE_SELECTION_PARAMS: dict = config.get("FEATURE_SELECTION_PARAMS")

        cls.N_FEATURES: list = FEATURE_SELECTION_PARAMS.get("N_FEATURES")

        """
        MODELING PARAMETERS
        """
        # ML DATASETS PARAMETERS
        ML_DATASETS_PARAMS: dict = config.get("ML_DATASETS_PARAMS")

        cls.TEST_SIZE: float = ML_DATASETS_PARAMS.get("TEST_SIZE")

        # CLASSIFICATION PARAMETERS
        CLASSIFICATION_PARAMS: dict = config.get("CLASSIFICATION_PARAMS")

        cls.BALANCE_TRAIN: bool = CLASSIFICATION_PARAMS.get("BALANCE_TRAIN")
        cls.BALANCE_METHOD: str = CLASSIFICATION_PARAMS.get("BALANCE_METHOD")
        cls.CLASS_WEIGHT: dict = CLASSIFICATION_PARAMS.get("CLASS_WEIGHT")
        cls.CUTOFF: float = CLASSIFICATION_PARAMS.get("CUTOFF")

        # REGRESSION PARAMETERS
        REGRESSION_PARAMS: dict = config.get("REGRESSION_PARAMS")

        # FORECASTING PARAMETERS
        FORECASTING_PARAMS: dict = config.get("FORECASTING_PARAMS")

        # HYPER PARAMETER TUNING PARAMETERS
        HYPER_PARAMETER_TUNING_PARAMS: dict = config.get("HYPER_PARAMETER_TUNING_PARAMS")

        cls.ALGORITHMS: List[str] = HYPER_PARAMETER_TUNING_PARAMS.get("ALGORITHMS")
        cls.SEARCH_SPACE: dict = HYPER_PARAMETER_TUNING_PARAMS.get("SEARCH_SPACE")
        cls.N_CANDIDATES: int = HYPER_PARAMETER_TUNING_PARAMS.get("N_CANDIDATES")
        cls.OPTIMIZATION_METRIC: str = HYPER_PARAMETER_TUNING_PARAMS.get("OPTIMIZATION_METRIC")
        cls.VAL_SPLITS: int = HYPER_PARAMETER_TUNING_PARAMS.get("VAL_SPLITS")
        cls.MAX_EVALS: int = HYPER_PARAMETER_TUNING_PARAMS.get("MAX_EVALS")
        cls.LOSS_THRESHOLD: float = HYPER_PARAMETER_TUNING_PARAMS.get("LOSS_THRESHOLD")
        cls.TIMEOUT_MINS: float = HYPER_PARAMETER_TUNING_PARAMS.get("TIMEOUT_MINS")
        cls.MIN_PERFORMANCE: float = HYPER_PARAMETER_TUNING_PARAMS.get("MIN_PERFORMANCE")

        # FEATURE IMPORTANCE PARAMETERS
        FEATURE_IMPORTANCE_PARAMS: dict = config.get("FEATURE_IMPORTANCE_PARAMS")

        cls.IMPORTANCE_METHOD: str = FEATURE_IMPORTANCE_PARAMS.get("IMPORTANCE_METHOD")

        """
        WORKFLOW PARAMETERS
        """
        # MODEL BUILDING
        MODEL_BUILDING_PARAMS: dict = config.get("MODEL_BUILDING_PARAMS")

        # Data Processing
        cls.FIT_TRANSFORMERS: bool = MODEL_BUILDING_PARAMS.get("FIT_TRANSFORMERS")
        cls.SAVE_TRANSFORMERS: bool = MODEL_BUILDING_PARAMS.get("SAVE_TRANSFORMERS")
        cls.PERSIST_DATASETS: bool = MODEL_BUILDING_PARAMS.get("PERSIST_DATASETS")
        cls.WRITE_MODE: str = MODEL_BUILDING_PARAMS.get("WRITE_MODE")

        """
        OTHER PARAMETERS
        """
        # LOG PARAMETERS
        LOG_PARAMS: dict = config.get("LOG_PARAMS")

        cls.LEVEL: str = LOG_PARAMS.get("LEVEL")
        cls.TXT_FMT: str = LOG_PARAMS.get("TXT_FMT")
        cls.JSON_FMT: str = LOG_PARAMS.get("JSON_FMT")
        cls.FILTER_LVLS: str = LOG_PARAMS.get("FILTER_LVLS")
        cls.LOG_FILE: str = LOG_PARAMS.get("LOG_FILE")
        cls.BACKUP_COUNT: int = LOG_PARAMS.get("BACKUP_COUNT")

        # DEPLOYMENT PARAMETERS
        DEPLOYMENT_PARAMS: dict = config.get("DEPLOYMENT_PARAMS")

        # INFRASTRUCTURE PARAMETERS
        INFRASTRUCTURE_PARAMS: dict = config.get("INFRASTRUCTURE_PARAMS")
        
        # COMPUTE PARAMETERS
        def get_gpu_count():
            cmd = "system_profiler SPDisplaysDataType | grep Chipset"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8")
            return len(output.split("\n"))

        cls.GPUS = 1 # get_gpu_count()
        cls.CPUS = multiprocessing.cpu_count()

        cls.initialized = True


if not Params.initialized:
    # print("Initializing Params.\n")
    Params.load()