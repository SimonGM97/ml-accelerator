import pandas as pd
import yaml
from pathlib import Path
from git.repo.base import Repo
import multiprocessing
import subprocess
import os
from typing import List


pd.options.display.max_rows = 500
pd.set_option("display.max_columns", None)


def find_base_repo_root(project_name: str) -> Path:
    base_path = os.path.dirname(os.path.abspath(__file__))
    if project_name in base_path:
        base_path = base_path[:base_path.find(project_name)]
    
    try:
        base_path = Path(Repo(base_path, search_parent_directories=True).working_tree_dir or base_path)
    except Exception as e:
        # print(f"Unable to load base_path.\n"
        #       f"Exception: {e}\n\n")
        # except InvalidGitRepositoryError:
        base_path = Path(base_path)

    return base_path / project_name


class Params:
    initialized: bool = False

    # GENERAL PARAMETERS
    PROJECT_NAME: str

    # ENVIRONMENT PARAMETERS
    ENV: str
    BUCKET: str
    CWD: Path

    """
    DATA PARAMETERS
    """

    # DATASET PATHS PARAMETERS
    TRAINING_PATH: List[str]
    INFERENCE_PATH: List[str]

    # DATA CLEANING PARAMETERS
    DATETIME_COLS: List[str]
    NON_NEG_COLS: List[str]

    # FEATURE ENGINEERING PARAMETERS

    # DATA TRANSFORMING PARAMETERS

    # FEATURE SELECTION PARAMETERS
    N_FEATURES: int

    """
    MODELING PARAMETERS
    """

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
    OPTIMIZATION_METRIC: str
    LOSS_THRESHOLD: float
    TIMEOUT_MINS: float
    MIN_PERFORMANCE: float

    """
    OTHERS
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

        """
        GENERAL PARAMETERS
        """
        GENERAL_PARAMS: dict = config.get("GENERAL_PARAMS")

        cls.PROJECT_NAME: str = GENERAL_PARAMS.get("PROJECT_NAME")
        cls.TARGET: str = GENERAL_PARAMS.get("TARGET")

        """
        ENVIRONMENT PARAMETERS
        """
        ENV_PARAMS: dict = config.get("ENV_PARAMS")

        cls.ENV: str = ENV_PARAMS.get("ENV")
        cls.BUCKET: str = ENV_PARAMS.get("BUCKET")
        cls.CWD = find_base_repo_root(project_name=cls.PROJECT_NAME)

        """
        DATASET PATHS PARAMETERS
        """
        DATASET_PATHS_PARAMS: dict = config.get("DATASET_PATHS_PARAMS")

        cls.TRAINING_PATH: List[str] = DATASET_PATHS_PARAMS.get("TRAINING_PATH")
        cls.INFERENCE_PATH: List[str] = DATASET_PATHS_PARAMS.get("INFERENCE_PATH")

        """
        DATA CLEANING PARAMETERS
        """
        DATA_CLEANING_PARAMS: dict = config.get("DATA_CLEANING_PARAMS")

        cls.DATETIME_COLS: List[str] = DATA_CLEANING_PARAMS.get("DATETIME_COLS")
        cls.NON_NEG_COLS: List[str] = DATA_CLEANING_PARAMS.get("NON_NEG_COLS")

        """
        FEATURE ENGINEERING PARAMETERS
        """
        FEATURE_ENGINEERING_PARAMS: dict = config.get("FEATURE_ENGINEERING_PARAMS")

        """
        DATA TRANSFORMING PARAMETERS
        """
        DATA_TRANSFORMING: dict = config.get("DATA_TRANSFORMING")
        
        """
        FEATURE SELECTION PARAMETERS
        """
        FEATURE_SELECTION_PARAMS: dict = config.get("FEATURE_SELECTION_PARAMS")

        cls.N_FEATURES: list = FEATURE_SELECTION_PARAMS.get("N_FEATURES")

        """
        CLASSIFICATION PARAMETERS
        """
        CLASSIFICATION_PARAMS: dict = config.get("CLASSIFICATION_PARAMS")

        cls.BALANCE_TRAIN: bool = CLASSIFICATION_PARAMS.get("BALANCE_TRAIN")
        cls.BALANCE_METHOD: str = CLASSIFICATION_PARAMS.get("BALANCE_METHOD")
        cls.CLASS_WEIGHT: dict = CLASSIFICATION_PARAMS.get("CLASS_WEIGHT")
        cls.CUTOFF: float = CLASSIFICATION_PARAMS.get("CUTOFF")

        """
        REGRESSION PARAMETERS
        """
        REGRESSION_PARAMS: dict = config.get("REGRESSION_PARAMS")

        """
        FORECASTING PARAMETERS
        """
        FORECASTING_PARAMS: dict = config.get("FORECASTING_PARAMS")

        """
        HYPER PARAMETER TUNING PARAMETERS
        """
        HYPER_PARAMETER_TUNING_PARAMS: dict = config.get("HYPER_PARAMETER_TUNING_PARAMS")

        cls.ALGORITHMS: List[str] = HYPER_PARAMETER_TUNING_PARAMS.get("ALGORITHMS")
        cls.SEARCH_SPACE: dict = HYPER_PARAMETER_TUNING_PARAMS.get("SEARCH_SPACE")
        cls.OPTIMIZATION_METRIC: str = HYPER_PARAMETER_TUNING_PARAMS.get("OPTIMIZATION_METRIC")
        cls.LOSS_THRESHOLD: float = HYPER_PARAMETER_TUNING_PARAMS.get("LOSS_THRESHOLD")
        cls.TIMEOUT_MINS: float = HYPER_PARAMETER_TUNING_PARAMS.get("TIMEOUT_MINS")
        cls.MIN_PERFORMANCE: float = HYPER_PARAMETER_TUNING_PARAMS.get("MIN_PERFORMANCE")

        """
        LOG PARAMETERS
        """
        LOG_PARAMS: dict = config.get("LOG_PARAMS")

        cls.LEVEL: str = LOG_PARAMS.get("LEVEL")
        cls.TXT_FMT: str = LOG_PARAMS.get("TXT_FMT")
        cls.JSON_FMT: str = LOG_PARAMS.get("JSON_FMT")
        cls.FILTER_LVLS: str = LOG_PARAMS.get("FILTER_LVLS")
        cls.LOG_FILE: str = LOG_PARAMS.get("LOG_FILE")
        cls.BACKUP_COUNT: int = LOG_PARAMS.get("BACKUP_COUNT")
        
        """
        COMPUTE PARAMETERS
        """
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