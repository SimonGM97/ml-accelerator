import pandas as pd
import yaml
from pathlib import Path
from git.repo.base import Repo
import multiprocessing
import subprocess
import os


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

    # DATA CLEANING PARAMETERS

    # MODELING PARAMETERS

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
        cls.BUCKET: str = cls.PROJECT_NAME + "-" + cls.ENV
        cls.CWD = find_base_repo_root(project_name=cls.PROJECT_NAME)

        """
        DATA PARAMETERS
        """
        DATA_PARAMS: dict = config.get("DATA_PARAMS")

        """
        DATA CLEANING PARAMETERS
        """
        DATA_CLEANING_PARAMS: dict = DATA_PARAMS.get("DATA_CLEANING_PARAMS")

        cls.NON_NEG_COLS: list = DATA_CLEANING_PARAMS.get("NON_NEG_COLS")
        
        """
        FEATURE SELECTION PARAMETERS
        """
        FEATURE_SELECTION_PARAMS: dict = DATA_PARAMS.get("FEATURE_SELECTION_PARAMS")

        cls.N_FEATURES: list = FEATURE_SELECTION_PARAMS.get("N_FEATURES")

        """
        MODELING PARAMETERS
        """
        MODELING_PARAMS: dict = config.get("MODELING_PARAMS")

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