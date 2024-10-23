from ml_accelerator.utils.logging.logger_helper import get_logger
from git.repo.base import Repo
from git.exc import InvalidGitRepositoryError
from dotenv import load_dotenv, find_dotenv
import os


# Get logger
LOGGER = get_logger(name=__name__)


def get_current_branch() -> str:
    try:
        # Get the current repository from the current working directory
        repo = Repo(search_parent_directories=True)
        # Extract the active branch name
        branch_name = repo.active_branch.name
        return branch_name
    except InvalidGitRepositoryError:
        return "Not a git repository"


class Env:
    initialized: bool = False

    @classmethod
    def initialize(cls):
        if cls.initialized:
            return
        
        LOGGER.info('Initializing Env.')

        # Set environment parameters from .env
        load_dotenv(
            dotenv_path=find_dotenv(),
            override=True
        )

        # Extract parameters to validate
        ENV: str = cls.get('ENV')
        BUCKET_NAME: str = cls.get('BUCKET_NAME')

        # Validate parameters
        if ENV not in ['dev', 'prod']:
            raise ValueError(f'ENV must be either dev or prod. Got: {ENV}')
        if BUCKET_NAME.split('-')[-1] not in ['dev', 'prod']:
            raise ValueError(f'BUCKET_NAME suffix must be either dev or prod. Got: {BUCKET_NAME.split("-")[-1]} ({BUCKET_NAME})')
        
        if ENV != BUCKET_NAME.split('-')[-1]:
            raise Exception(f'ENV ({ENV}) and BUCKET_NAME suffix ({BUCKET_NAME.split("-")[-1]}) must match.')
        
        # Extract branch
        branch_name: str = get_current_branch()
        
        # Validate main environment parameters
        if branch_name != "Not a git repository":
            if branch_name == 'main':
                if ENV != 'prod':
                    raise ValueError(f'ENV must be prod for main branch. Got: {ENV}')
                if BUCKET_NAME.split('-')[-1] != 'prod':
                    raise ValueError(f'BUCKET_NAME suffix must be prod for main branch. Got: {BUCKET_NAME.split("-")[-1]} ({BUCKET_NAME})')
            else:
                if ENV == 'prod':
                    raise ValueError(f'ENV cannot be "prod" for {branch_name} branch.')
                if BUCKET_NAME.split('-')[-1] == 'prod':
                    raise ValueError(f'BUCKET_NAME suffix cannot be "prod" for {branch_name} branch.')

        cls.initialized = True

    @staticmethod
    def get(var_name: str) -> str:
        # Extract environment parameter
        param: str = os.environ.get(var_name)

        # Validate parameter
        if param is None:
            raise ValueError(f'{var_name} could not be extracted from environment.') 
        
        return param


# .ml_accel_venv/bin/python ml_accelerator/config/env.py
if not Env.initialized:
    Env.initialize()