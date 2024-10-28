from ml_accelerator.utils.logging.logger_helper import get_logger
from git.repo.base import Repo
from git.exc import InvalidGitRepositoryError
from dotenv import load_dotenv, find_dotenv
import os
from typing import Dict, List


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


def extract_suffix(parameter_name: str, parameter_value: str) -> str:
    if '-' in parameter_value:
        suffix = parameter_value.split('-')[-1]
    elif '_' in parameter_value:
        suffix = parameter_value.split('_')[-1]
    else:
        raise ValueError(f'{parameter_name} ({parameter_value}) must contain a "-" or "_" separator.')
    
    if '.' in suffix:
        suffix = suffix.split('.')[0]
    
    return suffix


def validate_suffix(env: str, parameter_name: str, suffix: str, valid_sufixes: List[str]) -> None:
    if suffix.upper() not in [s.upper() for s in valid_sufixes]:
        raise ValueError(f'{parameter_name} suffix ({suffix}) must be one of: {valid_sufixes}')
    
    if env.upper() != suffix.upper():
        raise ValueError(f'ENV ({env}) and {parameter_name} suffix ({suffix}) must match.')


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

        # Extract ENV
        ENV: str = cls.get('ENV')

        # Extract branch
        branch_name: str = get_current_branch()
        
        # Validate main environment parameters
        if branch_name != "Not a git repository":
            if branch_name == 'main' and ENV != 'prod':
                raise ValueError(f'ENV must be prod for main branch. Got: {ENV}')
            elif branch_name != 'main' and ENV == 'prod':
                raise ValueError(f'ENV must be dev for {branch_name} branch. Got: {ENV}')

            # Define parameters to validate
            params: Dict[str, str] = {
                'BUCKET_NAME': cls.get('BUCKET_NAME'), 
                'ETL_LAMBDA_FUNCTION_NAME': cls.get('ETL_LAMBDA_FUNCTION_NAME'), 
                'LAMBDA_EXECUTION_ROLE_NAME': cls.get('LAMBDA_EXECUTION_ROLE_NAME'), 
                'LAMBDA_EXECUTION_ROLE_ARN': cls.get('LAMBDA_EXECUTION_ROLE_ARN'),
                'SAGEMAKER_EXECUTION_ROLE_NAME': cls.get('SAGEMAKER_EXECUTION_ROLE_NAME'), 
                'SAGEMAKER_EXECUTION_ROLE_ARN': cls.get('SAGEMAKER_EXECUTION_ROLE_ARN'), 
                'MODEL_BUILDING_STEP_FUNCTIONS_NAME': cls.get('MODEL_BUILDING_STEP_FUNCTIONS_NAME'),
                'MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME': cls.get('MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME'), 
                'STEP_FUNCTIONS_EXECUTION_ROLE_NAME': cls.get('STEP_FUNCTIONS_EXECUTION_ROLE_NAME'), 
                'STEP_FUNCTIONS_EXECUTION_ROLE_ARN': cls.get('STEP_FUNCTIONS_EXECUTION_ROLE_ARN')
            }
        else:
            # Define parameters to validate
            params: Dict[str, str] = {
                'BUCKET_NAME': cls.get('BUCKET_NAME')
            }

        # Validate parameter suffixes
        for param in params:
            # Extract suffix
            suffix: str = extract_suffix(param, params[param])

            # Validate suffix
            validate_suffix(ENV, param, suffix, ['dev', 'prod'])

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