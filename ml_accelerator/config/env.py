from ml_accelerator.utils.filesystem.filesystem_helper import find_paths
from ml_accelerator.utils.logging.logger_helper import get_logger
from git.repo.base import Repo
from git.exc import InvalidGitRepositoryError
from dotenv import load_dotenv, find_dotenv
import os
import hcl2
import argparse
from typing import Dict, List, Set


# Get logger
LOGGER = get_logger(name=__name__)

# Extract ENV
ENV: str = os.environ['ENV']

# Find terraform files
if ENV == 'prod':
    terraform_files: Set[str] = find_paths(
        bucket_name='', 
        directory='terraform/production'
    )
else:
    terraform_files: Set[str] = find_paths(
        bucket_name='', 
        directory='terraform/development'
    )

# Load TF_VARS
TF_VARS: dict = {}
for file in terraform_files:
    if file.endswith('.tfvars'):
        with open(file, 'r') as f:
            TF_VARS.update(hcl2.load(f))

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
    # Extract : from ARN
    if ':' in parameter_value:
        parameter_value = parameter_value.split(':')[-1]
    
    # Extract suffix
    if '-' in parameter_value:
        suffix = parameter_value.split('-')[-1]
    elif '_' in parameter_value:
        suffix = parameter_value.split('_')[-1]
    else:
        raise ValueError(f'{parameter_name} ({parameter_value}) must contain a "-" or "_" separator.')
    
    # Remove unrequired file extension
    if '.' in suffix:
        suffix = suffix.split('.')[0]
    
    return suffix


def validate_suffix(
    env: str, 
    parameter_name: str, 
    parameter_value,
    suffix: str, 
    valid_sufixes: List[str]
) -> None:
    if suffix.upper() not in [s.upper() for s in valid_sufixes]:
        raise ValueError(f'{parameter_name} suffix ({suffix}) must be one of: {valid_sufixes}')
    
    if env.upper() != suffix.upper():
        raise ValueError(f'ENV ({env}) and {parameter_name} ({parameter_value}) suffix ({suffix}) must match.')


class Env:
    initialized: bool = False

    @classmethod
    def initialize(cls):
        if cls.initialized:
            return
        
        # LOGGER.debug('Initializing Env.')

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
                'DOCKER_REPOSITORY_NAME': cls.get('DOCKER_REPOSITORY_NAME'),
                'ETL_LAMBDA_FUNCTION_NAME': cls.get('ETL_LAMBDA_FUNCTION_NAME'),
                'MODEL_BUILDING_STEP_FUNCTIONS_NAME': cls.get('MODEL_BUILDING_STEP_FUNCTIONS_NAME'),
                'MODEL_BUILDING_STEP_FUNCTIONS_ARN': cls.get('MODEL_BUILDING_STEP_FUNCTIONS_ARN'),
                'MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME': cls.get('MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME'),
            }
        else:
            params: Dict[str, str] = {
                'BUCKET_NAME': cls.get('BUCKET_NAME')
            }
        
        # Validate parameter suffixes
        for param in params:
            # Extract suffix
            suffix: str = extract_suffix(param, params[param])

            # Validate suffix
            validate_suffix(ENV, param, params[param], suffix, ['dev', 'prod'])

        cls.initialized = True

    @staticmethod
    def get(var_name: str) -> str:
        def get_terraform_var(var_name: str) -> str:
            # Extract param_
            param_: str = TF_VARS.get(var_name) # f'{var_name}_{ENV.upper()}')
            
            if isinstance(param_, str):
                return (
                    param_
                    .replace('${', '')
                    .replace('}', '')
                    .replace(' ', '')
                )
            return param_

        
        # Extract environment parameter
        param: str = os.environ.get(var_name)
        
        # Validate parameter
        if param is None:
            param = get_terraform_var(var_name)
            if param is None:
                raise ValueError(f'{var_name} could not be extracted from environment (nor from .tfvars).') 
        
        try:
            return eval(str(param))
        except:
            return str(param)

    @staticmethod
    def clean_env():
        for var_name in [
            'BUCKET_NAME',
            'ETL_LAMBDA_FUNCTION_NAME',
            'MODEL_BUILDING_STEP_FUNCTIONS_NAME',
            'MODEL_BUILDING_STEP_FUNCTIONS_ARN',
            'MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME',
            'DOCKER_REPOSITORY_NAME',
            'ECR_REPOSITORY_URI'
        ]:
            # Unset environment variable
            os.environ.pop(var_name, None)


# Initialize environment
if not Env.initialized:
    # Clean environment
    Env.clean_env()

    # Instanciate Env class
    Env.initialize()

# .ml_accel_venv/bin/python ml_accelerator/config/env.py --env_param None
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Environment script.')

    # Add arguments
    parser.add_argument('--env_param', type=str, default=None)

    # Extract arguments from parser
    args = parser.parse_args()
    env_param: str = args.env_param

    if env_param is not None:
        # Get Env value
        print(Env.get(env_param))