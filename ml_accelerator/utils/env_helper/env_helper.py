import os
from dotenv import load_dotenv, find_dotenv


# Set environment parameters from .env
load_dotenv(
    dotenv_path=find_dotenv(),
    override=True
)

def find_env_var(var_name: str) -> str:
    # Extract environment parameter
    param: str = os.environ.get(var_name)

    # Validate parameter
    if param is None:
        raise ValueError(f'{var_name} could not be extracted from environment.') 
    
    return param

# .ml_accel_venv/bin/python ml_accelerator/utils/env_helper/env_helper.py