from ml_accelerator.config.env import Env
from ml_accelerator.config.params import Params
from ml_accelerator.utils.aws.sagemaker.sagemaker_jobs_helper import get_job_parameters
from ml_accelerator.utils.filesystem.filesystem_helper import save_to_filesystem
from ml_accelerator.utils.aws.s3_helper import save_to_s3
from ml_accelerator.utils.logging.logger_helper import get_logger
import argparse
import json
import os


# Get logger
LOGGER = get_logger(name=__name__)


def find_state_definitions(step_function_name: str) -> dict:
    # Define default states definition
    state_definitions: dict = {
        # Step Functions name
        "Comment": step_function_name,

        # Initial step
        "StartAt": f"{Params.STEP_FUNCTION_STATES[0]['job_name']}-job",

        # Define empty states
        "States": {}
    }

    # Loop over step function steps
    for state in Params.STEP_FUNCTION_STATES:
        # Define empty step parameters
        state_definition: dict = {}

        # Extract state values
        step_n: int = state["step_n"]
        resource: str = state["resource"]
        job_name: str = state["job_name"]
        next_step: str = state["next_step"]
        end: bool = state["end"]

        # Extract job parameters
        job_parameters: dict = get_job_parameters(job_name)

        # Add parameters
        state_definition["Parameters"] = job_parameters

        # Add default keys
        state_definition.update(**{
            "Type": "Task",
            "Resource": f"arn:aws:states:::{resource}"
        })

        # Add Next step
        if next_step is not None:
            state_definition["Next"] = f"{next_step}-job"

        # Add End step
        if end:
            state_definition["End"] = True

        # Add state definition
        state_definitions["States"][f"{job_name}-job"] = state_definition

    return state_definitions


"""
source .ml_accel_venv/bin/activate
conda deactivate
.ml_accel_venv/bin/python ml_accelerator/utils/aws/step_functions_helper.py
"""
if __name__ == "__main__":
    # Extract step_function_name & file_name
    step_function_name: str = Env.get("MODEL_BUILDING_STEP_FUNCTIONS_NAME")
    file_name: str = Env.get("MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME")

    # Extract state_definitions
    state_definitions: dict = find_state_definitions(step_function_name)

    # Show state definitions
    LOGGER.info('%s state definitions: %s', step_function_name, json.dumps(state_definitions, indent=4))

    # Save state definitions
    ENV: str = Env.get("ENV")
    if ENV == 'prod':
        save_to_filesystem(
            asset=state_definitions, 
            path=os.path.join('terraform', 'production', 'step_functions', file_name),
            write_mode='overwrite'
        )
    else:
        save_to_filesystem(
            asset=state_definitions, 
            path=os.path.join('terraform', 'development', 'step_functions', file_name),
            write_mode='overwrite'
        )
    
