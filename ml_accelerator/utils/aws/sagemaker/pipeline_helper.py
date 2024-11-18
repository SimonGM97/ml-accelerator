from ml_accelerator.config.env import Env
from ml_accelerator.utils.logging.logger_helper import get_logger
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession
import boto3
from datetime import datetime
from typing import Dict


# Get logger
LOGGER = get_logger(name=__name__)


def find_pipeline_session() -> PipelineSession | LocalPipelineSession:
    if Env.get("MODEL_BUILDING_ENV") == 'sagemaker':
        return PipelineSession()
    else:
        return LocalPipelineSession()


def find_pipeline_name(pipeline_name: str) -> str:
    if "model-building" in pipeline_name:
        return f"ModelBuildingPipeline{Env.get('ENV').title()}"
    elif "model-updating" in pipeline_name:
        return f"ModelUpdatingPipeline{Env.get('ENV').title()}"
    elif "inference" in pipeline_name:
        return f"InferencePipeline{Env.get('ENV').title()}"
    
    raise Exception(f'Invalid "pipeline_name" was received: {pipeline_name}')


def find_pipeline_desc(pipeline_name: str) -> str:
    if "model-building" in pipeline_name:
        return "Pipeline that will process datasets, tune, train, evaluate & select SageMaker Models."
    elif "model-updating" in pipeline_name:
        return "Pipeline that will re-train, re-evaluete & re-select SageMaker Models."
    elif "inference" in pipeline_name:
        return "Pipeline that will expose a SageMaker endpoint to generate new inferences on demand."
    
    raise Exception(f'Invalid "pipeline_name" was received: {pipeline_name}')


def find_execution_display_name(pipeline_name: str) -> str:
    # Extract datetime parameters
    date = datetime.today()
    year = str(date.year)
    month = ('0' + str(date.month))[-2:]
    day = ('0' + str(date.day))[-2:]
    hs = ('0' + str(date.hour))[-2:]
    mins = ('0' + str(date.minute))[-2:]
    secs = ('0' + str(date.second))[-2:]

    # Redefine name
    return f"{pipeline_name}-execution-{year}-{month}-{day}-{hs}{mins}{secs}"


def pipeline_exists(pipeline_name: str) -> bool:
    """
    Check if a SageMaker pipeline with the given name exists.

    :param pipeline_name: Name of the SageMaker pipeline to check.
    :param region_name: AWS region where the pipeline is located.
    :return: True if the pipeline exists, False otherwise.
    """
    sagemaker_client = boto3.client('sagemaker', region_name=Env.get("REGION_NAME"))
    
    # List pipelines with a filter on the name (if pipeline names contain unique substrings)
    response = sagemaker_client.list_pipelines(
        PipelineNamePrefix=pipeline_name
    )

    # Check if the pipeline exists in the response
    for pipeline in response['PipelineSummaries']:
        if pipeline['PipelineName'] == pipeline_name:
            LOGGER.info(f"Pipeline {pipeline_name} exists.")
            return True
    
    LOGGER.info(f"Pipeline {pipeline_name} does not exist.")
    return False
