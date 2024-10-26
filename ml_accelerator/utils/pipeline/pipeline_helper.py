from ml_accelerator.config.env import Env
from ml_accelerator.config.params import Params
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.pipeline.pipeline_helper import get_image_uri, get_data_uri
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.timing.timing_helper import timing
import boto3
import argparse
from typing import List
# Guide: https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-auto-run.html


# Get logger
LOGGER = get_logger(name=__name__)


def get_image_uri(
    docker_repository_type: str = Env.get("DOCKER_REPOSITORY_TYPE"),
    docker_repository_name: str = Env.get("DOCKER_REPOSITORY_NAME"),
    dockerhub_username: str = Env.get("DOCKERHUB_USERNAME"),
    ecr_repository_uri: str = Env.get("ECR_REPOSITORY_URI"),
    env: str = Env.get("ENV"),
    version: str = Params.VERSION
) -> str:
    if docker_repository_type == "dockerhub":
        return f"{dockerhub_username}/{docker_repository_name}:{env}-image-{version}"
    elif docker_repository_type == "ECR":
        return f"{ecr_repository_uri}/{docker_repository_name}:{env}-image-{version}"
    else:
        raise ValueError(f"Invalid docker_repository_type: {docker_repository_type}")
    

def get_instance_type(job_name: str) -> str:
    if 'data-processing' in job_name:
        return Env.get('PROCESSING_INSTANCE_TYPE')
    elif 'tuning' in job_name:
        return Env.get('TUNING_INSTANCE_TYPE')
    elif 'training' in job_name:
        return Env.get('TRAINING_INSTANCE_TYPE')
    elif 'evaluating' in job_name:
        return Env.get('EVALUATING_INSTANCE_TYPE')
    elif 'inference' in job_name:
        return Env.get('INFERENCE_INSTANCE_TYPE')
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default instance type.", job_name
        )
        return Env.get('DEFAULT_INSTANCE_TYPE')


def get_entrypoint(job_name: str) -> str:
    # Define file_name
    file_name = job_name.replace('-', '_')
    
    # Define entrypoint
    entrypoint = f"scripts/{file_name}/{file_name}.py"
    # f'/opt/ml/processing/{script_name}'

    return entrypoint


def get_container_arguments(job_name: str) -> List[str]:
    if 'data-processing' in job_name:
        return [
            "--fit_transformers", Params.MODEL_BUILDING_PARAMS["fit_transformers"], 
            "--save_transformers", Params.MODEL_BUILDING_PARAMS["save_transformers"],
            "--persist_datasets", Params.MODEL_BUILDING_PARAMS["persist_datasets"],
            "--write_mode", Params.MODEL_BUILDING_PARAMS["write_mode"]
        ]
    elif 'tuning' in job_name:
        return []
    elif 'training' in job_name:
        return [
            "--train_prod_pipe", Params.MODEL_BUILDING_PARAMS["train_prod_pipe"],
            "--train_staging_pipes", Params.MODEL_BUILDING_PARAMS["train_staging_pipes"],
            "--train_dev_pipes", Params.MODEL_BUILDING_PARAMS["train_dev_pipes"]
        ]
    elif 'evaluating' in job_name:
        return [
            "--evaluate_prod_pipe", Params.MODEL_BUILDING_PARAMS["evaluate_prod_pipe"],
            "--evaluate_staging_pipes", Params.MODEL_BUILDING_PARAMS["evaluate_staging_pipes"],
            "--evaluate_dev_pipes", Params.MODEL_BUILDING_PARAMS["evaluate_dev_pipes"],
            "--update_model_stages", Params.MODEL_BUILDING_PARAMS["update_model_stages"],
            "--update_prod_model", Params.MODEL_BUILDING_PARAMS["update_prod_model"]
        ]
    elif 'inference' in job_name:
        return [ "pred_id", "0" ]
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default container arguments.", job_name
        )
        return []


def get_instance_count(job_name: str) -> int:
    if 'data-processing' in job_name:
        return int(Env.get('PROCESSING_INSTANCE_COUNT'))
    elif 'tuning' in job_name:
        return int(Env.get('TUNING_INSTANCE_COUNT'))
    elif 'training' in job_name:
        return int(Env.get('TRAINING_INSTANCE_COUNT'))
    elif 'evaluating' in job_name:
        return int(Env.get('EVALUATING_INSTANCE_COUNT'))
    elif 'inference' in job_name:
        return int(Env.get('INFERENCE_INSTANCE_COUNT'))
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default instance count.", job_name
        )
        return int(Env.get('DEFAULT_INSTANCE_COUNT'))
    

def get_volume_size(job_name: str) -> int:
    if 'data-processing' in job_name:
        return int(Env.get('PROCESSING_VOLUME_SIZE'))
    elif 'tuning' in job_name:
        return int(Env.get('TUNING_VOLUME_SIZE'))
    elif 'training' in job_name:
        return int(Env.get('TRAINING_VOLUME_SIZE'))
    elif 'evaluating' in job_name:
        return int(Env.get('EVALUATING_VOLUME_SIZE'))
    elif 'inference' in job_name:
        return int(Env.get('INFERENCE_VOLUME_SIZE'))
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default volume size.", job_name
        )
        return int(Env.get('DEFAULT_VOLUME_SIZE'))
    

def get_max_runtime(job_name: str) -> int:
    if 'data-processing' in job_name:
        return int(Env.get('PROCESSING_MAX_RUNTIME'))
    elif 'tuning' in job_name:
        return int(Env.get('TUNING_MAX_RUNTIME'))
    elif 'training' in job_name:
        return int(Env.get('TRAINING_MAX_RUNTIME'))
    elif 'evaluating' in job_name:
        return int(Env.get('EVALUATING_MAX_RUNTIME'))
    elif 'inference' in job_name:
        return int(Env.get('INFERENCE_MAX_RUNTIME'))
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default max runtime.", job_name
        )
        return int(Env.get('DEFAULT_MAX_RUNTIME'))


def get_data_uri(dataset_name: str) -> str:
    # Find subdir
    if 'raw' in dataset_name:
        subdir = Env.get('RAW_DATASETS_PATH')
    else:
        subdir = Env.get('PROCESSING_DATASETS_PATH')

    # Find bucket name
    bucket_name = Env.get('BUCKET_NAME')

    # Find extention
    extention = Params.DATA_EXTENTION

    if extention == "csv":
        return f"s3://{bucket_name}/{subdir}/{dataset_name}.csv"
    elif extention == "parquet":
        return f"s3://{bucket_name}/{subdir}/{dataset_name}/"
    else:
        raise ValueError(f"Invalid data_extention: {extention}")


def get_processing_inputs(job_name: str) -> List[str]:
    if Env.get("DATA_STORAGE_ENV") != "S3":
        raise ValueError(
            f"SageMaker jobs can only be ran within an S3 storage environment.\n"
            f"Current starage env: {Env.get('DATA_STORAGE_ENV')}"
        )

    # Find last Transformer step
    last_transformer: str = Params.TRANSFORMERS_STEPS[-1]
    
    if 'data-processing' in job_name:
        """
        TODO:
            - Decouple data_processing into etl & data_processing
            - When decoupled, the input data for data-processing will be the output from etl
        """
        return []
    elif 'tuning' in job_name:
        return [
            # X Datasets
            {
                'InputName': 'X-data',
                'S3Input': {
                    'S3Uri': get_data_uri(dataset_name=f'X_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/input',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            # y datasets
            {
                'InputName': 'y-data',
                'S3Input': {
                    'S3Uri': get_data_uri(dataset_name=f'y_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/input',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        ]
    else:
        raise NotImplementedError(f"Processing inputs for {job_name} have not been implemented yet.")
    

def get_processing_outputs(job_name: str) -> List[str]:
    if Env.get("DATA_STORAGE_ENV") != "S3":
        raise ValueError(
            f"SageMaker jobs can only be ran within an S3 storage environment.\n"
            f"Current starage env: {Env.get('DATA_STORAGE_ENV')}"
        )
    
    # Find last Transformer step
    last_transformer: str = Params.TRANSFORMERS_STEPS[-1]

    if 'data-processing' in job_name:
        # data-processing output should match tuning & training input
        return [
            # X Datasets
            {
                'OutputName': 'X-data',
                'S3Output': {
                    'S3Uri': get_data_uri(dataset_name=f'X_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/output',
                    'S3UploadMode': 'EndOfJob'
                },
                # 'FeatureStoreOutput': {
                #     'FeatureGroupName': 'string'
                # }
            },
            # y datasets
            {
                'OutputName': 'y-data',
                'S3Output': {
                    'S3Uri': get_data_uri(dataset_name=f'y_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/output',
                    'S3UploadMode': 'EndOfJob'
                },
                # 'FeatureStoreOutput': {
                #     'FeatureGroupName': 'string'
                # }
            }
        ]
    else:
        raise NotImplementedError(f"Processing output for {job_name} have not been implemented yet.")


def run_sagemaker_processing_job(job_name: str) -> dict:
    # Instanciate SageMaker client
    sagemaker_client = boto3.client('sagemaker')

    # Find image uri
    ecr_image_uri: str = get_image_uri()

    # Find sagemaker role ARN
    role_arn: str = Env.get("STEP_FUNCTIONS_EXECUTION_ROLE_ARN")

    # Find instance type
    instance_type: str = get_instance_type(job_name=job_name)

    # Find entrypoint
    entrypoint: str = get_entrypoint(job_name=job_name)

    # Find container arguments
    container_args: List[str] = get_container_arguments(job_name=job_name)

    # Find instance count
    instance_count: int = get_instance_count(job_name=job_name)

    # Find volume size
    volume_size: int = get_volume_size(job_name=job_name)

    # Find max runtime
    max_runtime: int = get_max_runtime(job_name=job_name)

    # Find processing inputs
    processing_inputs: List[str] = get_processing_inputs(job_name=job_name)

    # Find processing outputs
    processing_outputs: List[str] = get_processing_outputs(job_name=job_name)

    # Create Processing Job
    response: dict = sagemaker_client.create_processing_job(
        # Define job name
        ProcessingJobName=job_name,

        # Sagemaker role ARN
        RoleArn=role_arn,

        # Docker image parameters
        AppSpecification={
            'ImageUri': ecr_image_uri,
            'ContainerEntrypoint': ['python3', entrypoint],
            'ContainerArguments': container_args
        },

        # Resource configuration
        ProcessingResources={
            'ClusterConfig': {
                'InstanceCount': instance_count,
                'InstanceType': instance_type,
                'VolumeSizeInGB': volume_size
            }
        },

        # Environment configuration (already defined in Dockerfile)
        # Environment={
        #     'string': 'string'
        # },

        # Tags
        Tags=[
            {
                'Project': Params.PROJECT_NAME,
                'Version': Params.VERSION,
                'Environment': Env.get("ENV")
            },
        ],

        # Stopping condition
        StoppingCondition={
            'MaxRuntimeInSeconds': max_runtime
        },
        
        # Input configuration
        ProcessingInputs=[processing_inputs],

        # Output configuration
        ProcessingOutputConfig={
            'Outputs': processing_outputs
        },

        # Network configuration
        # NetworkConfig={
        #     'EnableInterContainerTrafficEncryption': True|False,
        #     'EnableNetworkIsolation': True|False,
        #     'VpcConfig': {
        #         'SecurityGroupIds': [
        #             'string',
        #         ],
        #         'Subnets': [
        #             'string',
        #         ]
        #     }
        # }
    )

    # Log response
    LOGGER.info(f"Processing job {job_name} created with response:\n{response}")

    return response


"""
source .ml_accel_venv/bin/activate
conda deactivate
.ml_accel_venv/bin/python pipelines/model_building/sagemaker/sagemaker_jobs.py --job_name data-processing
"""
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Data processing script.')

    # Add arguments
    parser.add_argument('--job_name', type=str, default='data-processing')

    # Extract arguments from parser
    args = parser.parse_args()
    job_name: str = args.job_name

    # Run processing job
    run_sagemaker_processing_job(job_name=job_name)

