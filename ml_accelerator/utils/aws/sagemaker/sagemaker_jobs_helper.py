from ml_accelerator.config.env import Env
from ml_accelerator.config.params import Params
from ml_accelerator.utils.logging.logger_helper import get_logger
import boto3
from datetime import datetime
import argparse
from typing import List
from pprint import pformat
# Guide: https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-auto-run.html


# Get logger
LOGGER = get_logger(name=__name__)


def define_job_name(job_name: str) -> str:
    """
    ProcessingJobName: The name of the processing job. The name must be unique within an Amazon 
    Web Services Region in the Amazon Web Services account.
    """
    date = datetime.today()
    year = str(date.year)
    month = ('0' + str(date.month))[-2:]
    day = ('0' + str(date.day))[-2:]
    hs = ('0' + str(date.hour))[-2:]
    mins = ('0' + str(date.minute))[-2:]
    secs = ('0' + str(date.second))[-2:]

    return f"{Env.get('ENV')}--{job_name}--{year}-{month}-{day}-{hs}{mins}{secs}"


def get_instance_count(job_name: str) -> int:
    if 'data-processing' in job_name:
        return int(Env.get('PROCESSING_INSTANCE_COUNT'))
    elif 'tuning' in job_name:
        return int(Env.get('TUNING_INSTANCE_COUNT'))
    elif 'training' in job_name:
        return int(Env.get('TRAINING_INSTANCE_COUNT'))
    elif 'evaluating' in job_name:
        return int(Env.get('EVALUATING_INSTANCE_COUNT'))
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default instance count.", job_name
        )
        return int(Env.get('DEFAULT_INSTANCE_COUNT'))


def get_instance_type(job_name: str) -> str:
    if 'data-processing' in job_name:
        return Env.get('PROCESSING_INSTANCE_TYPE')
    elif 'tuning' in job_name:
        return Env.get('TUNING_INSTANCE_TYPE')
    elif 'training' in job_name:
        return Env.get('TRAINING_INSTANCE_TYPE')
    elif 'evaluating' in job_name:
        return Env.get('EVALUATING_INSTANCE_TYPE')
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default instance type.", job_name
        )
        return Env.get('DEFAULT_INSTANCE_TYPE')
    

def get_volume_size(job_name: str) -> int:
    if 'data-processing' in job_name:
        return int(Env.get('PROCESSING_VOLUME_SIZE'))
    elif 'tuning' in job_name:
        return int(Env.get('TUNING_VOLUME_SIZE'))
    elif 'training' in job_name:
        return int(Env.get('TRAINING_VOLUME_SIZE'))
    elif 'evaluating' in job_name:
        return int(Env.get('EVALUATING_VOLUME_SIZE'))
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default volume size.", job_name
        )
        return int(Env.get('DEFAULT_VOLUME_SIZE'))
    

def get_processing_resources(job_name: str) -> dict:
    """
    ProcessingResources (dict) – [REQUIRED]: Identifies the resources, ML compute instances, 
    and ML storage volumes to deploy for a processing job. In distributed training, you specify 
    more than one instance.

    # ClusterConfig: (dict) – [REQUIRED]: The configuration for the resources in a cluster used 
    # to run the processing job.
    'ClusterConfig': {
        # InstanceCount (integer) – [REQUIRED]: The number of ML compute instances to use in 
        # the processing job. For distributed processing jobs, specify a value greater than 1. 
        # The default value is 1.
        'InstanceCount': 123,

        # InstanceType (string) – [REQUIRED]: The ML compute instance type for the processing 
        # job.
        'InstanceType':'ml.t3.medium'|'ml.m4.xlarge'|'ml.p2.8xlarge', ...

        # VolumeSizeInGB (integer) – [REQUIRED]: The size of the ML storage volume in gigabytes
        'VolumeSizeInGB': 123
    }
    """
    return {
        'ClusterConfig': {
            'InstanceCount': get_instance_count(job_name=job_name),
            'InstanceType': get_instance_type(job_name=job_name),
            'VolumeSizeInGB': get_volume_size(job_name=job_name)
        }
    }


def get_max_runtime(job_name: str) -> int:
    if 'data-processing' in job_name:
        return int(Env.get('PROCESSING_MAX_RUNTIME'))
    elif 'tuning' in job_name:
        return int(Env.get('TUNING_MAX_RUNTIME'))
    elif 'training' in job_name:
        return int(Env.get('TRAINING_MAX_RUNTIME'))
    elif 'evaluating' in job_name:
        return int(Env.get('EVALUATING_MAX_RUNTIME'))
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default max runtime.", job_name
        )
        return int(Env.get('DEFAULT_MAX_RUNTIME'))
    

def get_stopping_condition(job_name: str) -> dict:
    """
    StoppingCondition (dict): The time limit for how long the processing job is allowed to run.
    {
        # MaxRuntimeInSeconds (integer) – [REQUIRED]: Specifies the maximum runtime in seconds.
        'MaxRuntimeInSeconds': 123
    }
    """
    return {
        'MaxRuntimeInSeconds': get_max_runtime(job_name=job_name)
    }


def get_image_uri() -> str:
    # Extract parameters
    docker_repository_type: str = Env.get("DOCKER_REPOSITORY_TYPE")
    docker_repository_name: str = Env.get("DOCKER_REPOSITORY_NAME")
    dockerhub_username: str = Env.get("DOCKERHUB_USERNAME")
    ecr_repository_uri: str = Env.get("ECR_REPOSITORY_URI")
    env: str = Env.get("ENV")
    version: str = Params.VERSION

    if docker_repository_type == "dockerhub":
        return f"{dockerhub_username}/{docker_repository_name}:{env}-image-{version}"
    elif docker_repository_type == "ECR":
        return f"{ecr_repository_uri}/{docker_repository_name}:{env}-image-{version}"
    else:
        raise ValueError(f"Invalid docker_repository_type: {docker_repository_type}")
    

def get_entrypoint(job_name: str) -> str:
    # Define file_name
    file_name = job_name.split('--')[1].replace('-', '_')
    
    # Define entrypoint
    entrypoint = f"scripts/{file_name}/{file_name}.py"
    # f'/opt/ml/processing/{script_name}'

    return ['python3', entrypoint]


def get_container_arguments(job_name: str) -> List[str]:
    if 'data-processing' in job_name:
        return [
            "--fit_transformers", str(Params.MODEL_BUILDING_PARAMS["FIT_TRANSFORMERS"]), 
            "--save_transformers", str(Params.MODEL_BUILDING_PARAMS["SAVE_TRANSFORMERS"]),
            "--persist_datasets", str(Params.MODEL_BUILDING_PARAMS["PERSIST_DATASETS"]),
            "--write_mode", str(Params.MODEL_BUILDING_PARAMS["WRITE_MODE"])
        ]
    elif 'tuning' in job_name:
        return None
    elif 'training' in job_name:
        return [
            "--train_prod_pipe", str(Params.MODEL_BUILDING_PARAMS["TRAIN_PROD_PIPE"]),
            "--train_staging_pipes", str(Params.MODEL_BUILDING_PARAMS["TRAIN_STAGING_PIPES"]),
            "--train_dev_pipes", str(Params.MODEL_BUILDING_PARAMS["TRAIN_DEV_PIPES"])
        ]
    elif 'evaluating' in job_name:
        return [
            "--evaluate_prod_pipe", str(Params.MODEL_BUILDING_PARAMS["EVALUATE_PROD_PIPE"]),
            "--evaluate_staging_pipes", str(Params.MODEL_BUILDING_PARAMS["EVALUATE_STAGING_PIPES"]),
            "--evaluate_dev_pipes", str(Params.MODEL_BUILDING_PARAMS["EVALUATE_DEV_PIPES"]),
            "--update_model_stages", str(Params.MODEL_BUILDING_PARAMS["UPDATE_MODEL_STAGES"]),
            "--update_prod_model", str(Params.MODEL_BUILDING_PARAMS["UPDATE_PROD_MODEL"])
        ]
    else:
        LOGGER.warning(
            "Unexpected job_name was received %s\n."
            "Using default container arguments.", job_name
        )
        return None


def get_app_specification(job_name: str) -> dict:
    """
    AppSpecification (dict) – [REQUIRED]: Configures the processing job to run a specified 
    Docker container image.

    {
        # ImageUri (string) – [REQUIRED]: The container image to be run by the processing job.
        'ImageUri': 'string',

        # ContainerEntrypoint (list) – The entrypoint for a container used to run a processing 
        # job.
        'ContainerEntrypoint': [
            'string',
        ],

        # ContainerArguments (list) – The arguments for a container used to run a processing 
        # job.
        'ContainerArguments': [
            'string',
        ]
    }
    """
    # Define app_specification without ContainerArguments
    app_specification: dict = {
        'ImageUri': get_image_uri(),
        'ContainerEntrypoint': get_entrypoint(job_name=job_name)
    }

    # Find container arguments
    container_arguments: list = get_container_arguments(job_name=job_name)

    # Add container arguments if they exist
    if container_arguments is not None:
        app_specification['ContainerArguments'] = container_arguments
    
    return app_specification


def get_environment() -> dict:
    """
    Environment (dict): The environment variables to set in the Docker container. Up to 100 key 
    and values entries in the map are supported.

    {
        # Key: value
        'string': 'string'
    }
    """
    return {
        'ENV': Env.get("ENV"),
        'DATA_STORAGE_ENV': Env.get("DATA_STORAGE_ENV"),
        'MODEL_STORAGE_ENV': Env.get("MODEL_STORAGE_ENV"),

        'REGION_NAME': Env.get("REGION_NAME"),
        'BUCKET_NAME': Env.get("BUCKET_NAME"),

        'KXY_API_KEY': Env.get("KXY_API_KEY"),

        'INFERENCE_HOST': Env.get("INFERENCE_HOST"),
        'INFERENCE_PORT': Env.get("INFERENCE_PORT"),
        'WEBAPP_HOST': Env.get("WEBAPP_HOST"),
        'WEBAPP_PORT': Env.get("WEBAPP_PORT"),

        'RAW_DATASETS_PATH': Env.get("RAW_DATASETS_PATH"),
        'PROCESSING_DATASETS_PATH': Env.get("PROCESSING_DATASETS_PATH"),
        'INFERENCE_PATH': Env.get("INFERENCE_PATH"),
        'TRANSFORMERS_PATH': Env.get("TRANSFORMERS_PATH"),
        'MODELS_PATH': Env.get("MODELS_PATH"),
        'SCHEMAS_PATH': Env.get("SCHEMAS_PATH"),
        'MOCK_PATH': Env.get("MOCK_PATH"),

        'SEED': Env.get("SEED")
    }


def get_role_arn() -> str:
    """
    RoleArn (string) – [REQUIRED]: The Amazon Resource Name (ARN) of an IAM role that Amazon 
    SageMaker can assume to perform tasks on your behalf.
    """
    return Env.get("SAGEMAKER_EXECUTION_ROLE_ARN") # Env.get("BASE_ROLE_ARN") + "/" + 


def get_tags() -> List[dict]:
    """
    Tags (list) – (Optional): An array of key-value pairs.
        - A tag object that consists of a key and an optional value, used to manage metadata 
          for SageMaker Amazon Web Services resources.
        - You can add tags to notebook instances, training jobs, hyperparameter tuning jobs, 
          batch transform jobs, models, labeling jobs, work teams, endpoint configurations, 
          and endpoints.
    """
    return [
        {
            'Key': 'Project',
            'Value': Params.PROJECT_NAME
        },
        {
            'Key': 'Version',
            'Value': Params.VERSION
        },
        {   
            'Key': 'Environment',
            'Value': Env.get("ENV")
        }
    ]


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
    """
    ProcessingInputs (list): An array of inputs configuring the data to download into the 
    processing container.
        - The inputs for a processing job. The processing input must specify exactly one of 
          either S3Input or DatasetDefinition types.
    
    {
        # The name for the processing job input.
        "InputName": "string",

        # When True, input operations such as data download are managed natively by the 
        # processing job application. 
        # When False (default), input operations are managed by Amazon SageMaker.
        'AppManaged': True | False,

        # Configuration for downloading input data from Amazon S3 into the processing container.
        'S3Input': {
            # The URI of the Amazon S3 prefix Amazon SageMaker downloads data required to run a 
            # processing job.
            'S3Uri': 'string',

            # The local path in your container where you want Amazon SageMaker to write input 
            # data to. 
            #   - LocalPath is an absolute path to the input data and must begin with 
            #     /opt/ml/processing/. 
            #   - LocalPath is a required parameter when AppManaged is False (default).
            'LocalPath': 'string',

            # Whether you use an S3Prefix or a ManifestFile for the data type. 
            #   - If you choose S3Prefix, S3Uri identifies a key name prefix. 
            #       - Amazon SageMaker uses all objects with the specified key name prefix for 
            #         the processing job.
            #   - If you choose ManifestFile, S3Uri identifies an object that is a manifest file 
            #     containing a list of object keys that you want Amazon SageMaker to use for the 
            #     processing job.
            'S3DataType': 'ManifestFile'|'S3Prefix',

            # Whether to use File or Pipe input mode. 
            #   - In File mode, Amazon SageMaker copies the data from the input source onto the 
            #     local ML storage volume before starting your processing container. This is the 
            #     most commonly used input mode. 
            #   - In Pipe mode, Amazon SageMaker streams input data from the source directly to 
            #     your processing container into named pipes without using the ML storage volume.
            'S3InputMode': 'Pipe'|'File',

            # Whether to distribute the data from Amazon S3 to all processing instances with 
            # FullyReplicated, or whether the data from Amazon S3 is shared by Amazon S3 key, 
            # downloading one shard of data to each processing instance.
            'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',

            # Whether to GZIP-decompress the data in Amazon S3 as it is streamed into the processing 
            # container. Gzip can only be used when Pipe mode is specified as the S3InputMode. In 
            # Pipe mode, Amazon SageMaker streams input data from the source directly to your 
            # container without using the EBS volume.
            'S3CompressionType': 'None'|'Gzip'
    }
    """
    if Env.get("DATA_STORAGE_ENV") != "S3":
        raise ValueError(
            f"SageMaker jobs can only be ran within an S3 storage environment.\n"
            f"Current starage env: {Env.get('DATA_STORAGE_ENV')}"
        )

    # Find last Transformer step
    last_transformer: str = Params.TRANSFORMERS_STEPS[-1]
    
    if 'data-processing' in job_name:
        return [
            {
                'InputName': 'raw-data',
                'AppManaged': True,
                'S3Input': {
                    'S3Uri': get_data_uri(dataset_name='df_raw'),
                    'LocalPath': '/opt/ml/processing/input',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3CompressionType': 'None'
                }
            }
        ]
    elif 'tuning' in job_name:
        return [
            # X Datasets
            {
                'InputName': 'X-data',
                'AppManaged': True,
                'S3Input': {
                    'S3Uri': get_data_uri(dataset_name=f'X_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/input/X',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3CompressionType': 'None'
                }
            },
            # y datasets
            {
                'InputName': 'y-data',
                'AppManaged': True,
                'S3Input': {
                    'S3Uri': get_data_uri(dataset_name=f'y_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/input/y',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3CompressionType': 'None'
                }
            }
        ]
    elif 'training' in job_name:
        return [
            {
                'InputName': 'raw-data',
                'AppManaged': True,
                'S3Input': {
                    'S3Uri': get_data_uri(dataset_name='df_raw'),
                    'LocalPath': '/opt/ml/processing/input',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3CompressionType': 'None'
                }
            }
        ]
    elif 'evaluating' in job_name:
        return [
            {
                'InputName': 'raw-data',
                'AppManaged': True,
                'S3Input': {
                    'S3Uri': get_data_uri(dataset_name='df_raw'),
                    'LocalPath': '/opt/ml/processing/input',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3CompressionType': 'None'
                }
            }
        ]
    else:
        raise NotImplementedError(f"Processing inputs for {job_name} have not been implemented yet.")
    

def get_processing_outputs(job_name: str) -> List[str]:
    """
    ProcessingOutputConfig: Output configuration for the processing job.
        - An array of outputs configuring the data to upload from the processing container.

    Describes the results of a processing job. The processing output must specify exactly one of
    either S3Output or FeatureStoreOutput types.
    {
        # The name for the processing job output.
        'OutputName': 'string',

        # Configuration for processing job outputs in Amazon S3.
        'S3Output': {
            # A URI that identifies the Amazon S3 bucket where you want Amazon SageMaker to save 
            # the results of a processing job.
            'S3Uri': 'string',
            
            # The local path of a directory where you want Amazon SageMaker to upload its contents 
            # to Amazon S3. LocalPath is an absolute path to a directory containing output files. 
            # This directory will be created by the platform and exist when your container’s 
            # entrypoint is invoked.
            'LocalPath': 'string',

            # Whether to upload the results of the processing job continuously or after the job 
            # completes.
            'S3UploadMode': 'Continuous'|'EndOfJob'
        },

        # Configuration for processing job outputs in Amazon SageMaker Feature Store. This 
        # processing output type is only supported when AppManaged is specified.
        'FeatureStoreOutput': {
            # The name of the Amazon SageMaker FeatureGroup to use as the destination for 
            # processing job output. Note that your processing script is responsible for putting 
            # records into your Feature Store.
            'FeatureGroupName': 'string'
        },

        # When True, output operations such as data upload are managed natively by the processing 
        # job application. When False (default), output operations are managed by Amazon SageMaker.
        'AppManaged': True | False
    }
    """
    if Env.get("DATA_STORAGE_ENV") != "S3":
        raise ValueError(
            f"SageMaker jobs can only be ran within an S3 storage environment.\n"
            f"Current starage env: {Env.get('DATA_STORAGE_ENV')}"
        )
    
    # Find last Transformer step
    last_transformer: str = Params.TRANSFORMERS_STEPS[-1]

    if 'data-processing' in job_name:
        # data-processing output should match tuning & training input
        processing_outputs = [
            # X Datasets
            {
                'OutputName': 'X-data',
                'S3Output': {
                    'S3Uri': get_data_uri(dataset_name=f'X_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/output/X',
                    'S3UploadMode': 'EndOfJob'
                },
                # 'FeatureStoreOutput': {
                #     'FeatureGroupName': f'X_{last_transformer}_feature_group'
                # },
                'AppManaged': True
            },
            # y datasets
            {
                'OutputName': 'y-data',
                'S3Output': {
                    'S3Uri': get_data_uri(dataset_name=f'y_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/output/y',
                    'S3UploadMode': 'EndOfJob'
                },
                # 'FeatureStoreOutput': {
                #     'FeatureGroupName': None
                # },
                'AppManaged': True
            }
        ]
    elif 'tuning' in job_name:
        processing_outputs = [
            {
                'OutputName': 'tuning-output',
                'S3Output': {
                    'S3Uri': get_data_uri(dataset_name=f'X_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/output',
                    'S3UploadMode': 'EndOfJob'
                },
                'AppManaged': True
            }
        ]
    elif 'training' in job_name:
        processing_outputs = [
            {
                'OutputName': 'training-output',
                'S3Output': {
                    'S3Uri': get_data_uri(dataset_name=f'X_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/output',
                    'S3UploadMode': 'EndOfJob'
                },
                'AppManaged': True
            }
        ]
    elif 'evaluating' in job_name:
        processing_outputs = [
            {
                'OutputName': 'evaluating-output',
                'S3Output': {
                    'S3Uri': get_data_uri(dataset_name=f'X_{last_transformer}'),
                    'LocalPath': '/opt/ml/processing/output',
                    'S3UploadMode': 'EndOfJob'
                },
                'AppManaged': True
            }
        ]
    else:
        raise NotImplementedError(f"Processing output for {job_name} have not been implemented yet.")
    
    return {'Outputs': processing_outputs}


def get_job_parameters(job_name: str) -> dict:
    """
    # Experiment configuration
    ExperimentConfig={
        'ExperimentName': 'string',
        'TrialName': 'string',
        'TrialComponentDisplayName': 'string',
        'RunName': 'string'
    }

    # Network configuration
    NetworkConfig={
        'EnableInterContainerTrafficEncryption': True|False,
        'EnableNetworkIsolation': True|False,
        'VpcConfig': {
            'SecurityGroupIds': [
                'string',
            ],
            'Subnets': [
                'string',
            ]
        }
    }
    """
    # Define job name
    job_name = define_job_name(job_name=job_name)

    return {
        # Define job name
        'ProcessingJobName': job_name,
        
        # Define processing resources
        'ProcessingResources': get_processing_resources(job_name=job_name),

        # Define stopping condition
        'StoppingCondition': get_stopping_condition(job_name=job_name),

        # Define app specification
        'AppSpecification': get_app_specification(job_name=job_name),

        # Define environment variables
        'Environment': get_environment(), # {k: v for k, v in environmnent.items() if 'KEY' not in k},

        # Find role ARN
        'RoleArn': get_role_arn(),

        # Define tags
        'Tags': get_tags(),

        # Define processing inputs
        'ProcessingInputs': get_processing_inputs(job_name=job_name),

        # Define processing output configuration
        'ProcessingOutputConfig': get_processing_outputs(job_name=job_name)
    }


def run_sagemaker_processing_job(job_name: str) -> dict:
    # Instanciate SageMaker client
    sagemaker_client = boto3.client('sagemaker')

    # Find job parameters
    job_parameters: dict = get_job_parameters(job_name=job_name)

    # Show job details
    LOGGER.info(
        'Running %s job with parameters:\n%s',
        job_name,
        pformat({k: v for k, v in job_parameters.items() if k != 'Environment'})
    )

    # Create Processing Job
    response: dict = sagemaker_client.create_processing_job(**job_parameters)

    # Log response
    LOGGER.info(f"Processing job {job_name} created with response:\n{pformat(response)}")

    return response


"""
source .ml_accel_venv/bin/activate
conda deactivate
.ml_accel_venv/bin/python ml_accelerator/utils/aws/sagemaker/sagemaker_jobs_helper.py --job_name data-processing
"""
if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description='Data processing script.')

    # Add arguments
    parser.add_argument(
        '--job_name', 
        type=str, 
        default='data-processing',
        choices=['data-processing', 'tuning', 'training', 'evaluating'],
    )

    # Extract arguments from parser
    args = parser.parse_args()
    job_name: str = args.job_name

    # Run processing job
    run_sagemaker_processing_job(job_name=job_name)