from ml_accelerator.config.env import Env
from ml_accelerator.utils.aws.sagemaker.jobs_helper import (
    get_role_arn,
    get_image_uri,
    get_entrypoint,
    get_volume_size,
    get_max_runtime,
    get_session,
    get_environment,
    get_tags,
    get_inputs,
    get_outputs,
    get_container_arguments
)
from ml_accelerator.utils.logging.logger_helper import get_logger
from sagemaker.processing import Processor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString
)


# Get logger
LOGGER = get_logger(name=__name__)


def find_processing_step_name(job_name: str) -> str:
    return f"{job_name}-{Env.get('ENV')}"


def define_processing_step(
    job_name: str,
    instance_count: ParameterInteger,
    instance_type: ParameterString
) -> ProcessingStep:
    # Extract processing step name
    processing_step_name = find_processing_step_name(job_name=job_name)

    LOGGER.info('processing_step_name: %s', processing_step_name)
    
    # Instanciate Processor
    processor = Processor(
        # Extract role arn
        role=get_role_arn(),
        # Extract image uri
        image_uri=get_image_uri(),
        # Extract instance count
        instance_count=instance_count,
        # Extract instance type
        instance_type=instance_type,
        # Extract entrypoint
        entrypoint=get_entrypoint(job_name=job_name),
        # Extract volume
        volume_size_in_gb=get_volume_size(job_name=job_name),
        # Dummy kms parameters
        volume_kms_key=None,
        output_kms_key=None,
        # Extract max runtime
        max_runtime_in_seconds=get_max_runtime(job_name=job_name),
        # Define job name
        base_job_name=job_name,
        # Extract session
        sagemaker_session=get_session(),
        # Extract environment variables
        env=get_environment(),
        # Extract tags
        tags=get_tags(),
        # Dummy network config
        network_config=None
    )

    # Define ProcessingStep
    processing_step = ProcessingStep(
        name=processing_step_name,
        step_args=None, # not required if passing the processor
        processor=processor,
        display_name=processing_step_name,
        description="",
        inputs=get_inputs(job_name=job_name),
        outputs=get_outputs(job_name=job_name),
        job_arguments=get_container_arguments(job_name=job_name),
        code=None, # defined in the image_uri
        property_files=None, # not required
        cache_config=None, # can be setted to bypass unrequired process where the inputs have not been changed
        depends_on=None,
        retry_policies=None,
        kms_key=None
    )

    LOGGER.info('processing_step: %s', processing_step)

    return processing_step