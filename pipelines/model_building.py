from ml_accelerator.config.env import Env
from ml_accelerator.utils.aws.sagemaker.jobs_helper import (
    get_instance_count,
    get_instance_type,
    get_script_arg,
    get_role_arn, 
    get_tags
)
from ml_accelerator.utils.aws.sagemaker.processing_step_helper import define_processing_step
from ml_accelerator.utils.aws.sagemaker.pipeline_helper import (
    find_pipeline_session,
    find_pipeline_name,
    find_pipeline_desc,
    find_execution_display_name,
    pipeline_exists
)
from ml_accelerator.utils.logging.logger_helper import get_logger
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep


# Get logger
LOGGER = get_logger(name=__name__)

# .ml_accel_venv/bin/python pipelines/model_building.py
if __name__ == "__main__":
    # Define pipeline parameters
    pipeline_session = find_pipeline_session()
    pipeline_name = find_pipeline_name(pipeline_name='model-building')
    pipeline_desc = find_pipeline_desc(pipeline_name='model-building')
    execution_display_name = find_execution_display_name(pipeline_name=pipeline_name)

    # Define data processing step
    dp_instance_count = get_instance_count(job_name='data-processing')
    dp_instance_type = get_instance_type(job_name='data-processing')

    # fit_transformers = get_script_arg(pipeline_name, arg_name='fit_transformers')
    # save_transformers = get_script_arg(pipeline_name, arg_name='save_transformers')
    # persist_datasets = get_script_arg(pipeline_name, arg_name='persist_datasets')
    # write_mode = get_script_arg(pipeline_name, arg_name='write_mode')

    data_processing_step: ProcessingStep = define_processing_step(
        job_name='data-processing',
        instance_count=dp_instance_count,
        instance_type=dp_instance_type
    )

    # Define tuning step
    tuning_instance_count = get_instance_count(job_name='tuning')
    tuning_instance_type = get_instance_type(job_name='tuning')

    tuning_step: ProcessingStep = define_processing_step(
        job_name='tuning',
        instance_count=tuning_instance_count,
        instance_type=tuning_instance_type
    )
    
    # Define training step
    training_instance_count = get_instance_count(job_name='training')
    training_instance_type = get_instance_type(job_name='training')

    # train_prod_pipe = get_script_arg(pipeline_name, arg_name='train_prod_pipe')
    # train_staging_pipes = get_script_arg(pipeline_name, arg_name='train_staging_pipes')

    training_step: ProcessingStep = define_processing_step(
        job_name='training',
        instance_count=training_instance_count,
        instance_type=training_instance_type
    )

    # Define evaluating step
    eval_instance_count = get_instance_count(job_name='evaluating')
    eval_instance_type = get_instance_type(job_name='evaluating')

    # evaluate_prod_pipe = get_script_arg(pipeline_name, arg_name='evaluate_prod_pipe')
    # evaluate_staging_pipes = get_script_arg(pipeline_name, arg_name='evaluate_staging_pipes')
    # evaluate_dev_pipes = get_script_arg(pipeline_name, arg_name='evaluate_dev_pipes')
    # update_model_stages = get_script_arg(pipeline_name, arg_name='update_model_stages')
    # update_prod_model = get_script_arg(pipeline_name, arg_name='update_prod_model')

    evaluating_step: ProcessingStep = define_processing_step(
        job_name='evaluating',
        instance_count=eval_instance_count,
        instance_type=eval_instance_type,
    )

    # Add dependecies
    tuning_step.add_depends_on([data_processing_step])
    training_step.add_depends_on([tuning_step])
    evaluating_step.add_depends_on([training_step])
    
    # Define Pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            dp_instance_count, dp_instance_type,
            tuning_instance_count, tuning_instance_type,
            training_instance_count, training_instance_type,
            eval_instance_count, eval_instance_type,
        ],
        steps=[
            data_processing_step,
            tuning_step,
            training_step,
            evaluating_step
        ],
        sagemaker_session=pipeline_session,
    )

    # Create & register the Pipeline (if it does not already exist)
    if not pipeline_exists(pipeline_name=pipeline_name):
        pipeline.create(
            role_arn=get_role_arn(),
            description=pipeline_desc,
            tags=get_tags(),
            parallelism_config=None
        )

    # Run Pipeline
    execution = pipeline.start(
        execution_display_name=execution_display_name
    )

    # Monitor execution status
    execution.wait()