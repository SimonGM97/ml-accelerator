from ml_accelerator.config.env import Env
from ml_accelerator.config.params import Params
from ml_accelerator.utils.pipeline.pipeline_helper import get_image_uri
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.timing.timing_helper import timing

import sagemaker
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TuningStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.processing import ScriptProcessor
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.parameters import ParameterString, ParameterInteger


# Get logger
LOGGER = get_logger(name=__name__)

@timing
def run_pipeline():
    # Define pipeline instance types
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", 
        default_value=Env.get("PROCESSING_INSTANCE_TYPE")
    )
    # tuning_instance_type = ParameterString(
    #     name="TuningInstanceType", 
    #     default_value=Env.get("TUNING_INSTANCE_TYPE")
    # )
    # training_instance_type = ParameterString(
    #     name="TrainingInstanceType", 
    #     default_value=Env.get("TRAINING_INSTANCE_TYPE")
    # )

    # Define image URI
    image_uri = ParameterString(
        name="ImageURI",
        default_value=get_image_uri()
    )

    # Step 1: Data Processing
    processing_step = ProcessingStep(
        name="DataProcessing",
        processor=ScriptProcessor(
            role=Env.get("SAGEMAKER_EXECUTION_ROLE_ARN"),
            image_uri=image_uri,
            command=[
                "python",
                "scripts/data_processing/data_processing.py",
                "--fit_transformers",
                Params.MODEL_BUILDING_PARAMS["FIT_TRANSFORMERS"],
                "--save_transformers",
                Params.MODEL_BUILDING_PARAMS["SAVE_TRANSFORMERS"],
                "--persist_datasets",
                Params.MODEL_BUILDING_PARAMS["PERSIST_DATASETS"],
                "--write_mode",
                Params.MODEL_BUILDING_PARAMS["WRITE_MODE"]
            ],
            instance_type=processing_instance_type,
            instance_count=1
        ),
        # code="scripts/data_processing/data_processing.py"
    )

    # Step 2: Hyperparameter Tuning
    # tuning_step = TuningStep(
    #     name="HyperparameterTuning",
    #     tuner=HyperparameterTuner(
    #         estimator=sagemaker.estimator.Estimator(...),  # Your training image and script
    #         objective_metric_name="accuracy",
    #         hyperparameter_ranges={...},  # Define your hyperparameter search space
    #         max_jobs=10,
    #         max_parallel_jobs=2,
    #         instance_type=tuning_instance_type
    #     )
    # )

    # Step 3: Model Training
    # training_step = TrainingStep(
    #     name="ModelTraining",
    #     estimator=sagemaker.estimator.Estimator(
    #         image_uri="your-ecr-image",  # ECR image for model training
    #         instance_type=training_instance_type
    #     ),
    #     inputs="s3://your-bucket/processed-data/",  # Output from the data processing step
    #     job_name="model-training-job"
    # )

    # Step 4: Model Evaluation
    # evaluation_step = ProcessingStep(
    #     name="ModelEvaluation",
    #     processor=ScriptProcessor(
    #         image_uri="your-ecr-image",  # ECR image for evaluation
    #         command=["python3"],
    #         instance_type="ml.m5.large",
    #         instance_count=1
    #     ),
    #     code="scripts/evaluating.py"
    # )

    # Combine steps into a pipeline
    pipeline = Pipeline(
        name="ModelBuildingPipeline",
        steps=[processing_step] # , tuning_step, training_step, evaluation_step]
    )

    # Create the pipeline
    LOGGER.info("Creating ModelBuildingPipeline...")
    pipeline.create(
        role_arn=Env.get("SAGEMAKER_EXECUTION_ROLE_ARN"),
        description="Model Building Pipeline",
        tags={
            "Project": Params.PROJECT_NAME,
            "Version": Params.VERSION,
            "Environment": Env.get("ENV")
        }
    )

    # Start the pipeline
    LOGGER.info("Starting ModelBuildingPipeline...")
    pipeline.start(execution_display_name="ModelBuildingPipelineExecution")



# source .ml_accel_venv/bin/activate
# conda deactivate
# .ml_accel_venv/bin/python pipelines/model_building/sagemaker/model_building_pipeline.py
if __name__ == "__main__":
    # Run pipeline
    run_pipeline()