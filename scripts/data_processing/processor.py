from ml_accelerator.config.env import Env
from ml_accelerator.utils.aws.sagemaker.sagemaker_jobs_helper import get_role_arn
import sagemaker
from sagemaker.sklearn.estimator import SKLearn


# .ml_accel_venv/bin/python scripts/data_processing/processor.py
if __name__ == "__main__":
    # Define SageMaker session and role
    sagemaker_session = sagemaker.Session()
    role = get_role_arn()
    model_output_uri = f"s3://{Env.get('BUCKET_NAME')}/{Env.get('TRANSFORMERS_PATH')}"

    # Set up the SKLearn estimator for preprocessing
    sklearn_preprocessor = SKLearn(
        entry_point="scripts/data_processing/data_processing.py",
        framework_version="0.23-1",
        role=role,
        instance_type=Env.get("PROCESSING_INSTANCE_TYPE"), # "ml.m5.large",
        sagemaker_session=sagemaker_session,
        output_path=model_output_uri
    )

    # Fit the preprocessor
    sklearn_preprocessor.fit({"train": f"s3://{Env.get('BUCKET_NAME')}/{Env.get('RAW_DATASETS_PATH')}"})
