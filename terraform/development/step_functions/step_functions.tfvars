# Development step functions variables

SAGEMAKER_EXECUTION_ROLE_NAME           = "SageMakerExecutionRole-MLAccelerator-Dev"
SAGEMAKER_EXECUTION_ROLE_ARN            = "arn:aws:iam::097866913509:role/SageMakerExecutionRole-MLAccelerator-Dev"
MODEL_BUILDING_STEP_FUNCTIONS_NAME      = "ModelBuildingWorkflow_Dev"
MODEL_BUILDING_STEP_FUNCTIONS_ARN       = "arn:aws:states:sa-east-1:097866913509:stateMachine:ModelBuildingWorkflow_Dev"
MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME = "model_building_workflow_dev.json"
STEP_FUNCTIONS_EXECUTION_ROLE_NAME      = "StepFunctionsExecutionRole-MLAccelerator-Dev"
STEP_FUNCTIONS_EXECUTION_ROLE_ARN       = "arn:aws:iam::097866913509:role/StepFunctionsExecutionRole-MLAccelerator-Dev"

SKLEARN_PROCESSOR_FRAMEWORK_VERSION     = "0.23-1"

PROCESSING_INSTANCE_TYPE                = "ml.t3.large"
PROCESSING_INSTANCE_COUNT               = 1
PROCESSING_VOLUME_SIZE                  = 30
PROCESSING_MAX_RUNTIME                  = 300

TUNING_INSTANCE_TYPE                    = "ml.t3.large"
TUNING_INSTANCE_COUNT                   = 1
TUNING_VOLUME_SIZE                      = 30
TUNING_MAX_RUNTIME                      = 300

TRAINING_INSTANCE_TYPE                  = "ml.t3.large"
TRAINING_INSTANCE_COUNT                 = 1
TRAINING_VOLUME_SIZE                    = 30
TRAINING_MAX_RUNTIME                    = 300

EVALUATING_INSTANCE_TYPE                = "ml.t3.large"
EVALUATING_INSTANCE_COUNT               = 1
EVALUATING_VOLUME_SIZE                  = 30
EVALUATING_MAX_RUNTIME                  = 300

INFERENCE_INSTANCE_TYPE                 = "ml.t3.large"
INFERENCE_INSTANCE_COUNT                = 1
INFERENCE_VOLUME_SIZE                   = 30
INFERENCE_MAX_RUNTIME                   = 300

DEFAULT_INSTANCE_TYPE                   = "ml.t3.large"
DEFAULT_INSTANCE_COUNT                  = 1
DEFAULT_VOLUME_SIZE                     = 30
DEFAULT_MAX_RUNTIME                     = 300

# Instance types: https://aws.amazon.com/sagemaker/pricing/instance-types/
# - ml.t3.large:    2 vCPU   8 GiB