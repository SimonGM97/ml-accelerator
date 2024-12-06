#!/bin/bash
# chmod +x ./scripts/bash/model_building.sh
# ./scripts/bash/model_building.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Extract variables
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

FIT_TRANSFORMERS=$(yq eval '.MODEL_BUILDING_PARAMS.FIT_TRANSFORMERS' ${CONFIG_FILE})
SAVE_TRANSFORMERS=$(yq eval '.MODEL_BUILDING_PARAMS.SAVE_TRANSFORMERS' ${CONFIG_FILE})
PERSIST_DATASETS=$(yq eval '.MODEL_BUILDING_PARAMS.PERSIST_DATASETS' ${CONFIG_FILE})
WRITE_MODE=$(yq eval '.MODEL_BUILDING_PARAMS.WRITE_MODE' ${CONFIG_FILE})

TRAIN_PROD_PIPE=$(yq eval '.MODEL_BUILDING_PARAMS.TRAIN_PROD_PIPE' ${CONFIG_FILE})
TRAIN_STAGING_PIPES=$(yq eval '.MODEL_BUILDING_PARAMS.TRAIN_STAGING_PIPES' ${CONFIG_FILE})
TRAIN_DEV_PIPES=$(yq eval '.MODEL_BUILDING_PARAMS.TRAIN_DEV_PIPES' ${CONFIG_FILE})

EVALUATE_PROD_PIPE=$(yq eval '.MODEL_BUILDING_PARAMS.EVALUATE_PROD_PIPE' ${CONFIG_FILE})
EVALUATE_STAGING_PIPES=$(yq eval '.MODEL_BUILDING_PARAMS.EVALUATE_STAGING_PIPES' ${CONFIG_FILE})
EVALUATE_DEV_PIPES=$(yq eval '.MODEL_BUILDING_PARAMS.EVALUATE_DEV_PIPES' ${CONFIG_FILE})
UPDATE_MODEL_STAGES=$(yq eval '.MODEL_BUILDING_PARAMS.UPDATE_MODEL_STAGES' ${CONFIG_FILE})
UPDATE_PROD_MODEL=$(yq eval '.MODEL_BUILDING_PARAMS.UPDATE_PROD_MODEL' ${CONFIG_FILE})

# Extract variables from terraform env
BUCKET_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param BUCKET_NAME)
DOCKER_REPOSITORY_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param DOCKER_REPOSITORY_NAME)
ECR_REPOSITORY_URI=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param ECR_REPOSITORY_URI)
MODEL_BUILDING_STEP_FUNCTIONS_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param MODEL_BUILDING_STEP_FUNCTIONS_NAME)
MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME)
MODEL_BUILDING_STEP_FUNCTIONS_ARN=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param MODEL_BUILDING_STEP_FUNCTIONS_ARN)

# Show variables
echo "Extracted variables:"
echo "  - VERSION: ${VERSION}"

echo "  - FIT_TRANSFORMERS: ${FIT_TRANSFORMERS}"
echo "  - SAVE_TRANSFORMERS: ${SAVE_TRANSFORMERS}"
echo "  - PERSIST_DATASETS: ${PERSIST_DATASETS}"
echo "  - WRITE_MODE: ${WRITE_MODE}"

echo "  - TRAIN_PROD_PIPE: ${TRAIN_PROD_PIPE}"
echo "  - TRAIN_STAGING_PIPES: ${TRAIN_STAGING_PIPES}"
echo "  - TRAIN_DEV_PIPES: ${TRAIN_DEV_PIPES}"

echo "  - EVALUATE_PROD_PIPE: ${EVALUATE_PROD_PIPE}"
echo "  - EVALUATE_STAGING_PIPES: ${EVALUATE_STAGING_PIPES}"
echo "  - EVALUATE_DEV_PIPES: ${EVALUATE_DEV_PIPES}"
echo "  - UPDATE_MODEL_STAGES: ${UPDATE_MODEL_STAGES}"
echo "  - UPDATE_PROD_MODEL: ${UPDATE_PROD_MODEL}"

echo "  - BUCKET_NAME: ${BUCKET_NAME}"
echo "  - DOCKER_REPOSITORY_NAME: ${DOCKER_REPOSITORY_NAME}"
echo "  - ECR_REPOSITORY_URI: ${ECR_REPOSITORY_URI}"
echo "  - MODEL_BUILDING_STEP_FUNCTIONS_NAME: ${MODEL_BUILDING_STEP_FUNCTIONS_NAME}"
echo "  - MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME: ${MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME}"
echo "  - MODEL_BUILDING_STEP_FUNCTIONS_ARN: ${MODEL_BUILDING_STEP_FUNCTIONS_ARN}"
echo ""

if [ "${MODEL_BUILDING_ENV}" == "local" ]; then
    echo "Starting model-building workflow in local environment."

    # Run data_processing.py script
    .ml_accel_venv/bin/python scripts/data_processing/data_processing.py \
        --fit_transformers ${FIT_TRANSFORMERS} \
        --save_transformers ${SAVE_TRANSFORMERS} \
        --persist_datasets ${PERSIST_DATASETS} \
        --write_mode ${WRITE_MODE}

    # Run tuning.py script
    .ml_accel_venv/bin/python scripts/tuning/tuning.py

    # Run training.py script
    .ml_accel_venv/bin/python scripts/training/training.py \
        --train_prod_pipe ${TRAIN_PROD_PIPE} \
        --train_staging_pipes ${TRAIN_STAGING_PIPES} \
        --train_dev_pipes ${TRAIN_DEV_PIPES}
    
    # Run evaluating.py script
    .ml_accel_venv/bin/python scripts/evaluating/evaluating.py \
        --evaluate_prod_pipe ${EVALUATE_PROD_PIPE} \
        --evaluate_staging_pipes ${EVALUATE_STAGING_PIPES} \
        --evaluate_dev_pipes ${EVALUATE_DEV_PIPES} \
        --update_model_stages ${UPDATE_MODEL_STAGES} \
        --update_prod_model ${UPDATE_PROD_MODEL}

elif [ "${MODEL_BUILDING_ENV}" == "docker-compose" ]; then
    # Clean containers
    if [ "$(docker ps -aq)" ]; then
        docker rm -f $(docker ps -aq)
    fi

    # Define IMAGE_NAME
    if [ "${DOCKER_REPOSITORY_TYPE}" == "dockerhub" ]; then
        IMAGE_NAME=${DOCKER_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}
    elif [ "${DOCKER_REPOSITORY_TYPE}" == "ECR" ]; then
        IMAGE_NAME=${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}
    else
        echo "Unable to define IMAGE_NAME - Invalid DOCKER_REPOSITORY_TYPE: ${DOCKER_REPOSITORY_TYPE}"
        exit 1
    fi

    echo "Starting model-building workflow in docker-compose environment."

    # Run docker-compose
    IMAGE_NAME=${IMAGE_NAME} \
        VERSION=${VERSION} \
        FIT_TRANSFORMERS=${FIT_TRANSFORMERS} \
        SAVE_TRANSFORMERS=${SAVE_TRANSFORMERS} \
        PERSIST_DATASETS=${PERSIST_DATASETS} \
        WRITE_MODE=${WRITE_MODE} \
        TRAIN_PROD_PIPE=${TRAIN_PROD_PIPE} \
        TRAIN_STAGING_PIPES=${TRAIN_STAGING_PIPES} \
        TRAIN_DEV_PIPES=${TRAIN_DEV_PIPES} \
        EVALUATE_PROD_PIPE=${EVALUATE_PROD_PIPE} \
        EVALUATE_STAGING_PIPES=${EVALUATE_STAGING_PIPES} \
        EVALUATE_DEV_PIPES=${EVALUATE_DEV_PIPES} \
        UPDATE_MODEL_STAGES=${UPDATE_MODEL_STAGES} \
        UPDATE_PROD_MODEL=${UPDATE_PROD_MODEL} \
        BUCKET_NAME=${BUCKET_NAME} \
        docker-compose \
        -f docker/compose/docker-compose-model-building.yaml \
        --env-file .env \
        up

    # Remove running services
    IMAGE_NAME=${IMAGE_NAME} \
        VERSION=${VERSION} \
        FIT_TRANSFORMERS=${FIT_TRANSFORMERS} \
        SAVE_TRANSFORMERS=${SAVE_TRANSFORMERS} \
        PERSIST_DATASETS=${PERSIST_DATASETS} \
        WRITE_MODE=${WRITE_MODE} \
        TRAIN_PROD_PIPE=${TRAIN_PROD_PIPE} \
        TRAIN_STAGING_PIPES=${TRAIN_STAGING_PIPES} \
        TRAIN_DEV_PIPES=${TRAIN_DEV_PIPES} \
        EVALUATE_PROD_PIPE=${EVALUATE_PROD_PIPE} \
        EVALUATE_STAGING_PIPES=${EVALUATE_STAGING_PIPES} \
        EVALUATE_DEV_PIPES=${EVALUATE_DEV_PIPES} \
        UPDATE_MODEL_STAGES=${UPDATE_MODEL_STAGES} \
        UPDATE_PROD_MODEL=${UPDATE_PROD_MODEL} \
        BUCKET_NAME=${BUCKET_NAME} \
        docker-compose \
        -f docker/compose/docker-compose-model-building.yaml \
        --env-file .env \
        down

elif [ "${MODEL_BUILDING_ENV}" == "sagemaker" ]; then
    echo "Starting model-building workflow in SageMaker environment."

    # Run model_building.py script
    .ml_accel_venv/bin/python pipelines/model_building.py

elif [ "${MODEL_BUILDING_ENV}" == "deprecated" ]; then
    echo "Starting model-building workflow in SageMaker environment."

    # Update step functions json file
    .ml_accel_venv/bin/python ml_accelerator/utils/aws/step_functions_helper.py

    # Push new json file to step function
    if [ "${ENV}" == "prod" ]; then
        DEFINITION_FILE=file://terraform/production/step_functions/${MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME}
    else
        DEFINITION_FILE=file://terraform/development/step_functions/${MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME}
    fi
    
    aws stepfunctions update-state-machine \
        --state-machine-arn ${MODEL_BUILDING_STEP_FUNCTIONS_ARN} \
        --definition ${DEFINITION_FILE}

    # Generate a unique name using timestamp to avoid "ExecutionAlreadyExists" error
    EXECUTION_NAME="${MODEL_BUILDING_STEP_FUNCTIONS_NAME}_$(date +%s)"

    # Trigger model building step function with a unique execution name
    EXECUTION_ARN=$(aws stepfunctions start-execution \
        --state-machine-arn ${MODEL_BUILDING_STEP_FUNCTIONS_ARN} \
        --name ${EXECUTION_NAME} \
        --output text \
        --query 'executionArn')
    
    echo "Started Step Function execution with ARN: ${EXECUTION_ARN}"

    # Check execution status
    aws stepfunctions describe-execution --execution-arn "${EXECUTION_ARN}"

else
    echo "Unable to run model-building workflow - Invalid MODEL_BUILDING_ENV: ${MODEL_BUILDING_ENV}"
    exit 1
fi