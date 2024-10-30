#!/bin/bash
# chmod +x ./scripts/bash/image_pulling.sh
# ./scripts/bash/image_pulling.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Check if at least one variable is not "local"
if [[ 
    "$ETL_ENV" != "local" || 
    "$MODEL_BUILDING_ENV" != "local" || 
    "$APP_ENV" != "local" 
]]; then
    echo "Pulling ${ENV} - ${VERSION} docker images..."

    # Extract variables from config file
    CONFIG_FILE="config/config.yaml"

    VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

    # Extract variables from terraform env
    DOCKER_REPOSITORY_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param DOCKER_REPOSITORY_NAME)
    ECR_REPOSITORY_URI=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param ECR_REPOSITORY_URI)

    # Show variables
    echo "Extracted variables:"
    echo "  - VERSION: ${VERSION}"
    echo "  - PERSIST_DATASETS: ${PERSIST_DATASETS}"
    echo "  - WRITE_MODE: ${WRITE_MODE}"
    echo "  - BUCKET_NAME: ${BUCKET_NAME}"
    echo "  - ETL_LAMBDA_FUNCTION_NAME: ${ETL_LAMBDA_FUNCTION_NAME}"
    echo "  - DOCKER_REPOSITORY_NAME: ${DOCKER_REPOSITORY_NAME}"
    echo "  - ECR_REPOSITORY_URI: ${ECR_REPOSITORY_URI}"
    echo ""

    if [ "${DOCKER_REPOSITORY_TYPE}" == "dockerhub" ]; then
        # Pull images from dockerhub repository
        echo "Pulling images from dockerhub repository..."
        docker pull ${DOCKERHUB_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}
        docker pull ${DOCKERHUB_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}
    elif [ "${DOCKER_REPOSITORY_TYPE}" == "ECR" ]; then
        # Pull images from repository
        echo "Pulling images from ECR repository..."
        docker pull ${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}
        docker pull ${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}
    else
        echo "Unable to pull docker images - Invalid DOCKER_REPOSITORY_TYPE: ${DOCKER_REPOSITORY_TYPE}"
        exit 1
    fi
else
    echo "Skipping image pulling..."
fi