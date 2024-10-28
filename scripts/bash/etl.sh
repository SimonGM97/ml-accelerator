#!/bin/bash
# chmod +x ./scripts/bash/etl.sh
# ./scripts/bash/etl.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Extract variables from config file
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})
PERSIST_DATASETS=$(yq eval '.MODEL_BUILDING_PARAMS.PERSIST_DATASETS' ${CONFIG_FILE})
WRITE_MODE=$(yq eval '.MODEL_BUILDING_PARAMS.WRITE_MODE' ${CONFIG_FILE})

# Show variables
echo "ETL variables:"
echo "  - VERSION: ${VERSION}"
echo "  - ENV: ${ENV}"
echo "  - BUCKET_NAME: ${BUCKET_NAME}"
echo "  - PERSIST_DATASETS: ${PERSIST_DATASETS}"
echo "  - WRITE_MODE: ${WRITE_MODE}"
echo "  - ETL_ENV: ${ETL_ENV}"
echo ""

if [ "${ETL_ENV}" == "local" ]; then
    echo "Running ETL job on local environment..."

    # Run etl.py script
    .ml_accel_venv/bin/python scripts/etl/etl.py \
        --persist_datasets ${PERSIST_DATASETS} \
        --write_mode ${WRITE_MODE}

elif [ "${ETL_ENV}" == "docker-compose" ]; then
    echo "Running ETL job on docker environment..."

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

    # Run docker-compose
    IMAGE_NAME=${IMAGE_NAME} \
        VERSION=${VERSION} \
        PERSIST_DATASETS=${PERSIST_DATASETS} \
        WRITE_MODE=${WRITE_MODE} \
        docker-compose \
        -f docker/compose/docker-compose-etl.yaml \
        --env-file .env \
        up

    # Remove running services
    IMAGE_NAME=${IMAGE_NAME} \
        VERSION=${VERSION} \
        PERSIST_DATASETS=${PERSIST_DATASETS} \
        WRITE_MODE=${WRITE_MODE} \
        docker-compose \
        -f docker/compose/docker-compose-etl.yaml \
        --env-file .env \
        down
    
    echo "Finished running ETL job."

elif [ "${ETL_ENV}" == "lambda" ]; then
    echo "Running ETL job on lambda environment..."

    # Delete output_file
    OUTPUT_FILE="etl_output.json"
    rm "${OUTPUT_FILE}"

    # Define payload
    if [ "${PERSIST_DATASETS}" == "True" ]; then
        if [ "${WRITE_MODE}" == "overwrite" ]; then
            PAYLOAD='{ "persist_datasets": "True", "write_mode": "overwrite" }'
        else
            PAYLOAD='{ "persist_datasets": "True", "write_mode": "append" }'
        fi
    else
        if [ "${WRITE_MODE}" == "overwrite" ]; then
            PAYLOAD='{ "persist_datasets": "False", "write_mode": "overwrite" }'
        else
            PAYLOAD='{ "persist_datasets": "False", "write_mode": "append" }'
        fi
    fi

    # Show payload
    echo "Payload: ${PAYLOAD}"

    # Invoke lambda function synchronomously
    aws lambda invoke \
        --function-name ${ETL_LAMBDA_FUNCTION_NAME} \
        --invocation-type RequestResponse \
        --payload "${PAYLOAD}" \
        --region ${REGION_NAME} \
        --cli-binary-format raw-in-base64-out \
        --output json \
        ${OUTPUT_FILE}
         
        # --log-type Tail \
        # --query 'LogResult' \

    # Remove etl_output.json
    FILE_TO_DELETE="etl_output.json"

    # Check if the file exists
    if [ -f "${FILE_TO_DELETE}" ]; then
        # Delete the file
        rm "${FILE_TO_DELETE}"
        echo "File ${FILE_TO_DELETE} was deleted."
    fi

    echo "Finished running ETL job."

else
    echo "Unable to run etl script - Invalid ETL_ENV: ${ETL_ENV}"
    exit 1
fi