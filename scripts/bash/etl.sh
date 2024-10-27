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

elif [ "${ETL_ENV}" == "docker" ]; then
    echo "Running ETL job on docker environment..."

    # Delete current container
    docker rm -f ${ENV}_etl_container_${VERSION}

    # Run docker container
    docker run \
        --name ${ENV}_etl_container_${VERSION} \
        -it \
        --volume ./${BUCKET_NAME}:/app/${BUCKET_NAME} \
        --volume ./config:/app/config \
        --env-file .env \
        ${ENV}-image:${VERSION} \
        python scripts/etl/etl.py

        # -p host_port:container_port or --publish host_port:container_port
        # --log-driver json-file --log-opt max-size=10m
    
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
        --payload '{ "persist_datasets": "True", "write_mode": "overwrite" }' \
        --output json \
        --region ${REGION_NAME} \
        --cli-binary-format raw-in-base64-out \
        ${OUTPUT_FILE}
        # --log-type Tail \
        # --query 'LogResult' \

    echo "Finished running ETL job."

else
    echo "Unable to run etl script - Invalid ETL_ENV: ${ETL_ENV}"
    exit 1
fi