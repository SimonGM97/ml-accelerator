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
    # Run etl.py script
    .ml_accel_venv/bin/python scripts/etl/etl.py \
        --persist_datasets ${PERSIST_DATASETS} \
        --write_mode ${WRITE_MODE}

elif [ "${ETL_ENV}" == "docker" ]; then
    # Run docker container
    docker run \
        --name ${ENV}_etl_container_${VERSION} \
        --volume /${BUCKET_NAME}:/app/${BUCKET_NAME} \
        --volume /config:/app/config \
        --env-file .env

        # -p host_port:container_port or --publish host_port:container_port
        # --log-driver json-file --log-opt max-size=10m

elif [ "${ETL_ENV}" == "lambda" ]; then
    echo "Unable to run etl script - ETL_ENV ${ETL_ENV} is not yet implemented."
    exit 1

else
    echo "Unable to run etl script - Invalid ETL_ENV: ${ETL_ENV}"
    exit 1
fi