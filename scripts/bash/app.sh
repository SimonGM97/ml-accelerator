#!/bin/bash
# chmod +x ./scripts/bash/model_building_workflow.sh
# ./scripts/bash/model_building_workflow.sh

# Unset environment variables
unset VERSION ENV BUCKET INFERENCE_PORT

# Extract variables
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})
ENV=$(yq eval '.ENV_PARAMS.ENV' ${CONFIG_FILE})
BUCKET=$(yq eval '.ENV_PARAMS.BUCKET' ${CONFIG_FILE})
INFERENCE_PORT=$(yq eval '.DEPLOYMENT_PARAMS.INFERENCE_PORT' ${CONFIG_FILE})

# Show variables
echo "Model Building Workflow variables:"
echo "  - VERSION: ${VERSION}"
echo "  - ENV: ${ENV}"
echo "  - BUCKET: ${BUCKET}"
echo "  - INFERENCE_PORT: ${INFERENCE_PORT}"
echo ""

# Clean containers
if [ "$(docker ps -aq)" ]; then
    docker rm -f $(docker ps -aq)
fi

# Run docker-compose
VERSION=${VERSION} \
    ENV=${ENV} \
    BUCKET=${BUCKET} \
    INFERENCE_PORT=${INFERENCE_PORT} \
    docker-compose \
    -f docker/docker-compose-app.yaml \
    --env-file .env \
    up

# Remove running services
# VERSION=${VERSION} \
#     ENV=${ENV} \
#     BUCKET=${BUCKET} \
#     INFERENCE_PORT=${INFERENCE_PORT} \
#     docker-compose \
#     -f docker/docker-compose-app.yaml \
#     --env-file .env \
#     down