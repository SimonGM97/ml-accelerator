#!/bin/bash
# chmod +x ./scripts/bash/model_building_workflow.sh
# ./scripts/bash/model_building_workflow.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Unset environment variables
unset VERSION

# Extract variables
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

# Show variables
echo "Model Building Workflow variables:"
echo "  - VERSION: ${VERSION}"
echo ""

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
    docker-compose \
    -f docker/docker-compose-app.yaml \
    --env-file .env \
    up

# Remove running services
# VERSION=${VERSION} \
#     docker-compose \
#     -f docker/docker-compose-app.yaml \
#     --env-file .env \
#     down