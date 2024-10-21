#!/bin/bash
# chmod +x ./scripts/bash/image_building.sh
# ./scripts/bash/image_building.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Extract variables
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

# Show variables
# echo "Image building variables:"
# echo "  - VERSION: ${VERSION}"

# Clean containers
if [ "$(docker ps -aq)" ]; then
    docker rm -f $(docker ps -aq)
fi

# Clean local images
if [ "$(docker images -q)" ]; then
    docker rmi -f $(docker images -q)
fi

# Build Docker images
docker build \
    -t ${ENV}-base-image:${VERSION} \
    -f docker/Dockerfile.Base . \
    --build-arg ENV=${ENV} \
    --build-arg REGION=${REGION} \
    --build-arg BUCKET=${BUCKET} \
    --build-arg DATA_STORAGE_ENV=${DATA_STORAGE_ENV} \
    --build-arg MODEL_STORAGE_ENV=${MODEL_STORAGE_ENV} \
    --build-arg COMPUTE_ENV=${COMPUTE_ENV} \
    --build-arg KXY_API_KEY=${KXY_API_KEY} \
    --build-arg DOCKER_REPOSITORY_TYPE=${DOCKER_REPOSITORY_TYPE} \
    --build-arg DOCKER_USERNAME=${DOCKER_USERNAME} \
    --build-arg DOCKER_REPOSITORY_NAME=${DOCKER_REPOSITORY_NAME} \
    --build-arg DOCKER_TOKEN=${DOCKER_TOKEN} \
    --build-arg INFERENCE_HOST=${INFERENCE_HOST} \
    --build-arg INFERENCE_PORT=${INFERENCE_PORT} \
    --build-arg WEBAPP_HOST=${WEBAPP_HOST} \
    --build-arg WEBAPP_PORT=${WEBAPP_PORT} \
    --build-arg RAW_DATASETS_PATH=${RAW_DATASETS_PATH} \
    --build-arg TRAINING_PATH=${TRAINING_PATH} \
    --build-arg INFERENCE_PATH=${INFERENCE_PATH} \
    --build-arg TRANSFORMERS_PATH=${TRANSFORMERS_PATH} \
    --build-arg MODELS_PATH=${MODELS_PATH} \
    --build-arg SCHEMAS_PATH=${SCHEMAS_PATH} \
    --build-arg MOCK_PATH=${MOCK_PATH} \
    --build-arg SEED=${SEED}

docker build \
    -t ${ENV}-image:${VERSION} \
    -f docker/Dockerfile . \
    --build-arg VERSION=${VERSION} \
    --build-arg ENV=${ENV}

# login to docker
echo ${DOCKER_TOKEN} | docker login -u ${DOCKER_USERNAME} --password-stdin

# Tag docker images
docker tag ${ENV}-image:${VERSION} ${DOCKER_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}

# Push images to repository
docker push ${DOCKER_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}

# Pull images from repository
docker pull ${DOCKER_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}