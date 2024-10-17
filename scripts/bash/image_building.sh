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
ENV=$(yq eval '.ENV_PARAMS.ENV' ${CONFIG_FILE})

# Show variables
# echo "Image building variables:"
# echo "  - VERSION: ${VERSION}"
# echo "  - ENV: ${ENV}"

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
    --build-arg API_KEY_PLACEHOLDER=${API_KEY_PLACEHOLDER}

docker build \
    -t ${ENV}-image:${VERSION} \
    -f docker/Dockerfile . \
    --build-arg VERSION=${VERSION} \
    --build-arg ENV=${ENV}

# login to docker
echo ${DOCKERHUB_TOKEN} | docker login -u ${DOCKERHUB_USERNAME} --password-stdin

# Tag docker images
docker tag ${ENV}-image:${VERSION} ${DOCKERHUB_USERNAME}/${DOCKERHUB_REPOSITORY}:${ENV}-image-${VERSION}

# Push images to repository
docker push ${DOCKERHUB_USERNAME}/${DOCKERHUB_REPOSITORY}:${ENV}-image-${VERSION}

# Pull images from repository
docker pull ${DOCKERHUB_USERNAME}/${DOCKERHUB_REPOSITORY}:${ENV}-image-${VERSION}