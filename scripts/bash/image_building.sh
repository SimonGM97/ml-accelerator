#!/bin/bash
# chmod +x ./scripts/bash/image_building.sh
# ./scripts/bash/image_building.sh

# Source the .env file to load environment variables
set -o allexport
source .env
set +o allexport

# Clean containers
if [ "$(docker ps -aq)" ]; then
    docker rm -f $(docker ps -aq)
fi

# Clean local images
if [ "$(docker images -q)" ]; then
    docker rmi -f $(docker images -q)
fi

# Make scripts executable
chmod +x ./scripts/data_processing/data_processing.py

# Build Docker images
docker build \
    -t base-image:${VERSION} \
    -f docker/base/Dockerfile . \
    --build-arg API_KEY_PLACEHOLDER=${API_KEY_PLACEHOLDER}

docker build \
    -t data-processing-image:${VERSION} \
    -f docker/data_processing/Dockerfile . \
    --build-arg VERSION=${VERSION}

# login to docker
echo ${DOCKERHUB_TOKEN} | docker login -u ${DOCKERHUB_USERNAME} --password-stdin

# Tag docker images
docker tag data-processing-image:${VERSION} ${DOCKERHUB_USERNAME}/${DOCKERHUB_REPOSITORY}:data-processing-image-${VERSION}

# Push images to repository
docker push ${DOCKERHUB_USERNAME}/${DOCKERHUB_REPOSITORY}:data-processing-image-${VERSION}

# Pull images from repository
docker pull ${DOCKERHUB_USERNAME}/${DOCKERHUB_REPOSITORY}:data-processing-image-${VERSION}