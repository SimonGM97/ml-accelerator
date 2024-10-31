#!/bin/bash
# chmod +x ./scripts/bash/app.sh
# ./scripts/bash/app.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Unset environment variables
unset VERSION

# Extract variables from config file
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

# Extract variables from terraform env
BUCKET_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param BUCKET_NAME)
DOCKER_REPOSITORY_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param DOCKER_REPOSITORY_NAME)
ECR_REPOSITORY_URI=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param ECR_REPOSITORY_URI)

# Show variables
echo "Extracted variables:"
echo "  - VERSION: ${VERSION}"
echo "  - BUCKET_NAME: ${BUCKET_NAME}"
echo "  - DOCKER_REPOSITORY_NAME: ${DOCKER_REPOSITORY_NAME}"
echo "  - ECR_REPOSITORY_URI: ${ECR_REPOSITORY_URI}"
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

if [ "${APP_ENV}" == "local" ]; then
    # Run app.py
    # .ml_accel_venv/bin/python app.py

    # Run streamlit web app
    streamlit run scripts/web_app/web_app.py \
        --server.port ${WEBAPP_PORT} \
        --server.address ${WEBAPP_HOST}

elif [ "${APP_ENV}" == "docker-compose" ]; then
    # Run docker-compose
    IMAGE_NAME=${IMAGE_NAME} \
        ENV=${ENV} \
        VERSION=${VERSION} \
        BUCKET_NAME=${BUCKET_NAME} \
        docker-compose \
        -f docker/compose/docker-compose-app.yaml \
        --env-file .env \
        up

    # Remove running services
    # VERSION=${VERSION} \
    #     docker-compose \
    #     -f docker/compose/docker-compose-app.yaml \
    #     --env-file .env \
    #     down

elif [ "${APP_ENV}" == "EC2" ]; then
    echo "EC2 APP_ENV environment has not been implemented yet"

else
    echo "Unable to run app script - Invalid APP_ENV: ${APP_ENV}"
    exit 1
fi