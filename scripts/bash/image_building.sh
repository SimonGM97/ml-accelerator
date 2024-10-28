#!/bin/bash
# chmod +x ./scripts/bash/image_building.sh
# ./scripts/bash/image_building.sh build_new_images

# Set environment variables
set -o allexport
source .env
set +o allexport

# Extract variables from config file
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

# Show variables
echo "Docker image building variables:"
echo "  - VERSION: ${VERSION}"
echo "  - ENV: ${ENV}"
echo "  - DATA_STORAGE_ENV: ${DATA_STORAGE_ENV}"
echo "  - MODEL_STORAGE_ENV: ${MODEL_STORAGE_ENV}"
echo "  - REGION_NAME: ${REGION_NAME}"
echo "  - BUCKET_NAME: ${BUCKET_NAME}"
# echo "  - KXY_API_KEY: ${KXY_API_KEY}"
echo "  - INFERENCE_HOST: ${INFERENCE_HOST}"
echo "  - INFERENCE_PORT: ${INFERENCE_PORT}"
echo "  - WEBAPP_HOST: ${WEBAPP_HOST}"
echo "  - WEBAPP_PORT: ${WEBAPP_PORT}"
echo "  - RAW_DATASETS_PATH: ${RAW_DATASETS_PATH}"
echo "  - PROCESSING_DATASETS_PATH: ${PROCESSING_DATASETS_PATH}"
echo "  - INFERENCE_PATH: ${INFERENCE}"
echo "  - TRANSFORMERS_PATH: ${TRANSFORMERS_PATH}"
echo "  - MODELS_PATH: ${MODELS_PATH}"
echo "  - SCHEMAS_PATH: ${SCHEMAS_PATH}"
echo "  - MOCK_PATH: ${MOCK_PATH}"
echo "  - SEED: ${SEED}"
echo "  - DOCKER_REPOSITORY_TYPE: ${DOCKER_REPOSITORY_TYPE}"
echo ""

# Clean containers
if [ "$(docker ps -aq)" ]; then
    docker rm -f $(docker ps -aq)
fi

# Clean local images
if [ "$(docker images -q)" ]; then
    docker rmi -f $(docker images -q)
fi

# # Check if at least one variable is not "local"
# if [[ "$ETL_ENV" != "local" || "$MODEL_BUILDING_ENV" != "local" || "$APP_ENV" != "local" ]]; then
#     echo "At least one of the variables (ETL_ENV, MODEL_BUILDING_ENV, or APP_ENV) is not set to 'local'."
#     # Add your code here to execute if the condition is true
# else
#     echo "All variables (ETL_ENV, MODEL_BUILDING_ENV, and APP_ENV) are set to 'local'."
#     # Add your code here if all are "local"
# fi

if [ $1 == "build_new_images" ]; then
    # Show message
    echo "Building new ${ENV} - ${VERSION} docker images..."

    # Build base Docker image (compatible with linux/amd64 architecture)
    # docker buildx build \
    docker build \
        --platform linux/amd64 \
        -t ${ENV}-base-image:${VERSION} \
        -f docker/Dockerfile.Base . \
        --load \
        --build-arg ENV=${ENV} \
        --build-arg DATA_STORAGE_ENV=${DATA_STORAGE_ENV} \
        --build-arg MODEL_STORAGE_ENV=${MODEL_STORAGE_ENV} \
        --build-arg REGION_NAME=${REGION_NAME} \
        --build-arg BUCKET_NAME=${BUCKET_NAME} \
        --build-arg KXY_API_KEY=${KXY_API_KEY} \
        --build-arg INFERENCE_HOST=${INFERENCE_HOST} \
        --build-arg INFERENCE_PORT=${INFERENCE_PORT} \
        --build-arg WEBAPP_HOST=${WEBAPP_HOST} \
        --build-arg WEBAPP_PORT=${WEBAPP_PORT} \
        --build-arg RAW_DATASETS_PATH=${RAW_DATASETS_PATH} \
        --build-arg PROCESSING_DATASETS_PATH=${PROCESSING_DATASETS_PATH} \
        --build-arg INFERENCE_PATH=${INFERENCE_PATH} \
        --build-arg TRANSFORMERS_PATH=${TRANSFORMERS_PATH} \
        --build-arg MODELS_PATH=${MODELS_PATH} \
        --build-arg SCHEMAS_PATH=${SCHEMAS_PATH} \
        --build-arg MOCK_PATH=${MOCK_PATH} \
        --build-arg SEED=${SEED}

    # Build Docker image (compatible with linux/amd64 architecture)
    docker build \
        --platform linux/amd64 \
        -t ${ENV}-image:${VERSION} \
        -f docker/Dockerfile . \
        --load \
        --build-arg VERSION=${VERSION} \
        --build-arg ENV=${ENV}

    # Build lambda Docker images (compatible with linux/amd64 architecture)
    docker build \
        --platform linux/amd64 \
        -t ${ENV}-etl-lambda-image:${VERSION} \
        -f docker/Dockerfile.ETLLambda . \
        --load \
        --build-arg ENV=${ENV} \
        --build-arg DATA_STORAGE_ENV=${DATA_STORAGE_ENV} \
        --build-arg MODEL_STORAGE_ENV=${MODEL_STORAGE_ENV} \
        --build-arg REGION_NAME=${REGION_NAME} \
        --build-arg BUCKET_NAME=${BUCKET_NAME} \
        --build-arg RAW_DATASETS_PATH=${RAW_DATASETS_PATH} \
        --build-arg PROCESSING_DATASETS_PATH=${PROCESSING_DATASETS_PATH} \
        --build-arg INFERENCE_PATH=${INFERENCE_PATH} \
        --build-arg TRANSFORMERS_PATH=${TRANSFORMERS_PATH} \
        --build-arg MODELS_PATH=${MODELS_PATH} \
        --build-arg SCHEMAS_PATH=${SCHEMAS_PATH} \
        --build-arg MOCK_PATH=${MOCK_PATH}

    if [ "${DOCKER_REPOSITORY_TYPE}" == "dockerhub" ]; then
        # login to docker
        echo ${DOCKERHUB_TOKEN} | docker login -u ${DOCKERHUB_USERNAME} --password-stdin

        # Tag docker images
        docker tag ${ENV}-image:${VERSION} ${DOCKERHUB_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}
        docker tag ${ENV}-etl-lambda-image:${VERSION} ${DOCKERHUB_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}

        # Push images to repository
        echo "Pushing images to dockerhub repository..."
        docker push ${DOCKERHUB_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}
        docker push ${DOCKERHUB_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}

    elif [ "${DOCKER_REPOSITORY_TYPE}" == "ECR" ]; then
        # Log-in to ECR
        aws ecr get-login-password --region ${REGION_NAME} | docker login --username AWS --password-stdin ${ECR_REPOSITORY_URI}

        # Delete current ECR images
        DELETE_IMAGE_IDS=$(aws ecr list-images --repository-name $DOCKER_REPOSITORY_NAME --region $REGION_NAME --query 'imageIds[*]' --output json)
        # echo "DELETE_IMAGE_IDS: $DELETE_IMAGE_IDS"
        if [[ $DELETE_IMAGE_IDS != "[]" ]]; then
            echo "Deleting existing images in ECR repository..."
            aws ecr batch-delete-image \
                --repository-name $DOCKER_REPOSITORY_NAME \
                --region $REGION_NAME \
                --image-ids "$DELETE_IMAGE_IDS"
        else
            echo "No images found in ECR repository to delete."
        fi
        
        # Tag docker images
        docker tag ${ENV}-image:${VERSION} ${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}
        docker tag ${ENV}-etl-lambda-image:${VERSION} ${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}

        # Push images to repository
        echo "Pushing images to ECR repository..."
        docker push ${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-image-${VERSION}
        docker push ${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}

    else
        echo "Unable to push docker images - Invalid DOCKER_REPOSITORY_TYPE: ${DOCKER_REPOSITORY_TYPE}"
        exit 1
    fi
else
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
fi