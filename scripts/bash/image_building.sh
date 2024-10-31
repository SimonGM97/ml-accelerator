#!/bin/bash
# chmod +x ./scripts/bash/image_building.sh
# ./scripts/bash/image_building.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Check if at least one variable is not "local"
if [[ 
    "$ETL_ENV" != "local" || 
    "$MODEL_BUILDING_ENV" != "local" || 
    "$APP_ENV" != "local" 
]]; then
    echo "Building new ${ENV} - ${VERSION} docker images..."

    # Extract variables from config file
    CONFIG_FILE="config/config.yaml"

    VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

    # Extract variables from terraform env
    BUCKET_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param BUCKET_NAME)
    DOCKER_REPOSITORY_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param DOCKER_REPOSITORY_NAME)
    ECR_REPOSITORY_URI=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param ECR_REPOSITORY_URI)
    ETL_LAMBDA_FUNCTION_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param ETL_LAMBDA_FUNCTION_NAME)

    # Show variables
    echo "Extracted variables:"
    echo "  - VERSION: ${VERSION}"
    echo "  - BUCKET_NAME: ${BUCKET_NAME}"
    echo "  - DOCKER_REPOSITORY_NAME: ${DOCKER_REPOSITORY_NAME}"
    echo "  - ECR_REPOSITORY_URI: ${ECR_REPOSITORY_URI}"
    echo "  - ETL_LAMBDA_FUNCTION_NAME: ${ETL_LAMBDA_FUNCTION_NAME}"
    echo ""

    echo "Cleaning current docker images and containers..."

    # Clean containers
    if [ "$(docker ps -aq)" ]; then
        docker rm -f $(docker ps -aq)
    fi

    # Clean local images
    if [ "$(docker images -q)" ]; then
        docker rmi -f $(docker images -q)
    fi

    # Build base Docker image (compatible with linux/amd64 architecture)
    # docker buildx build \
    docker build \
        --platform linux/amd64 \
        -t ${ENV}-base-image:${VERSION} \
        -f docker/Dockerfile.Base . \
        --load \
        --build-arg ENV=${ENV} \
        --build-arg BUCKET_NAME=${BUCKET_NAME} \
        --build-arg REGION_NAME=${REGION_NAME} \
        --build-arg DATA_STORAGE_ENV=${DATA_STORAGE_ENV} \
        --build-arg MODEL_STORAGE_ENV=${MODEL_STORAGE_ENV} \
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
        --build-arg BUCKET_NAME=${BUCKET_NAME} \
        --build-arg REGION_NAME=${REGION_NAME} \
        --build-arg DATA_STORAGE_ENV=${DATA_STORAGE_ENV} \
        --build-arg MODEL_STORAGE_ENV=${MODEL_STORAGE_ENV} \
        --build-arg KXY_API_KEY=${KXY_API_KEY} \
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

        # Update image being ran by lambda function
        aws lambda update-function-code \
            --function-name ${ETL_LAMBDA_FUNCTION_NAME} \
            --image-uri ${DOCKERHUB_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}

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

        # Update image being ran by lambda function
        aws lambda update-function-code \
            --function-name ${ETL_LAMBDA_FUNCTION_NAME} \
            --image-uri ${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}

    else
        echo "Unable to push docker images - Invalid DOCKER_REPOSITORY_TYPE: ${DOCKER_REPOSITORY_TYPE}"
        exit 1
    fi

    # Wait for lambda function to be updated
    echo "Waiting for lambda function to be updated..."
    sleep 30

    # Delete lambda function
    # aws lambda delete-function --function-name ${ETL_LAMBDA_FUNCTION_NAME}
else
    echo "Skipping docker image building..."
    echo ""
fi