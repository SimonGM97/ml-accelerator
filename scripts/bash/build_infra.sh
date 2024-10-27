#!/bin/bash
# chmod +x ./scripts/bash/build_infra.sh
# ./scripts/bash/build_infra.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Extract variables from config file
CONFIG_FILE="config/config.yaml"

PROJECT_NAME=$(yq eval '.PROJECT_PARAMS.PROJECT_NAME' ${CONFIG_FILE})
VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

# Define ETL_LAMBDA_IMAGE_URI
if [ "${DOCKER_REPOSITORY_TYPE}" == "dockerhub" ]; then
    ETL_LAMBDA_IMAGE_URI=${DOCKER_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}
elif [ "${DOCKER_REPOSITORY_TYPE}" == "ECR" ]; then
    ETL_LAMBDA_IMAGE_URI=${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}
else
    echo "Unable to define IMAGE_NAME - Invalid DOCKER_REPOSITORY_TYPE: ${DOCKER_REPOSITORY_TYPE}"
    exit 1
fi

# Define ETL_LAMBDA_LOG_GROUP
ETL_LAMBDA_LOG_GROUP="/aws/lambda/${ETL_LAMBDA_FUNCTION_NAME}"

# Show variables
echo "Infrastructure variables:"
echo "  - PROJECT_NAME: ${PROJECT_NAME}"
echo "  - VERSION: ${VERSION}"
echo "  - ENV: ${ENV}"
echo "  - REGION_NAME: ${REGION_NAME}"
echo "  - BUCKET_NAME: ${BUCKET_NAME}"
echo ""
# echo "  - SECRET_ARN: ${SECRET_ARN}"
echo "  - LAMBDA_EXECUTION_ROLE_NAME: ${LAMBDA_EXECUTION_ROLE_NAME}"
echo "  - ETL_LAMBDA_FUNCTION_NAME: ${ETL_LAMBDA_FUNCTION_NAME}"
echo "  - ETL_LAMBDA_IMAGE_URI: ${ETL_LAMBDA_IMAGE_URI}"
echo "  - ETL_LAMBDA_LOG_GROUP: ${ETL_LAMBDA_LOG_GROUP}"
echo "  - ETL_LAMBDA_FUNCTION_MEMORY_SIZE: ${ETL_LAMBDA_FUNCTION_MEMORY_SIZE}"
echo "  - ETL_LAMBDA_FUNCTION_TIMEOUT: ${ETL_LAMBDA_FUNCTION_TIMEOUT}"
echo "  - SAGEMAKER_EXECUTION_ROLE_NAME: ${SAGEMAKER_EXECUTION_ROLE_NAME}"
echo "  - DOCKER_REPOSITORY_NAME: ${DOCKER_REPOSITORY_NAME}"
echo "  - DATA_STORAGE_ENV: ${DATA_STORAGE_ENV}"
echo "  - ETL_ENV: ${ETL_ENV}"
echo "  - MODEL_BUILDING_ENV: ${MODEL_BUILDING_ENV}"
echo "  - DOCKER_REPOSITORY_TYPE: ${DOCKER_REPOSITORY_TYPE}"
echo ""

# Define terraform variables
export TF_VAR_PROJECT_NAME=${PROJECT_NAME}
export TF_VAR_VERSION=${VERSION}
export TF_VAR_ENV=${ENV}
export TF_VAR_REGION_NAME=${REGION_NAME}
export TF_VAR_BUCKET_NAME=${BUCKET_NAME}

export TF_VAR_SECRET_ARN=${SECRET_ARN}
export TF_VAR_LAMBDA_EXECUTION_ROLE_NAME=${LAMBDA_EXECUTION_ROLE_NAME}
export TF_VAR_ETL_LAMBDA_FUNCTION_NAME=${ETL_LAMBDA_FUNCTION_NAME}
export TF_VAR_ETL_LAMBDA_IMAGE_URI=${ETL_LAMBDA_IMAGE_URI}
export TF_VAR_ETL_LAMBDA_LOG_GROUP=${ETL_LAMBDA_LOG_GROUP}
export TF_VAR_ETL_LAMBDA_FUNCTION_MEMORY_SIZE=${ETL_LAMBDA_FUNCTION_MEMORY_SIZE}
export TF_VAR_ETL_LAMBDA_FUNCTION_TIMEOUT=${ETL_LAMBDA_FUNCTION_TIMEOUT}
export TF_VAR_SAGEMAKER_EXECUTION_ROLE_NAME=${SAGEMAKER_EXECUTION_ROLE_NAME}

# Build S3 bucket
if [ "${DATA_STORAGE_ENV}" == "S3" ]; then
    echo "Building S3 bucket with Terraform..."
    echo ""

    # Initialize Terraform
    terraform -chdir=terraform/s3 init
    
    # Validate Terraform configuration
    # terraform -chdir=terraform/s3 validate

    # Shows what Terraform will apply
    # terraform -chdir=terraform/s3 plan

    # Apply the configurations and create resources
    terraform -chdir=terraform/s3 apply -auto-approve

    # Delete all resources created by terraform
    # terraform -chdir=terraform/s3 destroy -auto-approve
fi

# Build ETL lambda function
if [ "${ETL_ENV}" == "lambda" ]; then
    echo "Building ETL lambda function with Terraform..."
    echo ""

    # Initialize Terraform
    terraform -chdir=terraform/lambda init

    # Delete lambda function
    aws lambda delete-function --function-name ${ETL_LAMBDA_FUNCTION_NAME}

    # Validate Terraform configuration
    # terraform -chdir=terraform/lambda validate

    # Shows what Terraform will apply
    # terraform -chdir=terraform/lambda plan

    # Apply the configurations and create resources
    terraform -chdir=terraform/lambda apply -auto-approve

    # Destroy all resources created by terraform
    # terraform -chdir=terraform/lambda destroy -auto-approve
fi

# Build Model Building Step Function
if [ "${MODEL_BUILDING_ENV}" == "sagemaker" ]; then
    echo "Building Model Building Step Function with Terraform..."
    echo ""

# Build ECR repository
if [ "${DOCKER_REPOSITORY_TYPE}" == "ECR" ]; then
    echo "Building ECR repository with Terraform..."
    echo ""

    # Initialize Terraform
    terraform -chdir=terraform/ecr init
    
    # Validate Terraform configuration
    # terraform -chdir=terraform/ecr validate

    # Shows what Terraform will apply
    # terraform -chdir=terraform/ecr plan

    # Apply the configurations and create resources
    terraform -chdir=terraform/ecr apply -auto-approve

    # Delete all resources created by terraform
    # terraform -chdir=terraform/ecr destroy -auto-approve
fi

# Remove terraform.tfstate
FILE_TO_DELETE="terraform.tfstate"

# Check if the file exists
if [ -f "${FILE_TO_DELETE}" ]; then
    # Delete the file
    rm "${FILE_TO_DELETE}"
    echo "File ${FILE_TO_DELETE} was deleted."
fi