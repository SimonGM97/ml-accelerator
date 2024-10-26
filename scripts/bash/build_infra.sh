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

# Show variables
echo "Infrastructure variables:"
echo "  - PROJECT_NAME: ${PROJECT_NAME}"
echo "  - VERSION: ${VERSION}"
echo "  - ENV: ${ENV}"
echo "  - REGION: ${REGION}"
echo "  - BUCKET_NAME: ${BUCKET_NAME}"
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
export TF_VAR_REGION=${REGION}
export TF_VAR_BUCKET_NAME=${BUCKET_NAME}
export TF_VAR_SAGEMAKER_EXECUTION_ROLE_NAME=${SAGEMAKER_EXECUTION_ROLE_NAME}
export TF_VAR_DOCKER_REPOSITORY_NAME=${DOCKER_REPOSITORY_NAME}

# Build SageMaker execution role
if [ "${ETL_ENV}" == "lambda" ] || [ "${MODEL_BUILDING_ENV}" == "sagemaker" ]; then
    echo "Building execution roles with Terraform..."
    echo ""
    
    # Initialize Terraform
    terraform -chdir=terraform/iam init
    
    # Validate Terraform configuration
    # terraform -chdir=terraform/iam validate

    # Shows what Terraform will apply
    # terraform -chdir=terraform/iam plan

    # Apply the configurations and create resources
    terraform -chdir=terraform/iam apply -auto-approve

    # Delete all resources created by terraform
    # terraform -chdir=terraform/iam destroy -auto-approve
fi

# Build S3 bucket
if [ "${DATA_STORAGE_ENV}" == "S3" ]; then
    echo "DATA_STORAGE_ENV is set to S3, running Terraform commands for S3..."
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
fi

# Build ECR repository
if [ "${DOCKER_REPOSITORY_TYPE}" == "ECR" ]; then
    echo "DOCKER_REPOSITORY_TYPE is set to ECR, running Terraform commands for ECR..."
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