#!/bin/bash
# chmod +x ./scripts/bash/build_infra.sh
# ./scripts/bash/build_infra.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Define terraform variables
export TF_VAR_ENV=${ENV}
export TF_VAR_REGION=${REGION}
export TF_VAR_BUCKET_NAME=${BUCKET_NAME}
export TF_VAR_DOCKER_REPOSITORY_NAME=${DOCKER_REPOSITORY_NAME}

# Check if DOCKER_REPOSITORY_TYPE is equal to "ECR"
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

# Check if DATA_STORAGE_ENV is equal to "S3"
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

# Remove terraform.tfstate
FILE_TO_DELETE="terraform.tfstate"

# Check if the file exists
if [ -f "${FILE_TO_DELETE}" ]; then
    # Delete the file
    rm "${FILE_TO_DELETE}"
    echo "File ${FILE_TO_DELETE} was deleted."
fi