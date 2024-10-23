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
    terraform destroy -auto-approve
fi

