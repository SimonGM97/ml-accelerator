#!/bin/bash
# chmod +x ./scripts/bash/build_infra.sh
# ./scripts/bash/build_infra.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Check if at least one variable is not "local"
if [[ 
    "$DATA_STORAGE_ENV" != "filesystem" || 
    "$MODEL_STORAGE_ENV" != "filesystem" || 
    "$ETL_ENV" != "local" || 
    "$MODEL_BUILDING_ENV" != "local" || 
    "$APP_ENV" != "local" 
]]; then
    echo "Building AWS infrastructure..."

    # Extract variables from config file
    CONFIG_FILE="config/config.yaml"

    PROJECT_NAME=$(yq eval '.PROJECT_PARAMS.PROJECT_NAME' ${CONFIG_FILE})
    VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

    # Extract variables from terraform env
    BUCKET_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param BUCKET_NAME)
    DOCKER_REPOSITORY_NAME=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param DOCKER_REPOSITORY_NAME)
    ECR_REPOSITORY_URI=$(.ml_accel_venv/bin/python ml_accelerator/config/env.py  --env_param ECR_REPOSITORY_URI)

    # Define ETL_LAMBDA_IMAGE_URI
    if [ "${DOCKER_REPOSITORY_TYPE}" == "dockerhub" ]; then
        ETL_LAMBDA_IMAGE_URI=${DOCKER_USERNAME}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}
    elif [ "${DOCKER_REPOSITORY_TYPE}" == "ECR" ]; then
        ETL_LAMBDA_IMAGE_URI=${ECR_REPOSITORY_URI}/${DOCKER_REPOSITORY_NAME}:${ENV}-etl-lambda-image-${VERSION}
    else
        echo "Unable to define IMAGE_NAME - Invalid DOCKER_REPOSITORY_TYPE: ${DOCKER_REPOSITORY_TYPE}"
        exit 1
    fi

    # Show variables
    echo "Extracted variables:"
    echo "  - PROJECT_NAME: ${PROJECT_NAME}"
    echo "  - VERSION: ${VERSION}"
    echo "  - BUCKET_NAME: ${BUCKET_NAME}"
    echo "  - ETL_LAMBDA_IMAGE_URI: ${ETL_LAMBDA_IMAGE_URI}"
    echo ""

    # Define terraform variables
    export TF_VAR_PROJECT_NAME=${PROJECT_NAME}
    export TF_VAR_VERSION=${VERSION}
    export TF_VAR_REGION_NAME=${REGION_NAME}
    export TF_VAR_ETL_LAMBDA_IMAGE_URI=${ETL_LAMBDA_IMAGE_URI}

    # terraform -chdir=terraform/... -var-file="resource.tfvars" init: initialize Terraform & download the necessary provider plugins for AWS.
    # terraform -chdir=terraform/... -var-file="resource.tfvars" validate: validate Terraform configuration before applying it to ensure there are no syntax errors.
    # terraform -chdir=terraform/... -var-file="resource.tfvars" plan: shows what Terraform will do when applying the configuration (won’t make any changes).
    # terraform -chdir=terraform/... -var-file="resource.tfvars" apply: apply the configuration to create the resources.
    # terraform -chdir=terraform/... -var-file="resource.tfvars" destroy: delete all resources created by Terraform.

    # Build S3 bucket
    if [ "${DATA_STORAGE_ENV}" == "S3" ]; then
        if [ "${ENV}" == "prod" ]; then
            echo "Building S3 production bucket with Terraform..."
            echo ""

            # Initialize Terraform
            terraform \
                -chdir=terraform/production/s3 init \
                -compact-warnings \
                -var-file="s3.tfvars"

            # Apply the configurations and create resources
            terraform \
                -chdir=terraform/production/s3 apply \
                -compact-warnings \
                -auto-approve \
                -var-file="s3.tfvars"
        else
            echo "Building S3 development bucket with Terraform..."
            echo ""

            # Initialize Terraform
            terraform \
                -chdir=terraform/development/s3 init \
                -compact-warnings \
                -var-file="s3.tfvars"

            # Apply the configurations and create resources
            terraform \
                -chdir=terraform/development/s3 apply \
                -compact-warnings \
                -auto-approve \
                -var-file="s3.tfvars"
        fi
    fi

    # Build ETL lambda function
    if [ "${ETL_ENV}" == "lambda" ]; then
        if [ "${ENV}" == "prod" ]; then
            echo "Building production ETL lambda function with Terraform..."
            echo ""

            # Initialize Terraform
            terraform \
                -chdir=terraform/production/lambda init \
                -compact-warnings \
                -var-file="lambda.tfvars"

            # Apply the configurations and create resources
            terraform \
                -chdir=terraform/production/lambda apply \
                -compact-warnings \
                -auto-approve \
                -var-file="lambda.tfvars"
        else
            echo "Building development ETL lambda function with Terraform..."
            echo ""

            # Initialize Terraform
            terraform \
                -chdir=terraform/development/lambda init \
                -compact-warnings \
                -var-file="lambda.tfvars"

            # Apply the configurations and create resources
            terraform \
                -chdir=terraform/development/lambda apply \
                -compact-warnings \
                -auto-approve \
                -var-file="lambda.tfvars"
        fi
    fi

    # Build Model Building Step Function
    if [ "${MODEL_BUILDING_ENV}" == "sagemaker" ]; then
        if [ "${ENV}" == "prod" ]; then
            echo "Building production Model Building Step Function with Terraform..."

            # Initialize Terraform
            terraform \
                -chdir=terraform/production/step_functions init \
                -compact-warnings \
                -var-file="step_functions.tfvars"

            # Apply the configurations and create resources
            terraform \
                -chdir=terraform/production/step_functions apply \
                -compact-warnings \
                -auto-approve \
                -var-file="step_functions.tfvars"
        else
            echo "Building development Model Building Step Function with Terraform..."

            # Initialize Terraform
            terraform \
                -chdir=terraform/development/step_functions init \
                -compact-warnings \
                -var-file="step_functions.tfvars"

            # Apply the configurations and create resources
            terraform \
                -chdir=terraform/development/step_functions apply \
                -compact-warnings \
                -auto-approve \
                -var-file="step_functions.tfvars"
        fi
    fi

    # Remove terraform.tfstate
    FILE_TO_DELETE="terraform.tfstate"

    # Check if the file exists
    if [ -f "${FILE_TO_DELETE}" ]; then
        # Delete the file
        rm "${FILE_TO_DELETE}"
        echo "File ${FILE_TO_DELETE} was deleted."
    fi
else
    echo "Skipping infrastructure building..."
    echo ""
fi