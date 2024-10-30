#!/bin/bash
# chmod +x ./scripts/bash/build_infra.sh
# ./scripts/bash/build_infra.sh

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
    echo "Building docker repository..."

    # Extract variables from config file
    CONFIG_FILE="config/config.yaml"

    PROJECT_NAME=$(yq eval '.PROJECT_PARAMS.PROJECT_NAME' ${CONFIG_FILE})
    VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})


    # Show variables
    echo "Docker repository variables:"
    echo "  - PROJECT_NAME: ${PROJECT_NAME}"
    echo "  - ENV: ${ENV}"
    echo "  - VERSION: ${VERSION}"
    echo "  - REGION_NAME: ${REGION_NAME}"
    echo "  - DOCKER_REPOSITORY_TYPE: ${DOCKER_REPOSITORY_TYPE}"
    echo ""

    # Define terraform variables
    export TF_VAR_PROJECT_NAME=${PROJECT_NAME}
    export TF_VAR_VERSION=${VERSION}
    export TF_VAR_REGION_NAME=${REGION_NAME}

    # Build ECR repository
    if [ "${DOCKER_REPOSITORY_TYPE}" == "ECR" ]; then
        if [ "${ENV}" == "prod" ]; then
            echo "Building production ECR repository with Terraform..."
            echo ""

            # Initialize Terraform
            terraform -chdir=terraform/production/ecr init -var-file="ecr.tfvars"

            # Apply the configurations and create resources
            terraform -chdir=terraform/production/ecr apply -auto-approve -var-file="ecr.tfvars"
        else
            echo "Building development ECR repository with Terraform..."
            echo ""

            # Initialize Terraform
            terraform -chdir=terraform/development/ecr init -var-file="ecr.tfvars"

            # Apply the configurations and create resources
            terraform -chdir=terraform/development/ecr apply -auto-approve -var-file="ecr.tfvars"
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
    echo "Skipping docker repository building..."
    echo ""
fi