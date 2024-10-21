#!/bin/bash
# chmod +x ./scripts/bash/model_building_workflow.sh
# ./scripts/bash/model_building_workflow.sh

# Unset environment variables
unset VERSION ENV BUCKET INFERENCE_PORT

# Extract variables
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

# Show variables
echo "Model Building Workflow variables:"
echo "  - VERSION: ${VERSION}"
echo ""

# Clean containers
if [ "$(docker ps -aq)" ]; then
    docker rm -f $(docker ps -aq)
fi

# Run docker-compose
VERSION=${VERSION} \
    docker-compose \
    -f docker/docker-compose-app.yaml \
    --env-file .env \
    up

# Remove running services
# VERSION=${VERSION} \
#     docker-compose \
#     -f docker/docker-compose-app.yaml \
#     --env-file .env \
#     down