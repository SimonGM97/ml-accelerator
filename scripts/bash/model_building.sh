#!/bin/bash
# chmod +x ./scripts/bash/model_building.sh
# ./scripts/bash/model_building.sh

# Unset environment variables
unset VERSION ENV BUCKET \
    FIT_TRANSFORMERS SAVE_TRANSFORMERS PERSIST_DATASETS WRITE_MODE \
    TRAIN_PROD_MODEL TRAIN_STAGING_MODELS TRAIN_DEV_MODELS

# Extract variables
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

FIT_TRANSFORMERS=$(yq eval '.MODEL_BUILDING_PARAMS.FIT_TRANSFORMERS' ${CONFIG_FILE})
SAVE_TRANSFORMERS=$(yq eval '.MODEL_BUILDING_PARAMS.SAVE_TRANSFORMERS' ${CONFIG_FILE})
PERSIST_DATASETS=$(yq eval '.MODEL_BUILDING_PARAMS.PERSIST_DATASETS' ${CONFIG_FILE})
WRITE_MODE=$(yq eval '.MODEL_BUILDING_PARAMS.WRITE_MODE' ${CONFIG_FILE})

TRAIN_PROD_PIPE=$(yq eval '.MODEL_BUILDING_PARAMS.TRAIN_PROD_PIPE' ${CONFIG_FILE})
TRAIN_STAGING_PIPES=$(yq eval '.MODEL_BUILDING_PARAMS.TRAIN_STAGING_PIPES' ${CONFIG_FILE})
TRAIN_DEV_PIPES=$(yq eval '.MODEL_BUILDING_PARAMS.TRAIN_DEV_PIPES' ${CONFIG_FILE})

EVALUATE_PROD_PIPE=$(yq eval '.MODEL_BUILDING_PARAMS.EVALUATE_PROD_PIPE' ${CONFIG_FILE})
EVALUATE_STAGING_PIPES=$(yq eval '.MODEL_BUILDING_PARAMS.EVALUATE_STAGING_PIPES' ${CONFIG_FILE})
EVALUATE_DEV_PIPES=$(yq eval '.MODEL_BUILDING_PARAMS.EVALUATE_DEV_PIPES' ${CONFIG_FILE})
UPDATE_MODEL_STAGES=$(yq eval '.MODEL_BUILDING_PARAMS.UPDATE_MODEL_STAGES' ${CONFIG_FILE})
UPDATE_PROD_MODEL=$(yq eval '.MODEL_BUILDING_PARAMS.UPDATE_PROD_MODEL' ${CONFIG_FILE})

# Show variables
echo "Model Building Workflow variables:"
echo "  - VERSION: ${VERSION}"

echo "  - FIT_TRANSFORMERS: ${FIT_TRANSFORMERS}"
echo "  - SAVE_TRANSFORMERS: ${SAVE_TRANSFORMERS}"
echo "  - PERSIST_DATASETS: ${PERSIST_DATASETS}"
echo "  - WRITE_MODE: ${WRITE_MODE}"

echo "  - TRAIN_PROD_PIPE: ${TRAIN_PROD_PIPE}"
echo "  - TRAIN_STAGING_PIPES: ${TRAIN_STAGING_PIPES}"
echo "  - TRAIN_DEV_PIPES: ${TRAIN_DEV_PIPES}"

echo "  - EVALUATE_PROD_PIPE: ${EVALUATE_PROD_PIPE}"
echo "  - EVALUATE_STAGING_PIPES: ${EVALUATE_STAGING_PIPES}"
echo "  - EVALUATE_DEV_PIPES: ${EVALUATE_DEV_PIPES}"
echo "  - UPDATE_MODEL_STAGES: ${UPDATE_MODEL_STAGES}"
echo "  - UPDATE_PROD_MODEL: ${UPDATE_PROD_MODEL}"
echo ""

# Clean containers
if [ "$(docker ps -aq)" ]; then
    docker rm -f $(docker ps -aq)
fi

# Run docker-compose
VERSION=${VERSION} \
    FIT_TRANSFORMERS=${FIT_TRANSFORMERS} \
    SAVE_TRANSFORMERS=${SAVE_TRANSFORMERS} \
    PERSIST_DATASETS=${PERSIST_DATASETS} \
    WRITE_MODE=${WRITE_MODE} \
    TRAIN_PROD_PIPE=${TRAIN_PROD_PIPE} \
    TRAIN_STAGING_PIPES=${TRAIN_STAGING_PIPES} \
    TRAIN_DEV_PIPES=${TRAIN_DEV_PIPES} \
    EVALUATE_PROD_PIPE=${EVALUATE_PROD_PIPE} \
    EVALUATE_STAGING_PIPES=${EVALUATE_STAGING_PIPES} \
    EVALUATE_DEV_PIPES=${EVALUATE_DEV_PIPES} \
    UPDATE_MODEL_STAGES=${UPDATE_MODEL_STAGES} \
    UPDATE_PROD_MODEL=${UPDATE_PROD_MODEL} \
    docker-compose \
    -f docker/model_building/docker-compose-model-building.yaml \
    --env-file .env \
    up

# Remove running services
VERSION=${VERSION} \
    FIT_TRANSFORMERS=${FIT_TRANSFORMERS} \
    SAVE_TRANSFORMERS=${SAVE_TRANSFORMERS} \
    PERSIST_DATASETS=${PERSIST_DATASETS} \
    WRITE_MODE=${WRITE_MODE} \
    TRAIN_PROD_PIPE=${TRAIN_PROD_PIPE} \
    TRAIN_STAGING_PIPES=${TRAIN_STAGING_PIPES} \
    TRAIN_DEV_PIPES=${TRAIN_DEV_PIPES} \
    EVALUATE_PROD_PIPE=${EVALUATE_PROD_PIPE} \
    EVALUATE_STAGING_PIPES=${EVALUATE_STAGING_PIPES} \
    EVALUATE_DEV_PIPES=${EVALUATE_DEV_PIPES} \
    UPDATE_MODEL_STAGES=${UPDATE_MODEL_STAGES} \
    UPDATE_PROD_MODEL=${UPDATE_PROD_MODEL} \
    docker-compose \
    -f docker/model_building/docker-compose-model-building.yaml \
    --env-file .env \
    down