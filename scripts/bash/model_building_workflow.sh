#!/bin/bash
# chmod +x ./scripts/bash/model_building_workflow.sh
# ./scripts/bash/model_building_workflow.sh

# Extract variables
CONFIG_FILE="config/config.yaml"

VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})
ENV=$(yq eval '.ENV_PARAMS.ENV' ${CONFIG_FILE})
BUCKET=$(yq eval '.ENV_PARAMS.BUCKET' ${CONFIG_FILE})

FIT_TRANSFORMERS=$(yq eval '.MODEL_BUILDING_PARAMS.FIT_TRANSFORMERS' ${CONFIG_FILE})
SAVE_TRANSFORMERS=$(yq eval '.MODEL_BUILDING_PARAMS.SAVE_TRANSFORMERS' ${CONFIG_FILE})
PERSIST_DATASETS=$(yq eval '.MODEL_BUILDING_PARAMS.PERSIST_DATASETS' ${CONFIG_FILE})
WRITE_MODE=$(yq eval '.MODEL_BUILDING_PARAMS.WRITE_MODE' ${CONFIG_FILE})

MAX_EVALS=$(yq eval '.HYPER_PARAMETER_TUNING_PARAMS.MAX_EVALS' ${CONFIG_FILE})
LOSS_THRESHOLD=$(yq eval '.HYPER_PARAMETER_TUNING_PARAMS.LOSS_THRESHOLD' ${CONFIG_FILE})
TIMEOUT_MINS=$(yq eval '.HYPER_PARAMETER_TUNING_PARAMS.TIMEOUT_MINS' ${CONFIG_FILE})

# Run docker-compose
VERSION=${VERSION} \
    ENV=${ENV} \
    BUCKET=${BUCKET} \
    FIT_TRANSFORMERS=${FIT_TRANSFORMERS} \
    SAVE_TRANSFORMERS=${SAVE_TRANSFORMERS} \
    PERSIST_DATASETS=${PERSIST_DATASETS} \
    WRITE_MODE=${WRITE_MODE} \
    MAX_EVALS=${MAX_EVALS} \
    LOSS_THRESHOLD=${LOSS_THRESHOLD} \
    TIMEOUT_MINS=${TIMEOUT_MINS} \
    docker-compose \
    -f docker/docker-compose.yaml \
    --env-file .env \
    up