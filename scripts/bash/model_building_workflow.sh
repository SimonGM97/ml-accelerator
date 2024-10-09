#!/bin/bash
# chmod +x ./scripts/bash/model_building_workflow.sh
# ./scripts/bash/model_building_workflow.sh

# Data Processing vars
FIT_TRANSFORMERS=True
SAVE_TRANSFORMERS=True
PERSIST_DATASETS=True
OVERWRITE=True

# Model Tuning vars
MAX_EVALS=100
LOSS_THRESHOLD=0.995
TIMEOUT_MINS=15

# Model Training vars
TRAIN_PROD_MODEL=True
TRAIN_STAGING_MODELS=True
TRAIN_DEV_MODELS=False

# Pipeline Evaluation vars
EVALUATE_PROD_PIPE=True
EVALUATE_STAGING_PIPES=True
EVALUATE_DEV_PIPES=True
UPDATE_MODEL_STAGES=True
UPDATE_PROD_MODEL=True

# Run data processing script
.ml_accel_venv/bin/python scripts/data_processing/data_processing.py \
    --fit_transformers ${FIT_TRANSFORMERS} \
    --save_transformers ${SAVE_TRANSFORMERS} \
    --persist_datasets ${PERSIST_DATASETS} \
    --overwrite ${OVERWRITE}

# Run model tuning script
.ml_accel_venv/bin/python scripts/tuning/tuning.py \
    --max_evals ${MAX_EVALS} \
    --loss_threshold ${LOSS_THRESHOLD} \
    --timeout_mins ${TIMEOUT_MINS}

# Run model training script
.ml_accel_venv/bin/python scripts/training/training.py \
    --train_prod_model ${TRAIN_PROD_MODEL} \
    --train_staging_models ${TRAIN_STAGING_MODELS} \
    --train_dev_models ${TRAIN_DEV_MODELS}

# Run pipeline evaluation script
.ml_accel_venv/bin/python scripts/evaluation/evaluation.py \
    --evaluate_prod_pipe ${EVALUATE_PROD_PIPE} \
    --evaluate_staging_pipes ${EVALUATE_STAGING_PIPES} \
    --evaluate_dev_pipes ${EVALUATE_DEV_PIPES}  \
    --update_model_stages ${UPDATE_MODEL_STAGES}  \
    --update_prod_model ${UPDATE_PROD_MODEL}