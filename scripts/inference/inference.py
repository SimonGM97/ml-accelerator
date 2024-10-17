#!/usr/bin/env python3
from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.etl import ExtractTransformLoad
from ml_accelerator.modeling.model_registry import ModelRegistry
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.logging.logger_helper import get_logger, log_params
from ml_accelerator.utils.timing.timing_helper import timing

from flask import Flask, request, json, jsonify
import pandas as pd
import numpy as np


# Get logger
LOGGER = get_logger(name=__name__)


@timing
def new_inference(X: pd.DataFrame) -> dict:
    LOGGER.info('Received X:\n%s', X)

    # Instanciate ModelRegistry
    MR: ModelRegistry = ModelRegistry()

    # Load production MLPipeline
    pipeline: MLPipeline = MR.load_prod_pipe()

    # Predict test
    y_pred: np.ndarray = pipeline.predict(X=X)

    # Interpret probabilities
    if 'classification' in pipeline.task:
        y_pred: np.ndarray = pipeline.model.interpret_score(y_pred)

    # Prepare new inference
    inference: dict = {
        'prediction': y_pred.tolist()
    }

    return inference


# Instanciate Flask application
app = Flask(__name__)

# Define default method
@app.route("/")
def hello():
    return jsonify(message="Welcome to the ML Accelerator!")

# Define predict method
@app.route("/predict", methods=["GET"])
def predict():
    # Extract url args
    pred_id = eval(request.args.get("pred_id"))
    LOGGER.info('pred_id: %s (%s)', pred_id, type(pred_id))

    # Instanciate ExtractTransformLoad
    ETL: ExtractTransformLoad = ExtractTransformLoad()

    # Extract new X
    X, _ = ETL.run_pipeline(pred_id=pred_id)

    # Predict y
    inference: dict = new_inference(X=X)

    # Append useful data to inference
    inference['pred_id'] = pred_id
    inference['features'] = X.columns.tolist()
    inference['X'] = X.values.tolist()

    return jsonify(inference)

# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python scripts/inference/inference.py
if __name__ == "__main__":
    # Run application
    app.run(
        host=Params.INFERENCE_HOST, 
        port=Params.INFERENCE_PORT, 
        debug=True
    )