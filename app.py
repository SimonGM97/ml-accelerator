#!/usr/bin/env python3
from ml_accelerator.config.env import Env
from scripts.data_processing.data_processing import data_pipeline
from scripts.tuning.tuning import tuning_pipeline
from scripts.training.training import training_pipeline
from scripts.evaluating.evaluating import evaluating_pipeline
from scripts.inference.inference import inference_pipeline
from scripts.drift.drift import drift_pipeline

from flask import Flask, request, json, jsonify
import pandas as pd
import os


# Instanciate Flask application
app = Flask(__name__)

# Define default method
@app.route("/")
def hello():
    return jsonify(message="Welcome to the ML Accelerator!")

# Define data_processing method
@app.route("/data_processing", methods=["GET"])
def data_processing() -> json:
    # Extract url args
    fit_transformers: bool = eval(request.args.get("fit_transformers", "False"))
    save_transformers: bool = eval(request.args.get("save_transformers", "False"))
    persist_datasets: bool = eval(request.args.get("persist_datasets", "True"))
    write_mode: str = request.args.get("write_mode", "append")

    # Run data pipeline
    X, y = data_pipeline(
        fit_transformers=fit_transformers,
        save_transformers=save_transformers,
        persist_datasets=persist_datasets,
        write_mode=write_mode
    )

    X: pd.DataFrame = X
    y: pd.DataFrame = y

    return jsonify({
        'X_columns': X.columns.tolist(),
        'X_values': X.values.tolist(),
        'y_columns': y.columns.tolist(),
        'y_values': y.values.tolist()
    })

# Define tuning method
@app.route("/tune", methods=["GET"])
def tune() -> None:
    # Run tuning pipeline
    tuning_pipeline()

    return jsonify({})

# Define training method
@app.route("/train", methods=["GET"])
def train() -> None:
    # Extract url args
    train_prod_pipe: bool = eval(request.args.get("train_prod_pipe", "True"))
    train_staging_pipes: bool = eval(request.args.get("train_staging_pipes", "True"))
    train_dev_pipes: bool = eval(request.args.get("train_dev_pipes", "False"))

    # Run training pipeline
    training_pipeline(
        train_prod_pipe=train_prod_pipe,
        train_staging_pipes=train_staging_pipes,
        train_dev_pipes=train_dev_pipes,
        debug=False
    )

    return jsonify({})

# Define evaluate method
@app.route("/evaluate", methods=["GET"])
def evaluate() -> None:
    # Extract url args
    evaluate_prod_pipe: bool = eval(request.args.get("evaluate_prod_pipe", "True"))
    evaluate_staging_pipes: bool = eval(request.args.get("evaluate_staging_pipes", "True"))
    evaluate_dev_pipes: bool = eval(request.args.get("evaluate_dev_pipes", "False"))
    update_model_stages: bool = eval(request.args.get("update_model_stages", "True"))
    update_prod_model: bool = eval(request.args.get("update_prod_model", "False"))

    # Run training pipeline
    evaluating_pipeline(
        evaluate_prod_pipe=evaluate_prod_pipe,
        evaluate_staging_pipes=evaluate_staging_pipes,
        evaluate_dev_pipes=evaluate_dev_pipes,
        update_model_stages=update_model_stages,
        update_prod_model=update_prod_model,
        debug=False
    )

    return jsonify({})

# Define predict method
@app.route("/predict", methods=["GET"])
def predict() -> json:
    # Extract url args
    pred_id = eval(request.args.get("pred_id", "None"))

    # Run inference pipeline
    inference: dict = inference_pipeline(pred_id=pred_id)

    return jsonify(inference)

# Define drift method
@app.route("/drift", methods=["GET"])
def drift() -> json:
    # Extract url args
    param1 = eval(request.args.get("param1", "None"))

    # Run drift pipeline
    drift_pipeline(param1=param1)

    return jsonify({})


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python app.py
if __name__ == "__main__":
    # Run application
    app.run(
        host=Env.get("INFERENCE_HOST"),
        port=Env.get("INFERENCE_PORT"), 
        debug=True
    )