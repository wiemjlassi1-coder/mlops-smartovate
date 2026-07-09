"""
score.py — Inference script for the Managed Online Endpoint (US 3.2)
Azure ML calls init() once at startup and run() on every prediction request.

Note (MLOPS-11): the registered model is a raw pickle file (outputs/model.pkl),
not an MLflow-format model — mlflow.sklearn.log_model() failed during MLOPS-6
(HTTP 404, documented in the rapport), so the model was serialized manually
via pickle.dump(). init() therefore loads it with pickle, not
mlflow.sklearn.load_model().
"""

import glob
import json
import os
import pickle

import pandas as pd


def init():
    """Load the model from the Azure ML model mount directory at startup."""
    global model

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")

    # The exact subpath under AZUREML_MODEL_DIR can vary depending on how the
    # model was registered (single-file vs. folder-based registration), so we
    # search for the .pkl file rather than hardcoding a path.
    candidates = glob.glob(os.path.join(model_dir, "**", "*.pkl"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No .pkl model file found under AZUREML_MODEL_DIR={model_dir}"
        )

    model_path = candidates[0]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded successfully from {model_path}")


def run(raw_data: str) -> str:
    """
    Receive a JSON payload, return predictions as JSON.

    Expected input format:
        {"data": [[val1, val2, ..., val30], [val1, val2, ..., val30]]}

    Returns:
        {"predictions": [1, 0, ...]}
    """
    try:
        payload = json.loads(raw_data)
        df = pd.DataFrame(payload["data"])

        predictions = model.predict(df)

        # Convert numpy types to native Python — prevents JSON serialisation
        # error (Bug 3 in the cahier des charges)
        return json.dumps({"predictions": predictions.tolist()})

    except Exception as e:
        return json.dumps({"error": str(e)})
    