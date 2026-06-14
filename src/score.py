"""
score.py — Inference script for the Managed Online Endpoint (US 3.2)
Azure ML calls init() once at startup and run() on every prediction request.
"""

import json
import os
import numpy as np
import pandas as pd
import mlflow.sklearn


def init():
    """Load the model from the Azure ML model registry at startup."""
    global model

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR", "."), "model"
    )
    model = mlflow.sklearn.load_model(model_path)
    print("Model loaded successfully.")


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
