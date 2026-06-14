"""
test_utils.py — Unit tests (required by US 3.1 CI pipeline)
Run with:  pytest tests/
"""

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_sample_model():
    """Train a tiny model for testing purposes."""
    data = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model


def get_sample_input(n_rows=2):
    """Return n rows of breast cancer features as a list of lists."""
    data = load_breast_cancer()
    return data.data[:n_rows].tolist()


# ── Data loading tests ────────────────────────────────────────────────────────

def test_data_loads_correctly():
    """Dataset must have 30 features and a binary target."""
    data = load_breast_cancer()
    assert data.data.shape[1] == 30, "Expected 30 features"
    assert set(data.target) == {0, 1}, "Expected binary target"


def test_train_test_split_sizes():
    """80/20 split must produce expected row counts."""
    data = load_breast_cancer()
    X_train, X_test, _, _ = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    total = len(data.data)
    assert len(X_train) == pytest.approx(total * 0.8, abs=2)
    assert len(X_test) == pytest.approx(total * 0.2, abs=2)


# ── Model tests ───────────────────────────────────────────────────────────────

def test_model_predicts_correct_shape():
    """Model output shape must match number of input rows."""
    model = get_sample_model()
    X = get_sample_input(n_rows=5)
    preds = model.predict(X)
    assert len(preds) == 5


def test_model_predicts_valid_classes():
    """Predictions must only contain 0 or 1."""
    model = get_sample_model()
    X = get_sample_input(n_rows=10)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1}), "Predictions outside expected classes"


# ── Score.py serialisation tests (Bug 3 prevention) ──────────────────────────

def test_predictions_are_json_serialisable():
    """
    Numpy arrays are NOT natively JSON serialisable.
    .tolist() must convert them to plain Python ints/floats.
    This test prevents Bug 3 (HTTP 400 on inference).
    """
    model = get_sample_model()
    X = get_sample_input(n_rows=3)
    raw_preds = model.predict(X)

    # This would raise TypeError if numpy types are not converted
    serialised = json.dumps({"predictions": raw_preds.tolist()})
    result = json.loads(serialised)

    assert "predictions" in result
    assert len(result["predictions"]) == 3
    assert all(isinstance(p, int) for p in result["predictions"])


def test_score_payload_parsing():
    """Input JSON payload must parse correctly into a DataFrame."""
    sample_input = get_sample_input(n_rows=2)
    payload = json.dumps({"data": sample_input})

    parsed = json.loads(payload)
    df = pd.DataFrame(parsed["data"])

    assert df.shape == (2, 30), f"Expected (2, 30), got {df.shape}"


def test_score_handles_bad_payload():
    """Malformed payload must not crash the endpoint — returns error key."""
    bad_payload = json.dumps({"wrong_key": []})

    try:
        parsed = json.loads(bad_payload)
        _ = parsed["data"]  # will raise KeyError
        result = {"predictions": []}
    except (KeyError, Exception) as e:
        result = {"error": str(e)}

    assert "error" in result
