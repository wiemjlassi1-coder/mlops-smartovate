"""
train.py — Baseline training script (US 1.2 + US 2.1)
Trains a Random Forest on the breast cancer dataset.
MLflow tracking is included from the start (covers US 2.1 automatically).
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load and split the breast cancer dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train(n_estimators: int, max_depth: int):
    """Train a Random Forest and log everything to MLflow."""

    X_train, X_test, y_train, y_test = load_data()

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with mlflow.start_run():

        # ── Log hyperparameters ──────────────────────────────────────────
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset", "breast_cancer")
        mlflow.log_param("test_size", 0.2)

        # ── Train ────────────────────────────────────────────────────────
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # ── Evaluate ─────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # ── Log metrics ──────────────────────────────────────────────────
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1-Score : {f1:.4f}")

        # ── Log model artefact ───────────────────────────────────────────
        import os
        import pickle
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/model.pkl", "wb") as f:
            pickle.dump(model, f)

        print("Model saved to outputs/model.pkl")
        print("Model logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=5)
    args = parser.parse_args()

    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
