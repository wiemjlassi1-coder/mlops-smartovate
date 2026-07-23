"""
reference_stats.py — MLOPS-14
Computes the reference (baseline) feature distribution from the training
dataset. This snapshot is what incoming production data gets compared
against to detect data drift. Run this once after training (or whenever
the training data changes) and commit the resulting JSON to the repo.
"""

import json

import numpy as np
from sklearn.datasets import load_breast_cancer


def compute_reference_bins(n_bins: int = 10):
    """
    For each feature, compute bin edges (deciles) from the training data.
    These bins are reused later to bucket production data the same way,
    which is required for a valid PSI (Population Stability Index) comparison.
    """
    data = load_breast_cancer()
    X = data.data
    feature_names = list(data.feature_names)

    reference = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        # Decile bin edges based on the training distribution
        bin_edges = np.quantile(col, np.linspace(0, 1, n_bins + 1)).tolist()
        # Reference proportions per bin (should be ~uniform since deciles)
        counts, _ = np.histogram(col, bins=bin_edges)
        proportions = (counts / counts.sum()).tolist()

        reference[name] = {
            "bin_edges": bin_edges,
            "reference_proportions": proportions,
        }

    return reference


if __name__ == "__main__":
    reference = compute_reference_bins()
    with open("monitoring/reference_stats.json", "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Reference statistics saved for {len(reference)} features "
          f"to monitoring/reference_stats.json")