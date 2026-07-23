# monitoring/simulate_production_data.py
import pandas as pd
import numpy as np

df = pd.read_csv("src/breast_cancer.csv")


# Simulate drift: shift a few features to mimic real-world distribution change
drifted = df.copy()
drifted["mean radius"] = drifted["mean radius"] * 1.25  # simulate 25% upward shift
drifted["mean texture"] = drifted["mean texture"] + 5     # simulate a mean shift

drifted.to_csv("monitoring/simulated_production_data.csv", index=False)
print(f"Simulated production data saved: {len(drifted)} rows")