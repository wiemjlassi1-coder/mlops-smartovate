from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Add target column
df["target"] = data.target

# Save to CSV
df.to_csv("breast_cancer.csv", index=False)

print("Dataset exported successfully.")