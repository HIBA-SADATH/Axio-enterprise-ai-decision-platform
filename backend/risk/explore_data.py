
import pandas as pd

# Load the final risk results
df = pd.read_csv("data/risk_results.csv")

print("Dataset shape:", df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(
    df[
        [
            "Sales",
            "Discount",
            "Profit",
            "Anomaly",
            "Risk Level",
            "Risk Explanation"
        ]
    ].head()
)

print("\nMissing values:")
print(df.isnull().sum())