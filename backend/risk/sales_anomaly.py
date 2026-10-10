
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("data/Cleaned_Superstore.csv")

# Select sales-related features
features = ["Sales", "Quantity", "Discount", "Profit"]
X = df[features].copy()

# Train anomaly detection model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X)

# Label anomalies
df["Sales Anomaly"] = pd.Series(
    predictions, index=df.index
).map({1: "Normal", -1: "Anomaly"})

# Add an explanation
df["Sales Anomaly Explanation"] = df.apply(
    lambda row: (
        "Unusual sales transaction; review sales, quantity, "
        "discount, and profit"
        if row["Sales Anomaly"] == "Anomaly"
        else "No unusual pattern detected"
    ),
    axis=1
)

# Save separately; existing files remain untouched
df.to_csv("data/sales_anomaly_results.csv", index=False)

print("SALES ANOMALY SUMMARY")
print("-" * 35)
print(df["Sales Anomaly"].value_counts())
print("\nSaved to data/sales_anomaly_results.csv")
