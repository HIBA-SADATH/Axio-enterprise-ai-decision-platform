
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Load the dataset
df = pd.read_csv("data/Cleaned_Superstore.csv")

# 2. Select transaction features relevant to inventory analysis
features = ["Quantity", "Sales", "Discount", "Profit"]
X = df[features].copy()

# 3. Train the anomaly detection model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X)

# 4. Label unusual transactions
df["Inventory Anomaly"] = pd.Series(
    predictions, index=df.index
).map({
    1: "Normal",
    -1: "Anomaly"
})

# 5. Add a simple explanation for review
df["Inventory Anomaly Explanation"] = df.apply(
    lambda row: (
        "Unusual transaction pattern; review quantity, sales, "
        "discount, and profit"
        if row["Inventory Anomaly"] == "Anomaly"
        else "No unusual pattern detected by this model"
    ),
    axis=1
)

# 6. Save the results separately
df.to_csv("data/inventory_anomaly_results.csv", index=False)

# 7. Display a summary
print("INVENTORY-RELATED ANOMALY SUMMARY")
print("-" * 40)
print(df["Inventory Anomaly"].value_counts())
print("\nResults saved to data/inventory_anomaly_results.csv")
