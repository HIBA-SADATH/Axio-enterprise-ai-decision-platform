import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

# 1. Load the dataset
df = pd.read_csv("data/Cleaned_Superstore.csv")

# 2. Select numerical features
features = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Profit Margin",
]

X = df[features].copy()

# 3. Create and train the Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X)

# 4. Label normal records and anomalies
df["Anomaly"] = pd.Series(predictions, index=df.index).map({
    1: "Normal",
    -1: "Anomaly"
})

print("Anomaly detection completed!")
print(df["Anomaly"].value_counts())

# 5. Prepare SHAP background data
# A small sample keeps explanations faster.
background = shap.sample(X, min(50, len(X)), random_state=42)

# Isolation Forest score_samples:
# Lower scores generally indicate more unusual records.
explainer = shap.Explainer(
    model.score_samples,
    background,
    algorithm="permutation"
)

# 6. Explain flagged records
anomaly_indices = df.index[df["Anomaly"] == "Anomaly"]
X_anomalies = X.loc[anomaly_indices]

df["SHAP Explanation"] = "Not calculated"

if not X_anomalies.empty:
    # Explain up to 100 anomalies in this run.
    # Increase this limit later if needed.
    rows_to_explain = X_anomalies.head(100)

    shap_values = explainer(
        rows_to_explain,
        max_evals=2 * len(features) + 1
    )

    # Identify the feature with the largest absolute SHAP value.
    for position, row_index in enumerate(rows_to_explain.index):
        values = shap_values.values[position]
        strongest_feature = features[
            int(abs(values).argmax())
        ]

        df.loc[row_index, "SHAP Explanation"] = (
            f"Strongest contributing feature: {strongest_feature}"
        )

    # Save individual feature contributions for analysis.
    shap_df = pd.DataFrame(
        shap_values.values,
        columns=[f"SHAP_{feature}" for feature in features],
        index=rows_to_explain.index
    )

    shap_df.index.name = "Original Row Index"
    shap_df.to_csv("data/shap_explanations.csv")

# 7. Display sample anomalies and explanations
print("\nSample flagged records:")
print(
    df.loc[
        df["Anomaly"] == "Anomaly",
        features + ["Anomaly", "SHAP Explanation"]
    ].head()
)

# 8. Save the full dataset with anomaly labels and explanations
df.to_csv("data/anomaly_results.csv", index=False)

print("\nResults saved to data/anomaly_results.csv")
print("SHAP contributions saved to data/shap_explanations.csv")
