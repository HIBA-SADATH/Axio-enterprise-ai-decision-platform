
from pathlib import Path

import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

input_path = DATA_DIR / "Cleaned_Superstore.csv"
output_path = DATA_DIR / "shap_explanations.csv"

# Load the same dataset used for anomaly detection
df = pd.read_csv(input_path)

features = ["Sales", "Quantity", "Discount", "Profit"]
X = df[features].copy()

# Train the anomaly detection model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
model.fit(X)

# Explain the model's anomaly scores
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Save feature contributions with their original row indexes
result = pd.DataFrame(
    shap_values,
    columns=[f"SHAP_{feature}" for feature in features]
)

result.insert(0, "Original Row Index", range(len(df)))
result.to_csv(output_path, index=False)

print(f"SHAP explanations saved to: {output_path}")
print(result.head())
