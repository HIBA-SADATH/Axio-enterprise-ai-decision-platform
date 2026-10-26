

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Path configuration (relative to repository root)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # <repo>/member1_customer_intelligence
PROCESSED_CSV = BASE_DIR / "data" / "processed" / "customer_features.csv"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Read the processed feature CSV.
    Returns a DataFrame with the original column ordering.
    """
    return pd.read_csv(PROCESSED_CSV)

# ---------------------------------------------------------------------------
# Split definition
# ---------------------------------------------------------------------------
TARGET_COL = "Churn"

def get_feature_target(df: pd.DataFrame):
    """Separate feature matrix `X` and target vector `y`.
    The target column is **never** included in `X`.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Perform a stratified train/test split.
    Returns ``X_train, X_test, y_train, y_test``.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
# Identify column types based on domain knowledge (same as Phase‑2 report)
CATEGORICAL_FEATURES = [
    "Ship Mode",
    "Segment",
    "State",
    "Region",
    "Category",
    "Sub-Category",
    "Sales Category",
    "discount_tier",
    "region_category_interaction",
]
NUMERICAL_FEATURES = ["Sales", "Quantity", "Discount", "log_sales", "sales_per_quantity"]

def build_preprocess_pipeline():
    """Create a ColumnTransformer that handles categorical and numeric columns.
    The transformer is *stateless* until ``fit`` is called on training data.
    """
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERICAL_FEATURES),
        ]
    )
    # Wrap in a Pipeline for easy use downstream
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipeline

# ---------------------------------------------------------------------------
# Public helper to obtain ready‑to‑use arrays
# ---------------------------------------------------------------------------
def prepare_data():
    """Load, split, and fit the preprocessing pipeline.
    Returns a dictionary containing:
        - X_train, X_test (numpy arrays after transformation)
        - y_train, y_test (pandas Series)
        - pipeline (the fitted scikit‑learn Pipeline)
        - feature_names (list of column names after one‑hot encoding)
    """
    df = load_data()
    X, y = get_feature_target(df)
    X_train_df, X_test_df, y_train, y_test = stratified_split(X, y)
    pipeline = build_preprocess_pipeline()
    # Fit only on training data to avoid leakage
    pipeline.fit(X_train_df)
    # Transform both splits
    X_train = pipeline.transform(X_train_df)
    X_test = pipeline.transform(X_test_df)
    # Obtain feature names after one‑hot encoding for reference
    cat_ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_features = cat_ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names = list(cat_features) + NUMERICAL_FEATURES
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline": pipeline,
        "feature_names": feature_names,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
    }

# ---------------------------------------------------------------------------
# Simple sanity‑check when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    payload = prepare_data()
    print("--- Phase 3 Preprocessing Summary ---")
    print(f"Training set rows: {payload['y_train'].shape[0]}")
    print(f"Test set rows: {payload['y_test'].shape[0]}")
    print("Class distribution in training set:")
    print(payload['y_train'].value_counts().sort_index())
    print("Class distribution in test set:")
    print(payload['y_test'].value_counts().sort_index())
    print(f"Feature matrix shape after encoding: train {payload['train_shape']}, test {payload['test_shape']}")
    print("Preprocessing pipeline fitted on training data only – ready for downstream modeling.")
