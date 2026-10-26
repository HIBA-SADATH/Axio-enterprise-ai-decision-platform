# src/data_validation.py
"""
Phase 3 – Data Verification

This script loads the processed feature CSV, validates the integrity of the dataset
according to the Phase‑3 requirements, and prints a concise verification report.
It is deliberately lightweight and has no side‑effects beyond console output.
"""

import os
import pandas as pd
import numpy as np

# Resolve paths relative to the repository root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "Cleaned_Superstore.csv")


def load_dataframe(path: str) -> pd.DataFrame:
    """Read a CSV file with default UTF‑8 encoding."""
    return pd.read_csv(path)


def verify_processed(df: pd.DataFrame) -> dict:
    """Run all verification checks on the processed dataframe.
    Returns a dictionary of boolean flags and explanatory messages.
    """
    results = {}
    # 1. Row count
    expected_rows = 9_977
    results["row_count"] = len(df) == expected_rows
    results["row_count_msg"] = f"Rows: {len(df)} (expected {expected_rows})"

    # 2. Column count (14 features + target)
    expected_cols = 15
    results["col_count"] = df.shape[1] == expected_cols
    results["col_count_msg"] = f"Columns (incl. target): {df.shape[1]} (expected {expected_cols})"
    feature_cols = [c for c in df.columns if c != "Churn"]
    results["feature_count"] = len(feature_cols) == 14
    results["feature_count_msg"] = f"Predictive features: {len(feature_cols)} (expected 14)"

    # 3. Missing values
    missing_total = df.isnull().sum().sum()
    results["no_missing"] = missing_total == 0
    results["missing_msg"] = f"Total missing values: {missing_total}"

    # 4. Infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    infinite_present = np.isinf(df[numeric_cols].values).any()
    results["no_infinite"] = not infinite_present
    results["infinite_msg"] = "Infinite values found" if infinite_present else "No infinite values"

    # 5. Churn binary check
    churn_unique = set(df["Churn"].unique())
    results["churn_binary"] = churn_unique.issubset({0, 1})
    results["churn_binary_msg"] = f"Churn unique values: {sorted(churn_unique)}"

    # 6. Excluded columns absence
    excluded = {"Customer ID", "Country", "Profit", "Profit Margin", "Profit Status", "Inventory Risk"}
    present_excluded = excluded.intersection(set(df.columns))
    results["excluded_absent"] = len(present_excluded) == 0
    results["excluded_msg"] = (
        f"Excluded columns present: {sorted(present_excluded)}" if present_excluded else "All excluded columns absent"
    )
    return results


def compare_target_distribution(processed: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Confirm that the target distribution matches the original raw data."""
    raw_counts = raw["Churn"].value_counts().sort_index().to_dict()
    proc_counts = processed["Churn"].value_counts().sort_index().to_dict()
    match = raw_counts == proc_counts
    return {
        "raw_counts": raw_counts,
        "processed_counts": proc_counts,
        "distribution_match": match,
        "msg": f"Raw vs processed target distribution match: {match}",
    }


def main():
    processed_df = load_dataframe(PROCESSED_CSV)
    raw_df = load_dataframe(RAW_CSV)
    print("--- Phase 3 Data Verification Report ---")
    ver = verify_processed(processed_df)
    for key in sorted(ver):
        if key.endswith("_msg"):
            print(ver[key])
    dist = compare_target_distribution(processed_df, raw_df)
    print(dist["msg"])
    print(f"Raw target counts: {dist['raw_counts']}")
    print(f"Processed target counts: {dist['processed_counts']}")
    print("Verification completed.")

if __name__ == "__main__":
    main()
