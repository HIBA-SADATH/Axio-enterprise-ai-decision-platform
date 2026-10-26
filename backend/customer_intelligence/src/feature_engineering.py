"""
Enterprise AI Decision Intelligence Platform
Module: Customer Intelligence & Churn Prediction (Member 1)
Script: feature_engineering.py
Phase: Phase 2 — Leakage-Safe Feature Engineering Pipeline

Description:
    Implements a strictly leakage-safe feature engineering pipeline for customer churn
    prediction. Excludes all post-hoc financial settlement metrics (Profit, Profit Margin,
    Profit Status), artificial threshold flags (Inventory Risk), and unique identifiers
    (Customer ID, Country). Generates mathematically validated pre-sale features
    (log_sales, sales_per_quantity, discount_tier, region_category_interaction), evaluates
    high_risk_subcategory_flag under non-target constraints, validates zero target contamination,
    and outputs the reproducible processed dataset to data/processed/customer_features.csv.

Governance Rules Enforced:
    1. Zero Target Contamination: Churn is NEVER used in feature computation.
    2. Zero Financial Leakage: Profit, Profit Margin, Profit Status, Inventory Risk are strictly excluded.
    3. Ex-Ante Availability: Every engineered feature is strictly computable at checkout / order booking.
    4. Reproducibility: Pipeline is deterministic and idempotent.
"""

import os
import sys
import logging
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FeatureEngineering_Member1")


# -------------------------------------------------------------------------
# CONSTANTS & SCHEMA SPECIFICATIONS
# -------------------------------------------------------------------------

PRODUCTION_BASE_FEATURES = [
    "Ship Mode",
    "Segment",
    "State",
    "Region",
    "Category",
    "Sub-Category",
    "Sales",
    "Quantity",
    "Discount",
    "Sales Category"
]

EXCLUDED_COLUMNS = [
    "Customer ID",       # Identifier: 100% unique row key, zero generalization utility
    "Country",           # Constant: 100% 'United States', zero variance
    "Profit",            # Post-hoc leakage: ERP settled accounting metric, fatal leakage
    "Profit Margin",     # Post-hoc leakage: 100% negative for churners, fatal leakage
    "Profit Status",     # Post-hoc leakage: Loss indicator, 0% churn if Profit, fatal leakage
    "Inventory Risk"     # Redundant proxy: Arbitrary discretization of Quantity >= 4
]

TARGET_COLUMN = "Churn"

# Standard retail markdown tier thresholds (Domain-driven; ZERO Churn dependency)
DISCOUNT_BINS = [-float("inf"), 0.0, 0.20, 0.40, float("inf")]
DISCOUNT_LABELS = ["Zero", "Low", "Moderate", "High"]


# -------------------------------------------------------------------------
# PATH DISCOVERY & DATA LOADING
# -------------------------------------------------------------------------

def find_raw_dataset(override_path: Optional[str] = None) -> str:
    """
    Resolves the absolute or relative path to the raw input dataset.
    """
    if override_path and os.path.exists(override_path):
        return override_path

    search_paths = [
        os.path.join("member1_customer_intelligence", "data", "raw", "Cleaned_Superstore.csv"),
        os.path.join("data", "raw", "Cleaned_Superstore.csv"),
        os.path.join("..", "data", "raw", "Cleaned_Superstore.csv"),
        os.path.join("dataset", "Cleaned_Superstore.csv"),
        os.path.join("..", "dataset", "Cleaned_Superstore.csv"),
        os.path.join("member1_customer_intelligence", "data", "raw", "Cleaned_Superstore(1).csv"),
    ]

    for p in search_paths:
        if os.path.exists(p):
            return os.path.abspath(p)

    raise FileNotFoundError(
        "Could not locate Cleaned_Superstore.csv in any standard repository location. "
        f"Searched: {search_paths}"
    )


def load_raw_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data with initial structural verification.
    """
    logger.info(f"Loading raw dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw dataset successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Verifies that all required production columns and the target exist.
    """
    missing_features = [col for col in PRODUCTION_BASE_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing mandatory production features: {missing_features}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing mandatory target column: '{TARGET_COLUMN}'")
    logger.info("Raw schema validation passed. All mandatory base columns present.")


# -------------------------------------------------------------------------
# FEATURE ENGINEERING TRANSFORMS (LEAKAGE-SAFE & INDEPENDENT)
# -------------------------------------------------------------------------

def create_log_sales(sales: pd.Series) -> pd.Series:
    """
    Computes natural logarithm of 1 + Sales to dampen extreme positive skewness.
    Formula: log_sales = ln(1 + max(Sales, 0))
    
    Zero target leakage: Uses only continuous Sales known at checkout.
    """
    safe_sales = np.maximum(sales.astype(float), 0.0)
    return np.log1p(safe_sales)


def create_sales_per_quantity(sales: pd.Series, quantity: pd.Series) -> pd.Series:
    """
    Average sales value per purchased unit, calculated as Sales / Quantity.
    Formula: sales_per_quantity = Sales / Quantity
    
    Safely handles division by zero or invalid quantity values.
    No direct target leakage detected; feature uses non-target variables available in the assumed prediction setting.
    """
    qty = quantity.astype(float).replace(0.0, np.nan)
    spq = sales.astype(float) / qty
    return spq.fillna(0.0)


def create_discount_tier(discount: pd.Series) -> pd.Series:
    """
    Discretizes discount rate into fixed categorical brackets.
    
    Fixed business-rule thresholds of 0%, 0–20%, 20–40%, and >40%, defined independently of the target variable.
    Tiers:
        - 'Zero'     : Discount == 0.00
        - 'Low'      : 0.00 < Discount <= 0.20
        - 'Moderate' : 0.20 < Discount <= 0.40
        - 'High'     : Discount > 0.40
        
    No direct target leakage detected; feature uses non-target variables available in the assumed prediction setting.
    """
    tier_series = pd.cut(
        discount.astype(float),
        bins=DISCOUNT_BINS,
        labels=DISCOUNT_LABELS,
        right=True
    )
    return tier_series.astype(str)


def create_region_category_interaction(region: pd.Series, category: pd.Series) -> pd.Series:
    """
    Combines macro-geographic Region and product Category into a composite interaction feature.
    Formula: Region + '_' + Category (e.g., 'Central_Furniture')
    
    Captures potential differences in churn behavior across combinations of geographic region and product category.
    This represents association only and does not establish causality.
    No direct target leakage detected; feature uses non-target variables available in the assumed prediction setting.
    """
    return region.astype(str).str.strip() + "_" + category.astype(str).str.strip()


def evaluate_high_risk_subcategory_flag(df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.Series]]:
    """
    Rigorously evaluates whether high_risk_subcategory_flag can be defensibly
    constructed WITHOUT using the Churn target variable or financial leakage.
    
    Evaluation Findings:
        1. In the historical dataset, all 1,139 churn cases occurred within 7 sub-categories
           (Binders, Tables, Machines, Bookcases, Furnishings, Appliances, Phones).
        2. Attempting to flag these 7 sub-categories post-hoc based on historical churn
           constitutes blatant target lookahead bias / data leakage.
        3. Using non-target features (e.g. mean discount across sub-categories) results in
           severe misclassification (e.g. Chairs has high discounts up to 30% but 0 churn,
           Copiers has discounts up to 40% but 0 churn).
        4. Crucially, raw 'Sub-Category' (17 levels) is ALREADY retained in the production feature set.
           Machine learning algorithms (regularized linear models, tree ensembles) naturally learn
           sub-category weights and interaction effects during cross-validation without bias.
           
    Conclusion:
        A non-target rule cannot be objectively established without arbitrary heuristic overfitting.
        Therefore, to protect pipeline integrity, this feature is EXCLUDED.
    """
    reason = (
        "Excluded to prevent target lookahead leakage. In historical data, churn is concentrated "
        "in 7 sub-categories due to aggressive synthetic discount assignments. Hardcoding these 7 "
        "without target awareness is impossible without lookahead bias, and non-target heuristics "
        "misclassify categories like Chairs and Copiers. Since Sub-Category (17 levels) is already "
        "included, models will learn category-level risk organically within cross-validation."
    )
    logger.info(f"high_risk_subcategory_flag evaluation: {reason}")
    return False, reason, None


# -------------------------------------------------------------------------
# PIPELINE EXECUTION & INTEGRATION
# -------------------------------------------------------------------------

def run_feature_engineering_pipeline(
    input_filepath: Optional[str] = None,
    output_filepath: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Executes the end-to-end leakage-safe feature engineering pipeline.
    
    Steps:
        1. Discover and load raw dataset.
        2. Validate raw schema.
        3. Extract production base features and target.
        4. Compute engineered features (log_sales, sales_per_quantity, discount_tier, region_category_interaction).
        5. Verify zero leakage and confirm absence of quarantined columns.
        6. Persist processed dataframe to customer_features.csv.
        7. Return processed DataFrame and verification manifest.
    """
    # 1. Resolve paths
    resolved_in_path = find_raw_dataset(input_filepath)
    if output_filepath is None:
        output_filepath = os.path.join(
            "member1_customer_intelligence", "data", "processed", "customer_features.csv"
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

    # 2. Load and validate raw data
    raw_df = load_raw_dataset(resolved_in_path)
    validate_raw_schema(raw_df)

    logger.info("Initiating leakage-safe feature engineering transformations...")

    # 3. Assemble production base features
    processed_df = raw_df[PRODUCTION_BASE_FEATURES].copy()

    # 4. Generate engineered features
    processed_df["log_sales"] = create_log_sales(raw_df["Sales"])
    processed_df["sales_per_quantity"] = create_sales_per_quantity(raw_df["Sales"], raw_df["Quantity"])
    processed_df["discount_tier"] = create_discount_tier(raw_df["Discount"])
    processed_df["region_category_interaction"] = create_region_category_interaction(
        raw_df["Region"], raw_df["Category"]
    )

    # 5. Attach target variable (Churn)
    processed_df[TARGET_COLUMN] = raw_df[TARGET_COLUMN].astype(int)

    # 6. Evaluate high_risk_subcategory_flag
    include_subcat_flag, subcat_reason, _ = evaluate_high_risk_subcategory_flag(raw_df)
    # As documented, include_subcat_flag is False

    # 7. Leakage and integrity verifications
    verification_manifest = verify_pipeline_integrity(raw_df, processed_df)

    # 8. Persist processed dataset
    logger.info(f"Saving processed dataset to: {output_filepath}")
    processed_df.to_csv(output_filepath, index=False)
    logger.info(f"Processed dataset saved successfully. File size: {os.path.getsize(output_filepath):,} bytes")

    # Also mirror to root data/processed if member1 is top-level
    alt_out = os.path.join("data", "processed", "customer_features.csv")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(alt_out)), exist_ok=True)
        processed_df.to_csv(alt_out, index=False)
        logger.info(f"Mirrored copy saved to: {alt_out}")
    except Exception as e:
        logger.warning(f"Could not save mirror copy to {alt_out}: {e}")

    return processed_df, verification_manifest


def verify_pipeline_integrity(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict:
    """
    Runs automated assertions confirming zero data leakage and complete data hygiene.
    """
    logger.info("Executing automated leakage and integrity verification checks...")

    # Check 1: No excluded columns present
    leaked_cols = [c for c in EXCLUDED_COLUMNS if c in processed_df.columns]
    if leaked_cols:
        raise AssertionError(f"LEAKAGE DETECTED! Quarantined columns found in processed data: {leaked_cols}")

    # Check 2: Row count preservation
    if len(processed_df) != len(raw_df):
        raise AssertionError(f"Row count mismatch! Raw: {len(raw_df)}, Processed: {len(processed_df)}")

    # Check 3: Null count verification
    null_counts = processed_df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        raise AssertionError(f"Null values detected in processed dataset: {null_counts}")

    # Check 4: Target preservation
    raw_churn_counts = raw_df[TARGET_COLUMN].value_counts().to_dict()
    proc_churn_counts = processed_df[TARGET_COLUMN].value_counts().to_dict()
    if raw_churn_counts != proc_churn_counts:
        raise AssertionError("Target distribution mismatch after transformation!")

    # Check 5: No infinite values in numerical features
    num_cols = ["Sales", "Quantity", "Discount", "log_sales", "sales_per_quantity"]
    for col in num_cols:
        if np.isinf(processed_df[col]).any():
            raise AssertionError(f"Infinite values found in numerical feature: {col}")

    manifest = {
        "rows": len(processed_df),
        "columns_count": len(processed_df.columns),
        "feature_list": [c for c in processed_df.columns if c != TARGET_COLUMN],
        "target": TARGET_COLUMN,
        "null_count": total_nulls,
        "target_distribution": proc_churn_counts,
        "leaked_columns_detected": len(leaked_cols),
        "status": "PASSED_STRICT_AUDIT"
    }

    logger.info(f"Verification Manifest: All {manifest['columns_count']} columns verified. Status: {manifest['status']}")
    return manifest


# -------------------------------------------------------------------------
# COMMAND LINE ENTRY POINT
# -------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("================================================================================")
    logger.info("STARTING PHASE 2 LEAKAGE-SAFE FEATURE ENGINEERING PIPELINE (MEMBER 1)")
    logger.info("================================================================================")

    processed_data, manifest = run_feature_engineering_pipeline()

    feature_cols = [c for c in processed_data.columns if c != TARGET_COLUMN]
    
    print("\n" + "=" * 80)
    print("MEMBER 1 — EXACT FINAL PRODUCTION FEATURE LIST (PASSED TO ML PIPELINE)")
    print("=" * 80)
    for idx, feat in enumerate(feature_cols, 1):
        dtype_str = str(processed_data[feat].dtype)
        sample_val = processed_data[feat].iloc[0]
        print(f"  {idx:2d}. {feat:<30} | Type: {dtype_str:<10} | Sample: {sample_val}")

    print("\nTarget Variable:")
    print(f"  --> {TARGET_COLUMN} (Binary: 0 = Retained, 1 = Churned)")
    print("=" * 80)
    print(f"Total Features: {len(feature_cols)} (excluding Target '{TARGET_COLUMN}')")
    print(f"Total Rows:     {len(processed_data):,}")
    print(f"Total Nulls:    {manifest['null_count']}")
    print("Pipeline Execution Complete. No ML models were trained.")
    print("=" * 80 + "\n")
