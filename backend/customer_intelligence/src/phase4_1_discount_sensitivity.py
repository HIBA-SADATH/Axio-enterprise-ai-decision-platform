# src/phase4_1_discount_sensitivity.py
"""
Phase 4.1 – Discount Sensitivity Experiments

Runs three controlled experiments (A, B, C) that vary the inclusion of the
`Discount` raw column and the engineered `discount_tier` feature.

* Experiment A – reference (all approved features).
* Experiment B – drop `discount_tier` only.
* Experiment C – drop both `Discount` and `discount_tier`.

The script uses the same stratified 80/20 split (random_state=42) and the
identical preprocessing pipeline logic from `preprocessing.py`.  Model
artifacts and a markdown report are written to the `reports/` directory with
explicit experiment identifiers so that existing Phase 4 results are preserved.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Helper to compute metrics (same as in train_baselines.py)
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["average_precision"] = average_precision_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics

# ---------------------------------------------------------------------------
# Load the processed feature CSV (produced by the existing feature‑engineering
# pipeline).  This file already contains the 14 approved features plus the
# target column `Churn`.
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[2]
processed_csv = repo_root / "member1_customer_intelligence" / "data" / "processed" / "customer_features.csv"
df = pd.read_csv(processed_csv)
TARGET_COL = "Churn"

# Original feature list (all columns except target)
all_features = [c for c in df.columns if c != TARGET_COL]

# ---------------------------------------------------------------------------
# Original categorical / numeric definitions (mirrored from preprocessing.py)
# ---------------------------------------------------------------------------
CATEGORICAL_ORIG = [
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
NUMERICAL_ORIG = ["Sales", "Quantity", "Discount", "log_sales", "sales_per_quantity"]

def build_pipeline(selected_features):
    cat_feats = [f for f in CATEGORICAL_ORIG if f in selected_features]
    num_feats = [f for f in NUMERICAL_ORIG if f in selected_features]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, cat_feats),
            ("num", numeric_transformer, num_feats),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor)])

# ---------------------------------------------------------------------------
# Define experiments
# ---------------------------------------------------------------------------
experiments = {
    "A": {"desc": "Reference – all approved features", "features": all_features},
    "B": {"desc": "Drop engineered tier", "features": [f for f in all_features if f != "discount_tier"]},
    "C": {"desc": "Drop Discount and tier", "features": [f for f in all_features if f not in {"Discount", "discount_tier"}]},
}

results = {}
for exp_key, cfg in experiments.items():
    feats = cfg["features"]
    X = df[feats]
    y = df[TARGET_COL]
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline = build_pipeline(feats)
    pipeline.fit(X_train_df)
    X_train = pipeline.transform(X_train_df)
    X_test = pipeline.transform(X_test_df)
    # Logistic Regression
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42, solver="lbfgs")
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]
    logreg_metrics = evaluate(y_test, logreg_pred, logreg_proba)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate(y_test, rf_pred, rf_proba)
    # Store results
    results[exp_key] = {
        "features_used": feats,
        "num_features_before": len(feats),
        "num_features_after": X_train.shape[1],
        "logreg": logreg_metrics,
        "rf": rf_metrics,
    }
    # Save model artifacts (pipeline + model) for reproducibility
    models_dir = repo_root / "member1_customer_intelligence" / "models"
    os.makedirs(models_dir, exist_ok=True)
    import joblib
    joblib.dump({"pipeline": pipeline, "model": logreg}, models_dir / f"logreg_exp_{exp_key}.joblib")
    joblib.dump({"pipeline": pipeline, "model": rf}, models_dir / f"rf_exp_{exp_key}.joblib")

# ---------------------------------------------------------------------------
# Write a markdown report summarising the experiments
# ---------------------------------------------------------------------------
report_path = repo_root / "member1_customer_intelligence" / "reports" / "phase4_1_discount_sensitivity.md"
csv_path = repo_root / "member1_customer_intelligence" / "reports" / "phase4_1_discount_sensitivity.csv"
with open(report_path, "w", encoding="utf-8") as f_md:
    f_md.write("# Phase 4.1 – Discount Sensitivity Experiments\n\n")
    f_md.write("These experiments keep the original stratified split (random_state=42) and only vary the inclusion of `Discount` and the engineered `discount_tier` feature.\n\n")
    f_md.write("## Experiment Overview\n\n| Exp | Description | Features before encoding | Features after encoding |\n|-----|-------------|--------------------------|--------------------------|\n")
    for k, cfg in experiments.items():
        f_md.write(f"| {k} | {cfg['desc']} | {len(cfg['features'])} | {results[k]['num_features_after']} |\n")
    f_md.write("\n## Logistic Regression Metrics\n\n| Exp | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC‑AUC | Avg‑Prec |\n|-----|----------|----------------|------------|--------|---------|----------|\n")
    for k in experiments:
        m = results[k]["logreg"]
        f_md.write(f"| {k} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} |\n")
    f_md.write("\n## Random Forest Metrics\n\n| Exp | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC‑AUC | Avg‑Prec |\n|-----|----------|----------------|------------|--------|---------|----------|\n")
    for k in experiments:
        m = results[k]["rf"]
        f_md.write(f"| {k} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} |\n")
    f_md.write("\n### Confusion Matrices – Logistic Regression\n\n")
    for k in experiments:
        cm = np.array(results[k]["logreg"]["confusion_matrix"])
        f_md.write(f"**Exp {k}**\n```{cm}```\n\n")
    f_md.write("### Confusion Matrices – Random Forest\n\n")
    for k in experiments:
        cm = np.array(results[k]["rf"]["confusion_matrix"])
        f_md.write(f"**Exp {k}**\n```{cm}```\n\n")
    f_md.write("---\n*All experiments used the exact same train/test split and preprocessing pipeline. No hyper‑parameter tuning was performed.*\n")

# CSV summary for easy import
import csv
with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["Experiment", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Avg_Prec", "Features_Before", "Features_After"])
    for k in experiments:
        for mdl in ["logreg", "rf"]:
            m = results[k][mdl]
            writer.writerow([
                k,
                mdl,
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "",
                f"{m['average_precision']:.4f}" if m['average_precision'] is not None else "",
                results[k]["num_features_before"],
                results[k]["num_features_after"],
            ])

print("--- Discount sensitivity experiments completed ---")
print(f"Report written to: {report_path}")
print(f"CSV summary written to: {csv_path}")
