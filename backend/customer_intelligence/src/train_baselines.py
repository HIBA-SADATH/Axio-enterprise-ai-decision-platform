# train_baselines.py
"""
Baseline model training and evaluation for Member 1 – Customer Intelligence & Churn Prediction.

- Uses the already‑preprocessed NumPy arrays returned by `prepare_data()`.
- Trains three approaches: Majority‑class baseline, Logistic Regression, Random Forest.
- Saves trained models (including the fitted preprocessing pipeline) and a detailed report.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Ensure the repository root is on the Python path so we can import the module.
# The script lives in member1_customer_intelligence/src/, so the repo root is two levels up.
# ---------------------------------------------------------------------------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Import the preprocessing helper.
from member1_customer_intelligence.src.preprocessing import prepare_data

def evaluate(y_true, y_pred, y_proba=None):
    """Compute a dictionary of evaluation metrics.
    `y_proba` is the probability of the positive class (1) if available.
    """
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

def main():
    payload = prepare_data()
    X_train = payload["X_train"]
    X_test = payload["X_test"]
    y_train = payload["y_train"]
    y_test = payload["y_test"]
    pipeline = payload["pipeline"]

    # Majority‑class baseline (predict 0 for everything)
    majority_pred = np.zeros_like(y_test, dtype=int)
    majority_metrics = evaluate(y_test, majority_pred)

    # Logistic Regression
    logreg = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    )
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]
    logreg_metrics = evaluate(y_test, logreg_pred, logreg_proba)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate(y_test, rf_pred, rf_proba)

    # Save models (pipeline + classifier)
    models_dir = os.path.join(repo_root, "member1_customer_intelligence", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "model": logreg}, os.path.join(models_dir, "logistic_regression.joblib"))
    joblib.dump({"pipeline": pipeline, "model": rf}, os.path.join(models_dir, "random_forest.joblib"))

    # Reporting
    reports_dir = os.path.join(repo_root, "member1_customer_intelligence", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    md_path = os.path.join(reports_dir, "phase4_baseline_model_evaluation.md")
    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write("# Phase 4 – Baseline Model Evaluation\n\n")
        f_md.write(f"- Training rows: {X_train.shape[0]}\n")
        f_md.write(f"- Test rows: {X_test.shape[0]}\n")
        f_md.write(f"- Feature dimension after encoding: {X_train.shape[1]}\n\n")
        def write_section(name, metrics, desc=""):
            f_md.write(f"## {name}\n")
            if desc:
                f_md.write(f"*{desc}*\n")
            f_md.write("| Metric | Value |\n|---|---|\n")
            f_md.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
            f_md.write(f"| Precision (class 1) | {metrics['precision']:.4f} |\n")
            f_md.write(f"| Recall (class 1) | {metrics['recall']:.4f} |\n")
            f_md.write(f"| F1‑score (class 1) | {metrics['f1']:.4f} |\n")
            if metrics['roc_auc'] is not None:
                f_md.write(f"| ROC‑AUC | {metrics['roc_auc']:.4f} |\n")
            else:
                f_md.write("| ROC‑AUC | N/A (baseline) |\n")
            if metrics['average_precision'] is not None:
                f_md.write(f"| Average Precision | {metrics['average_precision']:.4f} |\n")
            else:
                f_md.write("| Average Precision | N/A (baseline) |\n")
            f_md.write("\n**Confusion Matrix** (rows=actual, cols=predicted):\n\n```")
            f_md.write(str(np.array(metrics['confusion_matrix'])) + "\n``\n\n")
            f_md.write("**Classification report** (per‑class):\n\n| Class | Precision | Recall | F1‑score | Support |\n|---|---|---|---|---|\n")
            for cls, vals in metrics['classification_report'].items():
                if cls in ["accuracy", "macro avg", "weighted avg"]:
                    continue
                f_md.write(f"| {cls} | {vals['precision']:.4f} | {vals['recall']:.4f} | {vals['f1-score']:.4f} | {int(vals['support'])} |\n")
            f_md.write("\n---\n\n")
        write_section("Majority‑class baseline", majority_metrics, "Predicts class 0 for every example.")
        write_section("Logistic Regression", logreg_metrics, "LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)")
        write_section("Random Forest", rf_metrics, "RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)")

    # CSV summary
    csv_path = os.path.join(reports_dir, "phase4_metrics.csv")
    df = pd.DataFrame([
        {"model": "majority", **majority_metrics},
        {"model": "logistic_regression", **logreg_metrics},
        {"model": "random_forest", **rf_metrics},
    ])
    df.to_csv(csv_path, index=False)

    print("--- Baseline evaluation completed ---")
    print(f"Markdown report: {md_path}")
    print(f"CSV summary: {csv_path}")
    print(f"Logistic Regression saved: {os.path.join(models_dir, 'logistic_regression.joblib')}")
    print(f"Random Forest saved: {os.path.join(models_dir, 'random_forest.joblib')}")

if __name__ == "__main__":
    main()
