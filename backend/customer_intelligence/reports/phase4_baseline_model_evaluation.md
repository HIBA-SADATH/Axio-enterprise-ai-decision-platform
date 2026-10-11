# Phase 4 – Baseline Model Evaluation

- Training rows: 7981
- Test rows: 1996
- Feature dimension after encoding: 105

## Majority‑class baseline
*Predicts class 0 for every example.*
| Metric | Value |
|---|---|
| Accuracy | 0.8858 |
| Precision (class 1) | 0.0000 |
| Recall (class 1) | 0.0000 |
| F1‑score (class 1) | 0.0000 |
| ROC‑AUC | N/A (baseline) |
| Average Precision | N/A (baseline) |

**Confusion Matrix** (rows=actual, cols=predicted):

```[[1768    0]
 [ 228    0]]
``

**Classification report** (per‑class):

| Class | Precision | Recall | F1‑score | Support |
|---|---|---|---|---|
| 0 | 0.8858 | 1.0000 | 0.9394 | 1768 |
| 1 | 0.0000 | 0.0000 | 0.0000 | 228 |

---

## Logistic Regression
*LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)*
| Metric | Value |
|---|---|
| Accuracy | 0.9975 |
| Precision (class 1) | 0.9785 |
| Recall (class 1) | 1.0000 |
| F1‑score (class 1) | 0.9892 |
| ROC‑AUC | 1.0000 |
| Average Precision | 0.9999 |

**Confusion Matrix** (rows=actual, cols=predicted):

```[[1763    5]
 [   0  228]]
``

**Classification report** (per‑class):

| Class | Precision | Recall | F1‑score | Support |
|---|---|---|---|---|
| 0 | 1.0000 | 0.9972 | 0.9986 | 1768 |
| 1 | 0.9785 | 1.0000 | 0.9892 | 228 |

---

## Random Forest
*RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)*
| Metric | Value |
|---|---|
| Accuracy | 0.9995 |
| Precision (class 1) | 0.9956 |
| Recall (class 1) | 1.0000 |
| F1‑score (class 1) | 0.9978 |
| ROC‑AUC | 1.0000 |
| Average Precision | 0.9999 |

**Confusion Matrix** (rows=actual, cols=predicted):

```[[1767    1]
 [   0  228]]
``

**Classification report** (per‑class):

| Class | Precision | Recall | F1‑score | Support |
|---|---|---|---|---|
| 0 | 1.0000 | 0.9994 | 0.9997 | 1768 |
| 1 | 0.9956 | 1.0000 | 0.9978 | 228 |

---

