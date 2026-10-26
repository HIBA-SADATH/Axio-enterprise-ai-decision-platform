# Phase 4.1 – Discount Sensitivity Experiments

These experiments keep the original stratified split (random_state=42) and only vary the inclusion of `Discount` and the engineered `discount_tier` feature.

## Experiment Overview

| Exp | Description | Features before encoding | Features after encoding |
|-----|-------------|--------------------------|--------------------------|
| A | Reference – all approved features | 14 | 105 |
| B | Drop engineered tier | 13 | 101 |
| C | Drop Discount and tier | 12 | 100 |

## Logistic Regression Metrics

| Exp | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC‑AUC | Avg‑Prec |
|-----|----------|----------------|------------|--------|---------|----------|
| A | 0.9975 | 0.9785 | 1.0000 | 0.9892 | 1.0000 | 0.9999 |
| B | 0.9965 | 0.9702 | 1.0000 | 0.9849 | 1.0000 | 0.9999 |
| C | 0.9845 | 0.8803 | 1.0000 | 0.9363 | 0.9947 | 0.8801 |

## Random Forest Metrics

| Exp | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC‑AUC | Avg‑Prec |
|-----|----------|----------------|------------|--------|---------|----------|
| A | 0.9995 | 0.9956 | 1.0000 | 0.9978 | 1.0000 | 0.9999 |
| B | 0.9995 | 0.9956 | 1.0000 | 0.9978 | 1.0000 | 0.9999 |
| C | 0.9960 | 0.9741 | 0.9912 | 0.9826 | 0.9998 | 0.9984 |

### Confusion Matrices – Logistic Regression

**Exp A**
```[[1763    5]
 [   0  228]]```

**Exp B**
```[[1761    7]
 [   0  228]]```

**Exp C**
```[[1737   31]
 [   0  228]]```

### Confusion Matrices – Random Forest

**Exp A**
```[[1767    1]
 [   0  228]]```

**Exp B**
```[[1767    1]
 [   0  228]]```

**Exp C**
```[[1762    6]
 [   2  226]]```

---
*All experiments used the exact same train/test split and preprocessing pipeline. No hyper‑parameter tuning was performed.*
