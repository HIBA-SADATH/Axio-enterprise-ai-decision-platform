# Enterprise AI Decision Intelligence Platform
## Member 1 Module: Customer Intelligence & Churn Prediction

Welcome to the **Customer Intelligence & Churn Prediction** module repository. This module is developed by **Member 1** as an integral component of the 4-person **Enterprise AI Decision Intelligence Platform**.

---

### Team Project Context & Member Responsibilities
The **Enterprise AI Decision Intelligence Platform** is structured into four specialized, loosely-coupled microservice modules designed to integrate through RESTful FastAPI endpoints:

| Team Member | Module Responsibility | Status |
| :--- | :--- | :--- |
| **Member 1 (This Module)** | **Customer Intelligence & Churn Prediction** | **Phase 1 Complete** |
| Member 2 | Financial Risk & Pricing Optimization | Upstream Integration |
| Member 3 | Supply Chain & Inventory Intelligence | Cross-functional Partner |
| Member 4 | Unified Executive Dashboard & Decision Engine | Downstream Consumer |

---

### Full Module Scope (Roadmap)
1. **Customer Data Analysis** ✅ *(Phase 1)*
2. **Data Cleaning and Validation** ✅ *(Phase 1)*
3. **Exploratory Data Analysis (EDA)** ✅ *(Phase 1)*
4. **Feature Engineering** *(Phase 2)*
5. **Customer Churn Prediction (ML Classifiers)** *(Phase 2)*
6. **Model Evaluation & Metric Benchmarking** *(Phase 2)*
7. **Explainable AI using SHAP & Waterfalls** *(Phase 2)*
8. **Customer Risk Scoring Engine** *(Phase 3)*
9. **FastAPI Inference Microservice Endpoints** *(Phase 3)*
10. **Interactive Customer Intelligence Dashboard** *(Phase 4)*

---

### Phase 1 Summary: Data Understanding & EDA

In Phase 1, we executed a comprehensive investigation of `Cleaned_Superstore.csv`.

#### 1. Dataset Overview
- **Rows:** $9,977$ records
- **Columns:** $18$ features
- **Missing Values:** $0$ missing values across all columns
- **Duplicate Rows:** $0$ duplicates

#### 2. Customer ID Uniqueness & Aggregation Strategy
- **Audit Result:** $9,977$ unique `Customer ID` values across $9,977$ rows.
- **Decision:** **NO aggregation by Customer ID is performed.** Every record is already an individual customer account. Aggregation would be an unnecessary $1:1$ mapping.

#### 3. Churn Target Distribution
- **Retained ($0$):** $8,838$ customers ($88.58\%$)
- **Churned ($1$):** $1,139$ customers ($11.42\%$)
- **Imbalance Ratio:** $\approx 7.76:1$. Phase 2 will prioritize ROC-AUC, PR-AUC, and F1-score rather than raw accuracy.

#### 4. Dominant Drivers & Behavioral Findings
- **Promotional Discounting:** Heavy discounting is the primary driver of customer attrition ($r = +0.848$). No customer with a discount below $32\%$ churned, while $100\%$ of customers receiving discounts $\ge 45\%$ churned.
- **Geographic Disparity:** Central region has the highest churn rate ($21.13\%$), while the West region has the lowest ($3.70\%$).
- **Product Vulnerabilities:** Churn is concentrated in 7 sub-categories, led by Binders ($40.21\%$), Tables ($38.24\%$), and Machines ($37.39\%$). Ten sub-categories have $0.00\%$ churn.
- **Profit Margin Deficit:** Every churned customer generated negative net operating profit ($Profit < 0$).

#### 5. Data Leakage Diagnosis
- **`Profit Status`:** When `Profit Status == 'Profit'`, churn is $0.00\%$. Using this post-transaction field directly in an ML model creates severe data leakage.
- **`Profit Margin` / `Profit`:** Highly correlated ($-0.814$) post-hoc features. To ensure enterprise robustness, Phase 2 models will benchmark both an operational pre-transaction feature set and a full financial set.

---

### Project Structure

```
member1_customer_intelligence/
│
├── data/
│   ├── raw/
│   │   ├── Cleaned_Superstore.csv              # Primary dataset (9,977 records)
│   │   └── Cleaned_Superstore(1).csv           # Alternate alias copy
│   └── processed/                              # Output folder for Phase 2 engineered data
│
├── notebooks/
│   └── 01_customer_data_analysis.ipynb         # Step-by-step interactive Jupyter notebook
│
├── src/
│   ├── data_analysis.py                        # Python EDA analysis and figure generation script
│   └── generate_notebook.py                    # Programmatic notebook builder
│
├── models/                                     # Destination for Phase 2 serialized models
│
├── reports/
│   ├── customer_eda_report.md                  # Comprehensive Phase 1 EDA report
│   └── figures/                                # 12 high-resolution visual plots (300 DPI)
│       ├── 01_churn_distribution.png
│       ├── 02_churn_by_segment.png
│       ├── 03_churn_by_region.png
│       ├── 04_churn_by_category.png
│       ├── 05_sales_distribution_by_churn.png
│       ├── 06_profit_distribution_by_churn.png
│       ├── 07_discount_distribution_by_churn.png
│       ├── 08_profit_margin_distribution_by_churn.png
│       ├── 09_inventory_risk_vs_churn.png
│       ├── 10_correlation_heatmap.png
│       ├── 11_churn_by_subcategory.png
│       └── 12_profit_status_vs_churn.png
│
├── requirements.txt                            # Python dependencies
└── README.md                                   # This documentation file
```

---

### Quickstart & Reproduction Guide

#### 1. Environment Setup
Ensure Python 3.10+ (or 3.14) is installed. Install required packages:
```bash
pip install -r requirements.txt
```

#### 2. Run the Analysis Script
To execute all statistical audits and automatically generate the 12 visualization figures:
```bash
python src/data_analysis.py
```
*(On Windows systems where `python` points to the Windows Store shim, invoke using `py src/data_analysis.py`)*

#### 3. Inspect the Interactive Notebook
Launch Jupyter to explore `notebooks/01_customer_data_analysis.ipynb`:
```bash
jupyter notebook notebooks/01_customer_data_analysis.ipynb
```

#### 4. Read the Technical Report
Review [customer_eda_report.md](reports/customer_eda_report.md) for full statistical tables, charts, business interpretations, and interview-ready technical breakdowns.
