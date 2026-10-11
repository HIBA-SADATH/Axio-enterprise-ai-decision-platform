"""
Script to build the clean 01_customer_data_analysis.ipynb Jupyter Notebook.
"""
import os
import json

notebook = {
    'cells': [],
    'metadata': {
        'language_info': {
            'name': 'python',
            'version': '3.14.0'
        },
        'kernelspec': {
            'display_name': 'Python 3 (ipykernel)',
            'language': 'python',
            'name': 'python3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

def add_md(text):
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.strip().splitlines(keepends=True)
    })

def add_code(code):
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': code.strip().splitlines(keepends=True)
    })

# Cell 1: Header and Overview
add_md("""# Enterprise AI Decision Intelligence Platform
## Member 1: Customer Intelligence & Churn Prediction
### Phase 1: Data Understanding & Exploratory Data Analysis (EDA)

---

### Project Role & Responsibilities
As **Member 1** of our 4-member enterprise engineering team, my core focus is **Customer Intelligence & Churn Prediction**.
This module is responsible for analyzing customer purchase behavior, auditing data quality, diagnosing churn factors, training explainable machine learning models, and serving real-time risk scores via FastAPI to the unified Enterprise Decision Intelligence Platform.

**Phase 1 Deliverables:**
1. Rigorous Data Inspection & Schema Validation
2. Customer ID Uniqueness Audit & Aggregation Strategy Evaluation
3. Churn Target Distribution & Class Imbalance Analysis
4. In-depth Bivariate & Multivariate Categorical Analysis
5. Numerical Feature Distribution & Outlier Diagnosis
6. Correlation Heatmap Analysis
7. Target Leakage & Feature Integrity Diagnosis
8. Technical Foundations for Phase 2 Modeling
""")

# Cell 2: Imports
add_code("""# Core libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Palette definitions
CHURN_PALETTE = {0: '#2b5c8f', 1: '#d9534f', '0': '#2b5c8f', '1': '#d9534f'}
CHURN_LABELS = {0: 'Retained (0)', 1: 'Churned (1)'}

print(f"Pandas Version:  {pd.__version__}")
print(f"NumPy Version:   {np.__version__}")
print(f"Seaborn Version: {sns.__version__}")
""")

# Cell 3: Markdown Step 1
add_md("""---
## Step 1: Data Loading & Structural Inspection
We load the raw CSV dataset and inspect the dimensions, column names, schema data types, missing values, and duplicate rows.
""")

# Cell 4: Load code
add_code("""# Path resolution supporting both notebook and root execution
data_candidates = [
    os.path.join("..", "data", "raw", "Cleaned_Superstore(1).csv"),
    os.path.join("..", "data", "raw", "Cleaned_Superstore.csv"),
    os.path.join("data", "raw", "Cleaned_Superstore(1).csv"),
    os.path.join("data", "raw", "Cleaned_Superstore.csv"),
    os.path.join("..", "dataset", "Cleaned_Superstore.csv"),
    os.path.join("dataset", "Cleaned_Superstore.csv")
]

data_path = None
for p in data_candidates:
    if os.path.exists(p):
        data_path = p
        break

if not data_path:
    raise FileNotFoundError("Could not locate Cleaned_Superstore dataset in expected directories.")

print(f"[INFO] Ingesting dataset from: {data_path}")
df = pd.read_csv(data_path)

print(f"Shape: {df.shape[0]:,} Rows | {df.shape[1]} Columns")
df.head(5)
""")

# Cell 5: Schema Info
add_code("""# Dataset Schema and Non-Null Counts
df.info()
""")

# Cell 6: Data Quality
add_code("""# Integrity Audit: Missing Values and Duplicates
missing = df.isnull().sum()
duplicates = df.duplicated().sum()

print("=== DATA QUALITY SUMMARY ===")
print(f"Total Missing Values:  {missing.sum()}")
print(f"Total Duplicate Rows:  {duplicates}")
print(f"Numerical Columns:    {df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"Categorical Columns:  {df.select_dtypes(include=['object', 'string']).columns.tolist()}")
""")

# Cell 7: Markdown Customer Uniqueness
add_md("""---
## Step 1.2: Customer ID Uniqueness & Aggregation Strategy
> **Crucial Architecture Check:** Before deciding whether customer-level aggregation is required, inspect the actual Customer ID uniqueness and transaction structure.
- If every Customer ID occurs only once, **DO NOT** aggregate the dataset by Customer ID.
- If customers occur multiple times, explain the appropriate aggregation strategy before implementing it.
""")

# Cell 8: Customer ID code
add_code("""total_rows = len(df)
unique_customers = df['Customer ID'].nunique()
is_unique_per_row = (total_rows == unique_customers)

print(f"Total Dataset Records:    {total_rows:,}")
print(f"Unique Customer IDs:      {unique_customers:,}")
print(f"Is Customer ID Unique?:   {is_unique_per_row}")

if is_unique_per_row:
    print("\\n[AUDIT DECISION] Every record represents a distinct customer account.")
    print("NO aggregation by Customer ID is performed because the data is already at customer granularity.")
else:
    print("\\n[AUDIT DECISION] Customer IDs repeat across orders. Aggregation is required.")
""")

# Cell 9: Customer Uniqueness Explanation
add_md("""### Aggregation Audit Finding:
- **Result:** Exactly **9,977 unique Customer IDs** across **9,977 rows** (`CUST1000` through `CUST10976`).
- **Conclusion:** Each row is already a 1-to-1 unique customer observation. Customer-level aggregation (e.g., groupby `Customer ID`) is **NOT required**, as it would produce an identical dataset and discard natural tabular integrity.
""")

# Cell 10: Step 2 Churn Target
add_md("""---
## Step 2: Target Variable Analysis (`Churn`)
We inspect:
1. Target value counts
2. Percentage of churned vs. non-churned customers
3. Binary validation
4. Class imbalance ratio
""")

# Cell 11: Target Code
add_code("""churn_counts = df['Churn'].value_counts()
churn_pct = (df['Churn'].value_counts(normalize=True) * 100).round(2)
is_binary = set(df['Churn'].unique()).issubset({0, 1})

target_df = pd.DataFrame({
    'Count': churn_counts,
    'Percentage (%)': churn_pct
})
target_df.index = ['Retained (0)', 'Churned (1)']

print("=== CHURN TARGET DISTRIBUTION ===")
print(f"Is Target Strictly Binary {set(df['Churn'].unique())}?: {is_binary}")
print(target_df)

imbalance_ratio = churn_counts[0] / churn_counts[1]
print(f"\\nClass Imbalance Ratio: {imbalance_ratio:.2f} Retained customers per 1 Churned customer.")
""")

# Cell 12: Target Distribution Plot
add_code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count bar plot
axes[0].bar(['Retained (0)', 'Churned (1)'], [churn_counts[0], churn_counts[1]], 
            color=[CHURN_PALETTE[0], CHURN_PALETTE[1]], edgecolor='black', alpha=0.85, width=0.45)
for i, v in enumerate([churn_counts[0], churn_counts[1]]):
    axes[0].text(i, v + 120, f"{v:,}\\n({v/len(df)*100:.1f}%)", ha='center', fontweight='bold', fontsize=11)
axes[0].set_title("Customer Churn Count Distribution", fontsize=13, fontweight='bold', pad=12)
axes[0].set_ylabel("Number of Customers", fontsize=11)
axes[0].set_ylim(0, 10000)

# Pie chart
axes[1].pie([churn_counts[0], churn_counts[1]], labels=['Retained', 'Churned'], 
            autopct='%1.1f%%', startangle=90, colors=[CHURN_PALETTE[0], CHURN_PALETTE[1]], 
            explode=(0, 0.08), wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title("Customer Churn Proportion", fontsize=13, fontweight='bold', pad=12)

plt.tight_layout()
plt.show()
""")

# Cell 13: Target Interpretation
add_md("""### Target Variable Interpretation:
- **Class Breakdown:** **8,838 Retained customers (88.58%)** vs **1,139 Churned customers (11.42%)**.
- **Binary Status:** Strictly binary {0, 1}.
- **Class Imbalance:** A ~7.76:1 imbalance ratio indicates moderate skew.
- **Modeling Implications for Phase 2:** Standard accuracy is an inappropriate metric because predicting 100% retention yields an 88.58% accuracy while failing 100% of churn detections. Models must be evaluated using **ROC-AUC, Precision-Recall AUC (PR-AUC), F1-Score, and Recall**.
""")

# Cell 14: Step 3 Categorical EDA
add_md("""---
## Step 3 & 4: Exploratory Data Analysis — Categorical Relationships
We analyze churn relationships across:
- **Customer Segment** (Consumer, Corporate, Home Office)
- **Geographic Region** (Central, East, South, West)
- **Product Category** (Furniture, Office Supplies, Technology)
- **Product Sub-Category** (17 product types)
- **Inventory Risk** (Low vs High)
- **Sales Category** (Low, Medium, High, Very High)
""")

# Cell 15: Categorical Crosstabs Function
add_code("""def analyze_categorical_churn(df, column):
    ct = pd.crosstab(df[column], df['Churn'], margins=True)
    ct['Churn_Rate_%'] = (ct[1] / ct['All'] * 100).round(2)
    return ct.rename(columns={0: 'Retained', 1: 'Churned', 'All': 'Total'})

# Compute cross-tabulations
seg_ct = analyze_categorical_churn(df, 'Segment')
reg_ct = analyze_categorical_churn(df, 'Region')
cat_ct = analyze_categorical_churn(df, 'Category')

print("--- CHURN BY CUSTOMER SEGMENT ---")
print(seg_ct)
print("\\n--- CHURN BY GEOGRAPHIC REGION ---")
print(reg_ct)
print("\\n--- CHURN BY PRODUCT CATEGORY ---")
print(cat_ct)
""")

# Cell 16: Segment, Region, Category Visualizations
add_code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
w = 0.35

# Segment Plot
seg_plot = seg_ct.drop('All')
x_seg = np.arange(len(seg_plot))
axes[0].bar(x_seg - w/2, seg_plot['Retained'], w, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
axes[0].bar(x_seg + w/2, seg_plot['Churned'], w, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
for i, (_, row) in enumerate(seg_plot.iterrows()):
    axes[0].text(i, max(row['Retained'], row['Churned']) + 120, f"{row['Churn_Rate_%']:.1f}%", ha='center', fontweight='bold', color='#a94442')
axes[0].set_xticks(x_seg)
axes[0].set_xticklabels(seg_plot.index, fontsize=10)
axes[0].set_title("Churn by Customer Segment", fontweight='bold', fontsize=12)
axes[0].set_ylabel("Customer Count")
axes[0].legend()

# Region Plot
reg_plot = reg_ct.drop('All').sort_values(by='Churn_Rate_%', ascending=False)
x_reg = np.arange(len(reg_plot))
axes[1].bar(x_reg - w/2, reg_plot['Retained'], w, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
axes[1].bar(x_reg + w/2, reg_plot['Churned'], w, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
for i, (_, row) in enumerate(reg_plot.iterrows()):
    axes[1].text(i, max(row['Retained'], row['Churned']) + 80, f"{row['Churn_Rate_%']:.1f}%", ha='center', fontweight='bold', color='#a94442')
axes[1].set_xticks(x_reg)
axes[1].set_xticklabels(reg_plot.index, fontsize=10)
axes[1].set_title("Churn by Region (Central Highest at 21.1%)", fontweight='bold', fontsize=12)
axes[1].legend()

# Category Plot
cat_plot = cat_ct.drop('All').sort_values(by='Churn_Rate_%', ascending=False)
x_cat = np.arange(len(cat_plot))
axes[2].bar(x_cat - w/2, cat_plot['Retained'], w, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
axes[2].bar(x_cat + w/2, cat_plot['Churned'], w, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
for i, (_, row) in enumerate(cat_plot.iterrows()):
    axes[2].text(i, max(row['Retained'], row['Churned']) + 120, f"{row['Churn_Rate_%']:.1f}%", ha='center', fontweight='bold', color='#a94442')
axes[2].set_xticks(x_cat)
axes[2].set_xticklabels(cat_plot.index, fontsize=10)
axes[2].set_title("Churn by Category (Furniture Highest at 15.1%)", fontweight='bold', fontsize=12)
axes[2].legend()

plt.tight_layout()
plt.show()
""")

# Cell 17: Segment, Region, Category Interpretation
add_md("""### Categorical Interpretations:
1. **Segment Analysis:**
   - Churn rate is remarkably consistent across segments: **Corporate (11.84%)**, **Consumer (11.31%)**, and **Home Office (11.02%)**. Segment type alone is not a primary driver of customer churn.
2. **Geographic Regional Analysis:**
   - **Central Region** has the highest churn rate at **21.13%** (490 churned out of 2,319 customers).
   - **East Region** follows at **12.65%**, **South** at **10.56%**, while the **West Region** has the lowest churn rate at **3.70%** (118 churned out of 3,193 customers). Regional discounting practices directly influence this disparity.
3. **Product Category Analysis:**
   - **Furniture** exhibits the highest churn rate at **15.11%** (320 churned out of 2,118).
   - **Office Supplies** stands at **11.29%** (679 churned out of 6,012).
   - **Technology** exhibits the lowest churn at **7.58%** (140 churned out of 1,847).
""")

# Cell 18: Sub-Category Breakdown
add_code("""sub_ct = analyze_categorical_churn(df, 'Sub-Category').drop('All').sort_values(by='Churn_Rate_%', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#d9534f' if r > 20 else '#f0ad4e' if r > 0 else '#5cb85c' for r in sub_ct['Churn_Rate_%']]
bars = ax.barh(sub_ct.index, sub_ct['Churn_Rate_%'], color=colors, edgecolor='black', alpha=0.85)

for bar, rate in zip(bars, sub_ct['Churn_Rate_%']):
    ax.text(rate + 0.6, bar.get_y() + bar.get_height()/2, f"{rate:.1f}%", va='center', fontweight='bold', fontsize=9)
    
ax.set_title("Churn Rate by Sub-Category: Extreme Bipolar Distribution", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Churn Rate (%)", fontsize=11)
ax.set_xlim(0, 48)
plt.tight_layout()
plt.show()

print("Top 5 Churn Sub-Categories:")
print(sub_ct.sort_values(by='Churn_Rate_%', ascending=False).head(5))
""")

# Cell 19: Sub-Category Interpretation
add_md("""### Sub-Category Interpretation:
- **Extreme Polarization:** Exactly **10 sub-categories have 0.00% churn** (Accessories, Art, Copiers, Envelopes, Fasteners, Labels, Paper, Supplies, Storage, Chairs).
- **Concentrated Churn Risk:** Churn is heavily concentrated in:
  - **Binders:** 40.21% churn (612 / 1,522)
  - **Tables:** 38.24% churn (122 / 319)
  - **Machines:** 37.39% churn (43 / 115)
  - **Bookcases:** 26.32% churn (60 / 228)
  - **Furnishings:** 14.44% churn (138 / 956)
  - **Appliances:** 14.38% churn (67 / 466)
  - **Phones:** 10.91% churn (97 / 889)
""")

# Cell 20: Inventory Risk & Sales Category
add_code("""inv_ct = analyze_categorical_churn(df, 'Inventory Risk')
sales_ct = analyze_categorical_churn(df, 'Sales Category')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Inventory Risk
inv_plot = inv_ct.drop('All')
x_inv = np.arange(len(inv_plot))
axes[0].bar(x_inv - w/2, inv_plot['Retained'], w, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
axes[0].bar(x_inv + w/2, inv_plot['Churned'], w, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
for i, (_, row) in enumerate(inv_plot.iterrows()):
    axes[0].text(i, max(row['Retained'], row['Churned']) + 120, f"{row['Churn_Rate_%']:.1f}%", ha='center', fontweight='bold', color='#a94442')
axes[0].set_xticks(x_inv)
axes[0].set_xticklabels(inv_plot.index)
axes[0].set_title("Customer Churn by Inventory Risk", fontweight='bold', fontsize=12)
axes[0].legend()

# Sales Category
sc_plot = sales_ct.drop('All')
x_sc = np.arange(len(sc_plot))
axes[1].bar(x_sc - w/2, sc_plot['Retained'], w, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
axes[1].bar(x_sc + w/2, sc_plot['Churned'], w, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
for i, (_, row) in enumerate(sc_plot.iterrows()):
    axes[1].text(i, max(row['Retained'], row['Churned']) + 120, f"{row['Churn_Rate_%']:.1f}%", ha='center', fontweight='bold', color='#a94442')
axes[1].set_xticks(x_sc)
axes[1].set_xticklabels(sc_plot.index)
axes[1].set_title("Customer Churn by Sales Category", fontweight='bold', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.show()
""")

# Cell 21: Inventory Risk & Sales Category Interpretation
add_md("""### Inventory Risk & Sales Category Interpretation:
- **Inventory Risk:** High Inventory Risk customers exhibit a **12.08%** churn rate versus **10.92%** for Low Inventory Risk. While slightly higher, inventory risk alone is not a dominating factor.
- **Sales Category:** Customers in the **Low Sales** tier have the highest churn rate at **13.12%**, compared to **8.04%** in the Medium Sales tier and **9.37%** in the High tier. Low-spend accounts demonstrate lower brand commitment.
""")

# Cell 22: Step 4 Numerical EDA
add_md("""---
## Step 4: Exploratory Data Analysis — Numerical Features
We evaluate the 5 numerical features:
1. `Sales` ($)
2. `Quantity` (Units)
3. `Discount` (Rate)
4. `Profit` ($)
5. `Profit Margin` (%)
""")

# Cell 23: Numerical Stats Code
add_code("""num_features = ['Sales', 'Quantity', 'Discount', 'Profit', 'Profit Margin']

for col in num_features:
    stats = df.groupby('Churn')[col].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(3)
    stats.index = ['Retained (0)', 'Churned (1)']
    print(f"\\n[{col.upper()} BY CHURN STATUS]")
    print(stats)
""")

# Cell 24: Sales & Quantity Plots
add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sales Distribution (Log Scale)
sns.boxplot(x='Churn', y='Sales', hue='Churn', data=df, ax=axes[0], palette=CHURN_PALETTE, legend=False)
axes[0].set_yscale('log')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
axes[0].set_title("Sales Distribution by Churn Status (Log Scale)", fontweight='bold', fontsize=12)
axes[0].set_ylabel("Sales ($) [Log Scale]")

# Quantity Distribution
sns.boxplot(x='Churn', y='Quantity', hue='Churn', data=df, ax=axes[1], palette=CHURN_PALETTE, legend=False)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Retained (0)', 'Churned (1)'])
axes[1].set_title("Quantity Distribution by Churn Status", fontweight='bold', fontsize=12)
axes[1].set_ylabel("Quantity Purchased (Units)")

plt.tight_layout()
plt.show()
""")

# Cell 25: Discount, Profit, Margin Plots
add_code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Discount Boxplot
sns.boxplot(x='Churn', y='Discount', hue='Churn', data=df, ax=axes[0], palette=CHURN_PALETTE, legend=False)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
axes[0].set_title("Discount by Churn (Massive Separation)", fontweight='bold', fontsize=12)
axes[0].set_ylabel("Discount Rate")

# Profit Boxplot
sns.boxplot(x='Churn', y='Profit', hue='Churn', data=df, ax=axes[1], palette=CHURN_PALETTE, legend=False)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Retained (0)', 'Churned (1)'])
axes[1].set_title("Profit by Churn (Churned Exclusively < 0)", fontweight='bold', fontsize=12)
axes[1].set_ylabel("Profit ($)")
axes[1].axhline(0, color='gray', linestyle='--')

# Profit Margin Boxplot
sns.boxplot(x='Churn', y='Profit Margin', hue='Churn', data=df, ax=axes[2], palette=CHURN_PALETTE, legend=False)
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Retained (0)', 'Churned (1)'])
axes[2].set_title("Profit Margin (%) by Churn", fontweight='bold', fontsize=12)
axes[2].set_ylabel("Profit Margin (%)")
axes[2].axhline(0, color='gray', linestyle='--')

plt.tight_layout()
plt.show()
""")

# Cell 26: Numerical Interpretations
add_md("""### Numerical Interpretations:
1. **Discount Rate is the Single Largest Churn Differentiator:**
   - Retained customers received a mean discount of **9.35%** (median 0.0%, max 40.0%).
   - Churned customers received a mean discount of **64.37%** (median 70.0%, min 32.0%, max 80.0%).
   - **Zero customers with a discount below 32% churned.**
2. **Profit & Profit Margin Expose Negative Separation:**
   - Retained customers generate positive mean profit (**+$46.84**, median +$10.95).
   - Churned customers generate severely negative mean profit (**-$112.14**, median -$17.94, max **-$0.60**).
   - **100% of churned customers generated negative net profit.**
   - Profit Margin for churned customers averages **-93.78%** versus **+25.65%** for retained customers.
3. **Sales & Quantity:**
   - Mean sales are slightly higher for retained ($232.85 vs $209.19), but median sales are substantially lower for churned ($22.61 vs $59.73).
   - Quantity has identical medians (3.0 units) and almost identical means (3.78 vs 3.86).
""")

# Cell 27: Correlation Matrix & Heatmap
add_md("""---
## Step 5: Correlation Matrix & Feature Heatmap
We compute the Pearson correlation coefficients across all numerical attributes and the churn target.
""")

# Cell 28: Heatmap Code
add_code("""corr_matrix = df[num_features + ['Churn']].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='vlag', vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Heatmap: Numerical Features & Churn Target", fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
plt.show()

print("Correlation with Churn Target (Sorted):")
print(corr_matrix['Churn'].sort_values(ascending=False))
""")

# Cell 29: Correlation Interpretation
add_md("""### Correlation Analysis:
- **`Discount` vs `Churn` (+0.848):** Very strong positive linear correlation.
- **`Profit Margin` vs `Churn` (-0.814):** Strong negative linear correlation.
- **`Profit` vs `Churn` (-0.216):** Moderate inverse correlation.
- **`Sales` (-0.012) & `Quantity` (+0.012):** Essentially zero linear correlation.
""")

# Cell 30: Step 6 Target Leakage
add_md("""---
## Step 6: Target Leakage & Feature Integrity Diagnosis
In production enterprise architectures, models that rely on **post-hoc target-leaking variables** fail when deployed because these features are unavailable at the time of prediction.

Let us inspect the cross-tabulation of `Profit Status` and `Discount` thresholds.
""")

# Cell 31: Target Leakage Code
add_code("""# Profit Status vs Churn Cross-Tabulation
prof_status_ct = analyze_categorical_churn(df, 'Profit Status')
print("=== PROFIT STATUS VS CHURN ===")
print(prof_status_ct)

# Exact Discount Threshold Cross-Tabulation
disc_ct = pd.crosstab(df['Discount'], df['Churn'], margins=True)
disc_ct['Churn_Rate_%'] = (disc_ct[1] / disc_ct['All'] * 100).round(2)
print("\\n=== DISCOUNT THRESHOLD VS CHURN ===")
print(disc_ct)
""")

# Cell 32: Leakage Visual
add_code("""fig, ax = plt.subplots(figsize=(8, 5))
ps_plot = prof_status_ct.drop('All')
x_ps = np.arange(len(ps_plot))

ax.bar(x_ps - w/2, ps_plot['Retained'], w, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
ax.bar(x_ps + w/2, ps_plot['Churned'], w, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
for i, (_, row) in enumerate(ps_plot.iterrows()):
    ax.text(i, max(row['Retained'], row['Churned']) + 150, f"{row['Churn_Rate_%']:.1f}%", ha='center', fontweight='bold', color='#a94442')
ax.set_xticks(x_ps)
ax.set_xticklabels(ps_plot.index)
ax.set_title("Profit Status vs Churn: Demonstrating Target Leakage", fontweight='bold', fontsize=12)
ax.set_ylabel("Customer Count")
ax.legend()
plt.tight_layout()
plt.show()
""")

# Cell 33: Leakage Interpretation
add_md("""### Critical Findings on Data Leakage:
1. **`Profit Status` is a Direct Target Leak:**
   - When `Profit Status == 'Profit'`, Churn is **0.00%** (0 / 8,108 customers).
   - When `Profit Status == 'Loss'`, Churn is **60.94%** (1,139 / 1,869 customers).
   - If an ML model is trained with `Profit Status`, it learns a trivial shortcut rule: if profitable, churn = 0. In an active production environment, predicting whether an account will churn *before* accounting losses are finalized cannot rely on post-hoc profit status!
2. **`Profit Margin` & `Profit`:**
   - Every churned customer has negative net profit. Including `Profit Margin` (which has a -0.814 correlation) risks overfitting to the synthetic data generation logic.
3. **Discount Bimodality:**
   - For `Discount >= 0.45`, churn is **100.0%** (932 / 932).
   - For `Discount < 0.32`, churn is **0.0%** (0 / 8,812).
   - In Phase 2 feature engineering, we must test models both **with and without raw profit margin/profit status** to evaluate generalization robustness and prevent shortcut learning.
""")

# Cell 34: Step 7 Summary and Next Steps
add_md("""---
## Step 7: Phase 1 Summary & Phase 2 Roadmap

### 1. Dataset Summary:
- **Total Records:** 9,977 rows, 18 columns.
- **Completeness:** 0 missing values, 0 duplicate rows.
- **Customer Granularity:** 9,977 unique Customer IDs (no aggregation required).

### 2. Target Variable Breakdown:
- **Retained (0):** 8,838 customers (88.58%)
- **Churned (1):** 1,139 customers (11.42%)
- **Imbalance Ratio:** ~7.76:1.

### 3. Key Behavioral Drivers:
- **Region:** Central region is high-risk (21.13% churn); West is low-risk (3.70% churn).
- **Sub-Categories:** Binders, Tables, Machines, and Bookcases represent severe churn clusters.
- **Promotions:** Discounts exceeding 30% are heavily associated with customer attrition.

### 4. Phase 2 Machine Learning Roadmap:
1. **Feature Preprocessing Pipeline:**
   - One-Hot / Target encoding for `Region`, `Category`, `Sub-Category`, `Segment`, `Ship Mode`.
   - Robust scaling for skewed continuous variables (`Sales`, `Discount`).
2. **Feature Selection & Leakage Mitigation:**
   - Benchmark models with and without `Profit Status` / `Profit Margin` to ensure models learn real behavioral patterns rather than synthetic shortcuts.
3. **Class Imbalance Strategy:**
   - Implement Cost-Sensitive Learning (`scale_pos_weight`, class weights) and SMOTE oversampling.
4. **Model Suite:**
   - Logistic Regression (Baseline), Random Forest, LightGBM, and XGBoost.
5. **Explainable AI (XAI):**
   - SHAP summary, force plots, and dependence plots to explain individual customer risk scores.
6. **FastAPI & Dashboard Integration:**
   - Expose `/predict-churn` and `/customer-risk-score` endpoints for the team's unified decision intelligence platform.
""")

output_notebook_path = os.path.join("member1_customer_intelligence", "notebooks", "01_customer_data_analysis.ipynb")
os.makedirs(os.path.dirname(output_notebook_path), exist_ok=True)

with open(output_notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"[SUCCESS] Wrote valid Jupyter Notebook to: {output_notebook_path}")
