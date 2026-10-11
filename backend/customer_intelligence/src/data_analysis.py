"""
Enterprise AI Decision Intelligence Platform
Module: Customer Intelligence & Churn Prediction (Member 1)
Script: data_analysis.py
Phase: Phase 1 — Data Understanding & Exploratory Data Analysis (EDA)

Description:
    Loads the Cleaned_Superstore dataset, performs data validation, structural checks,
    customer uniqueness evaluation, churn distribution analysis, categorical and
    numerical relationship breakdowns, correlation analysis, and generates high-resolution
    visualizations for reports and dashboards.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set visualization theme and aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Palette definitions supporting both integer and string categories
CHURN_PALETTE = {
    0: '#2b5c8f', 
    1: '#d9534f', 
    '0': '#2b5c8f', 
    '1': '#d9534f',
    'Retained': '#2b5c8f',
    'Churned': '#d9534f'
}
CHURN_LABELS = {0: 'Retained (0)', 1: 'Churned (1)'}


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from the specified CSV path with error handling.
    
    Args:
        filepath: Relative or absolute path to the CSV file.
        
    Returns:
        pd.DataFrame containing the loaded data.
    """
    if not os.path.exists(filepath):
        # Check alternative common locations
        alt_paths = [
            os.path.join("data", "raw", "Cleaned_Superstore(1).csv"),
            os.path.join("data", "raw", "Cleaned_Superstore.csv"),
            os.path.join("..", "data", "raw", "Cleaned_Superstore(1).csv"),
            os.path.join("..", "data", "raw", "Cleaned_Superstore.csv"),
            os.path.join("dataset", "Cleaned_Superstore.csv"),
            os.path.join("..", "dataset", "Cleaned_Superstore.csv")
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                filepath = alt
                break
        else:
            raise FileNotFoundError(f"Could not find dataset at '{filepath}' or any alternative paths.")
            
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Successfully loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


def inspect_dataset_structure(df: pd.DataFrame) -> dict:
    """
    Performs initial data understanding checks:
    - Dimensions, column names, data types
    - Missing values and duplicate rows
    - Categorical and numerical column separation
    """
    num_rows, num_cols = df.shape
    missing_counts = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    summary = {
        "rows": num_rows,
        "columns": num_cols,
        "column_names": df.columns.tolist(),
        "missing_counts": missing_counts.to_dict(),
        "total_missing": int(missing_counts.sum()),
        "duplicate_rows": int(duplicate_rows),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols
    }
    return summary


def verify_customer_uniqueness(df: pd.DataFrame) -> dict:
    """
    CRITICAL CHECK:
    Inspects whether Customer IDs are unique per row or whether customers appear multiple times.
    This informs whether customer-level aggregation (e.g. groupby Customer ID) is required.
    """
    if 'Customer ID' not in df.columns:
        raise KeyError("Column 'Customer ID' not found in dataset.")
        
    total_records = len(df)
    unique_customers = df['Customer ID'].nunique()
    is_unique = (unique_customers == total_records)
    
    return {
        "total_records": total_records,
        "unique_customer_ids": unique_customers,
        "is_unique_per_row": is_unique,
        "aggregation_recommendation": (
            "DO NOT aggregate by Customer ID. Every row already corresponds to a single, unique customer "
            f"record ({unique_customers:,} unique IDs across {total_records:,} rows). Aggregation would "
            "be a 1:1 mapping and is completely unnecessary."
            if is_unique else
            "Multiple transactions exist per customer. Aggregation (RFM, totals, averages) is required."
        )
    }


def analyze_churn_target(df: pd.DataFrame) -> dict:
    """
    Analyzes the 'Churn' target variable:
    - Counts and proportions
    - Binary verification
    - Class imbalance assessment
    """
    if 'Churn' not in df.columns:
        raise KeyError("Target column 'Churn' not found in dataset.")
        
    counts = df['Churn'].value_counts()
    percentages = (df['Churn'].value_counts(normalize=True) * 100).round(2)
    is_binary = set(df['Churn'].dropna().unique()).issubset({0, 1})
    
    retained_count = int(counts.get(0, 0))
    churned_count = int(counts.get(1, 0))
    retained_pct = float(percentages.get(0, 0.0))
    churned_pct = float(percentages.get(1, 0.0))
    imbalance_ratio = round(retained_count / max(1, churned_count), 2)
    
    return {
        "is_binary": is_binary,
        "retained_count": retained_count,
        "churned_count": churned_count,
        "retained_pct": retained_pct,
        "churned_pct": churned_pct,
        "imbalance_ratio": imbalance_ratio,
        "assessment": (
            f"Moderate class imbalance: {retained_pct}% retained vs {churned_pct}% churned (~{imbalance_ratio}:1). "
            "In Phase 2, evaluation should prioritize ROC-AUC, Precision, Recall, and PR-AUC over raw Accuracy. "
            "Techniques like class weighting or SMOTE may be tested."
        )
    }


def calculate_churn_by_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Computes churn count, retention count, total, and churn rate percentage for a categorical column.
    """
    ct = pd.crosstab(df[column], df['Churn'], margins=True)
    if 0 not in ct.columns:
        ct[0] = 0
    if 1 not in ct.columns:
        ct[1] = 0
    ct['Churn_Rate_%'] = (ct[1] / ct['All'] * 100).round(2)
    ct = ct.rename(columns={0: 'Retained', 1: 'Churned', 'All': 'Total'})
    return ct


def calculate_numerical_summary_by_churn(df: pd.DataFrame, num_col: str) -> pd.DataFrame:
    """
    Computes descriptive statistics (count, mean, median, std, min, max) for a numerical feature grouped by Churn.
    """
    stats = df.groupby('Churn')[num_col].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(3)
    stats.index = [CHURN_LABELS.get(idx, str(idx)) for idx in stats.index]
    return stats


def generate_all_visualizations(df: pd.DataFrame, output_dir: str = "reports/figures") -> list:
    """
    Creates and saves all required Phase 1 visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # -------------------------------------------------------------
    # 1. Churn Distribution Chart
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    churn_counts = df['Churn'].value_counts()
    axes[0].bar(['Retained (0)', 'Churned (1)'], [churn_counts[0], churn_counts[1]], 
                color=[CHURN_PALETTE[0], CHURN_PALETTE[1]], edgecolor='black', alpha=0.85, width=0.5)
    for i, v in enumerate([churn_counts[0], churn_counts[1]]):
        axes[0].text(i, v + 120, f"{v:,} ({v/len(df)*100:.1f}%)", ha='center', fontweight='bold', fontsize=11)
    axes[0].set_title("Customer Churn Count Distribution", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel("Number of Customers", fontsize=11)
    axes[0].set_ylim(0, 10000)
    
    axes[1].pie([churn_counts[0], churn_counts[1]], labels=['Retained', 'Churned'], 
                autopct='%1.1f%%', startangle=90, colors=[CHURN_PALETTE[0], CHURN_PALETTE[1]], 
                explode=(0, 0.08), wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title("Customer Churn Proportion", fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "01_churn_distribution.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved_files.append(p1)

    # -------------------------------------------------------------
    # 2. Churn by Customer Segment
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    seg_ct = calculate_churn_by_categorical(df, 'Segment').drop('All', errors='ignore')
    x = np.arange(len(seg_ct))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, seg_ct['Retained'], width, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
    bars2 = ax.bar(x + width/2, seg_ct['Churned'], width, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
    
    for i, (_, row) in enumerate(seg_ct.iterrows()):
        rate = row['Churn_Rate_%']
        ax.text(i, max(row['Retained'], row['Churned']) + 150, f"Churn: {rate:.1f}%", 
                ha='center', fontweight='bold', color='#a94442', fontsize=10)
        
    ax.set_xticks(x)
    ax.set_xticklabels(seg_ct.index, fontsize=11)
    ax.set_title("Customer Churn by Customer Segment", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.legend(frameon=True)
    ax.set_ylim(0, seg_ct['Retained'].max() * 1.15)
    
    plt.tight_layout()
    p2 = os.path.join(output_dir, "02_churn_by_segment.png")
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved_files.append(p2)

    # -------------------------------------------------------------
    # 3. Churn by Region
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    reg_ct = calculate_churn_by_categorical(df, 'Region').drop('All', errors='ignore').sort_values(by='Churn_Rate_%', ascending=False)
    x = np.arange(len(reg_ct))
    
    ax.bar(x - width/2, reg_ct['Retained'], width, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
    ax.bar(x + width/2, reg_ct['Churned'], width, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
    
    for i, (_, row) in enumerate(reg_ct.iterrows()):
        ax.text(i, max(row['Retained'], row['Churned']) + 100, f"Rate: {row['Churn_Rate_%']:.1f}%", 
                ha='center', fontweight='bold', color='#a94442', fontsize=10)
        
    ax.set_xticks(x)
    ax.set_xticklabels(reg_ct.index, fontsize=11)
    ax.set_title("Customer Churn by Geographic Region (Central Highest at 21.1%)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.legend(frameon=True)
    ax.set_ylim(0, reg_ct['Retained'].max() * 1.15)
    
    plt.tight_layout()
    p3 = os.path.join(output_dir, "03_churn_by_region.png")
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved_files.append(p3)

    # -------------------------------------------------------------
    # 4. Churn by Product Category
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    cat_ct = calculate_churn_by_categorical(df, 'Category').drop('All', errors='ignore').sort_values(by='Churn_Rate_%', ascending=False)
    x = np.arange(len(cat_ct))
    
    ax.bar(x - width/2, cat_ct['Retained'], width, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
    ax.bar(x + width/2, cat_ct['Churned'], width, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
    
    for i, (_, row) in enumerate(cat_ct.iterrows()):
        ax.text(i, max(row['Retained'], row['Churned']) + 150, f"Rate: {row['Churn_Rate_%']:.1f}%", 
                ha='center', fontweight='bold', color='#a94442', fontsize=10)
        
    ax.set_xticks(x)
    ax.set_xticklabels(cat_ct.index, fontsize=11)
    ax.set_title("Customer Churn by Product Category (Furniture Highest at 15.1%)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.legend(frameon=True)
    ax.set_ylim(0, cat_ct['Retained'].max() * 1.15)
    
    plt.tight_layout()
    p4 = os.path.join(output_dir, "04_churn_by_category.png")
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    saved_files.append(p4)

    # -------------------------------------------------------------
    # 5. Sales Distribution by Churn (Log Scale & Boxplot)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.boxplot(x='Churn', y='Sales', hue='Churn', data=df, ax=axes[0], palette=CHURN_PALETTE, legend=False)
    axes[0].set_yscale('log')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
    axes[0].set_title("Sales Distribution by Churn (Log Scale)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Sales ($) [Log Scale]", fontsize=11)
    
    # KDE distribution for Sales <= 1000
    sns.kdeplot(df[df['Churn'] == 0]['Sales'], ax=axes[1], label='Retained (0)', color=CHURN_PALETTE[0], fill=True, alpha=0.3, clip=(0, 1000))
    sns.kdeplot(df[df['Churn'] == 1]['Sales'], ax=axes[1], label='Churned (1)', color=CHURN_PALETTE[1], fill=True, alpha=0.3, clip=(0, 1000))
    axes[1].set_title("Sales Density Distribution (Sales ≤ $1,000)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Sales ($)", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    p5 = os.path.join(output_dir, "05_sales_distribution_by_churn.png")
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    saved_files.append(p5)

    # -------------------------------------------------------------
    # 6. Profit Distribution by Churn
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.boxplot(x='Churn', y='Profit', hue='Churn', data=df, ax=axes[0], palette=CHURN_PALETTE, legend=False)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
    axes[0].set_title("Profit by Churn (Full Range)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Profit ($)", fontsize=11)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.7)
    
    sns.kdeplot(df[df['Churn'] == 0]['Profit'], ax=axes[1], label='Retained (0)', color=CHURN_PALETTE[0], fill=True, alpha=0.3, clip=(-250, 250))
    sns.kdeplot(df[df['Churn'] == 1]['Profit'], ax=axes[1], label='Churned (1)', color=CHURN_PALETTE[1], fill=True, alpha=0.3, clip=(-250, 250))
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.7, label='Zero Profit Line')
    axes[1].set_title("Profit Density Distribution Zoom (-$250 to +$250)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Profit ($)", fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    p6 = os.path.join(output_dir, "06_profit_distribution_by_churn.png")
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    saved_files.append(p6)

    # -------------------------------------------------------------
    # 7. Discount Distribution by Churn
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.boxplot(x='Churn', y='Discount', hue='Churn', data=df, ax=axes[0], palette=CHURN_PALETTE, legend=False)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
    axes[0].set_title("Discount Level by Churn Status", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Discount Rate", fontsize=11)
    
    sns.histplot(data=df, x='Discount', hue='Churn', palette=CHURN_PALETTE, multiple='dodge', bins=10, ax=axes[1], shrink=0.8)
    axes[1].set_title("Discount Frequency by Churn Status", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Discount Rate", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    
    plt.tight_layout()
    p7 = os.path.join(output_dir, "07_discount_distribution_by_churn.png")
    fig.savefig(p7, dpi=300)
    plt.close(fig)
    saved_files.append(p7)

    # -------------------------------------------------------------
    # 8. Profit Margin Distribution by Churn
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.boxplot(x='Churn', y='Profit Margin', hue='Churn', data=df, ax=axes[0], palette=CHURN_PALETTE, legend=False)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
    axes[0].set_title("Profit Margin (%) by Churn", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Profit Margin (%)", fontsize=11)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.6)
    
    sns.kdeplot(df[df['Churn'] == 0]['Profit Margin'], ax=axes[1], label='Retained (0)', color=CHURN_PALETTE[0], fill=True, alpha=0.3)
    sns.kdeplot(df[df['Churn'] == 1]['Profit Margin'], ax=axes[1], label='Churned (1)', color=CHURN_PALETTE[1], fill=True, alpha=0.3)
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.6, label='Break-Even Margin (0%)')
    axes[1].set_title("Profit Margin Density Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Profit Margin (%)", fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    p8 = os.path.join(output_dir, "08_profit_margin_distribution_by_churn.png")
    fig.savefig(p8, dpi=300)
    plt.close(fig)
    saved_files.append(p8)

    # -------------------------------------------------------------
    # 9. Inventory Risk vs Churn
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    inv_ct = calculate_churn_by_categorical(df, 'Inventory Risk').drop('All', errors='ignore')
    x = np.arange(len(inv_ct))
    
    ax.bar(x - width/2, inv_ct['Retained'], width, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
    ax.bar(x + width/2, inv_ct['Churned'], width, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
    
    for i, (_, row) in enumerate(inv_ct.iterrows()):
        ax.text(i, max(row['Retained'], row['Churned']) + 150, f"Rate: {row['Churn_Rate_%']:.1f}%", 
                ha='center', fontweight='bold', color='#a94442', fontsize=10)
        
    ax.set_xticks(x)
    ax.set_xticklabels(inv_ct.index, fontsize=11)
    ax.set_title("Customer Churn by Inventory Risk Level", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.legend(frameon=True)
    ax.set_ylim(0, inv_ct['Retained'].max() * 1.15)
    
    plt.tight_layout()
    p9 = os.path.join(output_dir, "09_inventory_risk_vs_churn.png")
    fig.savefig(p9, dpi=300)
    plt.close(fig)
    saved_files.append(p9)

    # -------------------------------------------------------------
    # 10. Correlation Heatmap for Numerical Variables
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Profit Margin', 'Churn']
    corr = df[num_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='vlag', vmin=-1, vmax=1, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features & Churn Target", fontsize=13, fontweight='bold', pad=14)
    
    plt.tight_layout()
    p10 = os.path.join(output_dir, "10_correlation_heatmap.png")
    fig.savefig(p10, dpi=300)
    plt.close(fig)
    saved_files.append(p10)

    # -------------------------------------------------------------
    # 11. Sub-Category Churn Rate (Analytical Value)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sub_ct = calculate_churn_by_categorical(df, 'Sub-Category').drop('All', errors='ignore').sort_values(by='Churn_Rate_%', ascending=True)
    
    colors = ['#d9534f' if rate > 15 else '#f0ad4e' if rate > 0 else '#5cb85c' for rate in sub_ct['Churn_Rate_%']]
    bars = ax.barh(sub_ct.index, sub_ct['Churn_Rate_%'], color=colors, edgecolor='black', alpha=0.85)
    
    for bar, rate in zip(bars, sub_ct['Churn_Rate_%']):
        ax.text(rate + 0.6, bar.get_y() + bar.get_height()/2, f"{rate:.1f}%", 
                va='center', fontweight='bold', fontsize=9)
        
    ax.set_title("Churn Rate by Product Sub-Category (High Risk in Binders, Tables, Machines)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Churn Rate (%)", fontsize=11)
    ax.set_xlim(0, 48)
    
    plt.tight_layout()
    p11 = os.path.join(output_dir, "11_churn_by_subcategory.png")
    fig.savefig(p11, dpi=300)
    plt.close(fig)
    saved_files.append(p11)

    # -------------------------------------------------------------
    # 12. Profit Status vs Churn (Target Leakage Flag)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    prof_ct = calculate_churn_by_categorical(df, 'Profit Status').drop('All', errors='ignore')
    x = np.arange(len(prof_ct))
    
    ax.bar(x - width/2, prof_ct['Retained'], width, label='Retained (0)', color=CHURN_PALETTE[0], alpha=0.85)
    ax.bar(x + width/2, prof_ct['Churned'], width, label='Churned (1)', color=CHURN_PALETTE[1], alpha=0.85)
    
    for i, (_, row) in enumerate(prof_ct.iterrows()):
        ax.text(i, max(row['Retained'], row['Churned']) + 200, f"Rate: {row['Churn_Rate_%']:.1f}%", 
                ha='center', fontweight='bold', color='#a94442', fontsize=11)
        
    ax.set_xticks(x)
    ax.set_xticklabels(prof_ct.index, fontsize=11)
    ax.set_title("Profit Status vs Churn (Demonstrating Quasi-Deterministic Target Linkage)", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.legend(frameon=True)
    ax.set_ylim(0, prof_ct['Retained'].max() * 1.15)
    
    plt.tight_layout()
    p12 = os.path.join(output_dir, "12_profit_status_vs_churn.png")
    fig.savefig(p12, dpi=300)
    plt.close(fig)
    saved_files.append(p12)

    print(f"[INFO] Successfully created {len(saved_files)} visualizations in '{output_dir}'.")
    return saved_files


def run_full_pipeline(filepath: str = "member1_customer_intelligence/data/raw/Cleaned_Superstore.csv",
                      figures_dir: str = "member1_customer_intelligence/reports/figures") -> None:
    """
    Main driver executing all Phase 1 checks and printing a comprehensive technical report.
    """
    print("=" * 80)
    print("ENTERPRISE AI DECISION INTELLIGENCE PLATFORM — MEMBER 1: CUSTOMER INTELLIGENCE")
    print("PHASE 1: DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    # Step 1: Load and inspect
    df = load_dataset(filepath)
    struct = inspect_dataset_structure(df)
    
    print("\n--- 1. DATASET STRUCTURE ---")
    print(f"Total Rows:               {struct['rows']:,}")
    print(f"Total Columns:            {struct['columns']}")
    print(f"Duplicate Rows:           {struct['duplicate_rows']}")
    print(f"Total Missing Values:     {struct['total_missing']}")
    print(f"Numerical Columns ({len(struct['numerical_cols'])}):  {struct['numerical_cols']}")
    print(f"Categorical Columns ({len(struct['categorical_cols'])}):{struct['categorical_cols']}")
    
    # Customer uniqueness check
    uniq = verify_customer_uniqueness(df)
    print("\n--- 2. CUSTOMER ID UNIQUENESS & AGGREGATION AUDIT ---")
    print(f"Unique Customer IDs:      {uniq['unique_customer_ids']:,} out of {uniq['total_records']:,} rows")
    print(f"Is Unique per Row:        {uniq['is_unique_per_row']}")
    print(f"Recommendation:           {uniq['aggregation_recommendation']}")
    
    # Step 2: Target distribution
    churn = analyze_churn_target(df)
    print("\n--- 3. CHURN TARGET ANALYSIS ---")
    print(f"Is Binary ({set(df['Churn'].unique())}):      {churn['is_binary']}")
    print(f"Retained (0):             {churn['retained_count']:,} ({churn['retained_pct']}%)")
    print(f"Churned (1):              {churn['churned_count']:,} ({churn['churned_pct']}%)")
    print(f"Imbalance Ratio:          {churn['imbalance_ratio']}:1 (Retained : Churned)")
    print(f"Assessment:               {churn['assessment']}")
    
    # Step 3: Categorical breakdowns
    print("\n--- 4. CHURN BY CATEGORICAL DIMENSIONS ---")
    for cat in ['Segment', 'Region', 'Category', 'Sales Category', 'Profit Status', 'Inventory Risk']:
        ct = calculate_churn_by_categorical(df, cat)
        print(f"\n[Churn by {cat}]:")
        print(ct.to_string())
        
    print("\n[Churn by Sub-Category (Sorted by Churn Rate %)]:")
    sub_ct = calculate_churn_by_categorical(df, 'Sub-Category').sort_values(by='Churn_Rate_%', ascending=False)
    print(sub_ct.to_string())
    
    # Step 3 (cont): Numerical comparisons
    print("\n--- 5. NUMERICAL DESCRIPTIVE STATS BY CHURN STATUS ---")
    num_features = ['Sales', 'Quantity', 'Discount', 'Profit', 'Profit Margin']
    for num in num_features:
        stats = calculate_numerical_summary_by_churn(df, num)
        print(f"\n[{num} by Churn]:")
        print(stats.to_string())
        
    # Correlations
    print("\n--- 6. CORRELATION WITH CHURN TARGET ---")
    corr = df[num_features + ['Churn']].corr()
    churn_corr = corr['Churn'].sort_values(ascending=False)
    print(churn_corr.to_string())
    
    # Step 4: Visualizations
    print("\n--- 7. GENERATING VISUALIZATIONS ---")
    generate_all_visualizations(df, output_dir=figures_dir)
    print("\n[DONE] Phase 1 Analysis pipeline finished successfully.")
    print("=" * 80)


if __name__ == "__main__":
    # Determine default paths based on current directory
    if os.path.exists("member1_customer_intelligence"):
        dataset_path = "member1_customer_intelligence/data/raw/Cleaned_Superstore.csv"
        figures_path = "member1_customer_intelligence/reports/figures"
    elif os.path.exists("data/raw/Cleaned_Superstore.csv"):
        dataset_path = "data/raw/Cleaned_Superstore.csv"
        figures_path = "reports/figures"
    else:
        dataset_path = "dataset/Cleaned_Superstore.csv"
        figures_path = "member1_customer_intelligence/reports/figures"
        
    run_full_pipeline(filepath=dataset_path, figures_dir=figures_path)
