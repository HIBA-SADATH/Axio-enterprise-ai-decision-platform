
import pandas as pd

# 1. Load the original dataset
df = pd.read_csv("data/Cleaned_Superstore.csv")

print("BUSINESS RISK ANALYSIS")
print("=" * 50)

# 2. Analyse sales and profit by category
print("\n1. SALES AND PROFIT BY CATEGORY")

category_analysis = (
    df.groupby("Category")
    .agg(
        Records=("Sales", "size"),
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Average_Discount=("Discount", "mean"),
        Loss_Records=("Profit", lambda x: (x < 0).sum()),
    )
    .round(2)
    .sort_values("Total_Profit")
)

print(category_analysis.to_string())

# 3. Analyse sales and profit by region
print("\n2. SALES AND PROFIT BY REGION")

region_analysis = (
    df.groupby("Region")
    .agg(
        Records=("Sales", "size"),
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Loss_Records=("Profit", lambda x: (x < 0).sum()),
    )
    .round(2)
    .sort_values("Total_Profit")
)

print(region_analysis.to_string())

# 4. Find product sub-categories with the lowest profit
print("\n3. FIVE LOWEST-PROFIT SUB-CATEGORIES")

subcategory_analysis = (
    df.groupby("Sub-Category")
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Loss_Records=("Profit", lambda x: (x < 0).sum()),
    )
    .round(2)
    .sort_values("Total_Profit")
)

print(subcategory_analysis.head(5).to_string())

# 5. Identify records with negative profit
print("\n4. LOSS ANALYSIS")

loss_records = df[df["Profit"] < 0]

print("Total records:", len(df))
print("Records with negative profit:", len(loss_records))
print(
    "Percentage with negative profit:",
    round(len(loss_records) / len(df) * 100, 2),
    "%"
)

# 6. Save reports separately
category_analysis.to_csv("data/category_risk_analysis.csv")
region_analysis.to_csv("data/region_risk_analysis.csv")
subcategory_analysis.to_csv("data/subcategory_risk_analysis.csv")

print("\nAnalysis completed successfully.")
print("Reports saved in the data folder.")
