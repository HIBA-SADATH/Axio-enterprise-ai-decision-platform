
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("data/Cleaned_Superstore.csv")

# 2. Analyze customer-related records
print("CUSTOMER BEHAVIOR ANALYSIS")
print("=" * 40)

print("\nTotal transaction records:", len(df))
print("Unique customers:", df["Customer ID"].nunique())

# Check whether customers have multiple records
records_per_customer = df.groupby("Customer ID").size()

print(
    "Customers with multiple records:",
    (records_per_customer > 1).sum()
)

# 3. Summarize churn labels
print("\nChurn Summary:")
print(df["Churn"].value_counts().sort_index())

# 4. Compare churn labels with profit status
print("\nChurn vs Profit Status:")
print(pd.crosstab(df["Churn"], df["Profit Status"]))

# 5. Save the analysis
summary = pd.crosstab(df["Churn"], df["Profit Status"])
summary.to_csv("data/customer_behavior_analysis.csv")

print("\nAnalysis saved to data/customer_behavior_analysis.csv")
