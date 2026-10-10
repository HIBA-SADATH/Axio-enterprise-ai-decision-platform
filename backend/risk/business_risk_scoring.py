
import pandas as pd

# Load the existing sub-category analysis report
df = pd.read_csv("data/subcategory_risk_analysis.csv")

# Calculate profit margin for each sub-category
df["Profit_Margin_Percent"] = (
    df["Total_Profit"] / df["Total_Sales"] * 100
).round(2)

# Assign a transparent business risk level
def assign_business_risk(row):
    if row["Total_Profit"] < 0:
        return "High"
    elif row["Profit_Margin_Percent"] < 5:
        return "Medium"
    else:
        return "Low"

df["Business_Risk_Level"] = df.apply(
    assign_business_risk,
    axis=1
)

# Explain the assigned risk level
def explain_business_risk(row):
    if row["Total_Profit"] < 0:
        return "Negative total profit"
    elif row["Profit_Margin_Percent"] < 5:
        return "Low profit margin below 5%"
    else:
        return "Positive profit margin of at least 5%"

df["Business_Risk_Explanation"] = df.apply(
    explain_business_risk,
    axis=1
)

# Show the results
df = df.sort_values(
    by=["Business_Risk_Level", "Total_Profit"]
)

print("BUSINESS RISK SCORES BY SUB-CATEGORY")
print("=" * 65)
print(df.to_string(index=False))

# Save to a new file
df.to_csv("data/business_risk_scores.csv", index=False)

print("\nBusiness risk scores saved successfully.")
