
import pandas as pd

# Load the anomaly detection results
df = pd.read_csv("data/anomaly_results.csv")

# Assign a risk level to each record
def assign_risk(row):
    if row["Anomaly"] == "Normal":
        return "Low"

    elif row["Profit"] < 0 and row["Discount"] >= 0.50:
        return "Critical"

    elif row["Profit"] < 0:
        return "High"

    else:
        return "Medium"


df["Risk Level"] = df.apply(assign_risk, axis=1)

# Display risk counts
print("Risk level counts:")
print(df["Risk Level"].value_counts())

# Display some high-risk records
print("\nHigh and Critical risk records:")
print(
    df[df["Risk Level"].isin(["High", "Critical"])][
        ["Sales", "Discount", "Profit", "Anomaly", "Risk Level"]
    ].head(10)
)


# Explain why each record received its risk level
def explain_risk(row):
    if row["Anomaly"] == "Normal":
        return "No unusual pattern detected by the model"

    reasons = []

    if row["Profit"] < 0:
        reasons.append("Negative profit")

    if row["Discount"] >= 0.50:
        reasons.append("High discount (50% or more)")

    if not reasons:
        reasons.append("Unusual sales or transaction pattern")

    return "; ".join(reasons)


df["Risk Explanation"] = df.apply(explain_risk, axis=1)

print("\nRisk explanations:")
print(
    df[df["Risk Level"].isin(["High", "Critical"])][
        ["Sales", "Discount", "Profit", "Risk Level", "Risk Explanation"]
    ].head(10)
)
# Save the updated results
df.to_csv("data/risk_results.csv", index=False)

print("\nResults saved to data/risk_results.csv")

