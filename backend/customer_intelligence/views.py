"""Read-only customer analytics generated from the checked-in dataset."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET


DATASET_PATH = (
    Path(settings.BASE_DIR)
    / "customer_intelligence"
    / "data"
    / "raw"
    / "Cleaned_Superstore.csv"
)
REQUIRED_COLUMNS = {
    "Customer ID",
    "Segment",
    "Region",
    "Category",
    "Sub-Category",
    "Sales",
    "Churn",
}


def _round(value):
    """Convert pandas/numpy numeric values into JSON-safe currency/rates."""
    return round(float(value), 2)


def load_customer_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Customer dataset not found.")

    df = pd.read_csv(DATASET_PATH)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Customer dataset is missing required columns: {', '.join(missing)}")

    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
    return df.dropna(subset=list(REQUIRED_COLUMNS)).copy()


def grouped_metrics(df, column):
    grouped = (
        df.groupby(column, dropna=False)
        .agg(
            customers=("Customer ID", "count"),
            sales=("Sales", "sum"),
            churned_customers=("Churn", "sum"),
            churn_rate=("Churn", "mean"),
        )
        .sort_values("customers", ascending=False)
    )
    return [
        {
            "name": str(name),
            "customers": int(row.customers),
            "sales": _round(row.sales),
            "churned_customers": int(row.churned_customers),
            "churn_rate": _round(row.churn_rate * 100),
        }
        for name, row in grouped.iterrows()
    ]


@require_GET
def customer_summary(request):
    """Return audited descriptive metrics; it does not make live predictions."""
    try:
        df = load_customer_data()
        total_customers = len(df)
        churned_customers = int(df["Churn"].sum())

        subcategory_metrics = grouped_metrics(df, "Sub-Category")
        highest_risk_subcategories = sorted(
            subcategory_metrics,
            key=lambda item: item["churn_rate"],
            reverse=True,
        )[:5]

        category_sales = (
            df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        )

        return JsonResponse(
            {
                "total_customers": total_customers,
                "total_sales": _round(df["Sales"].sum()),
                "average_sales_per_customer": _round(df["Sales"].mean()),
                "churned_customers": churned_customers,
                "churn_rate": _round((churned_customers / total_customers) * 100),
                "segment_breakdown": grouped_metrics(df, "Segment"),
                "region_breakdown": grouped_metrics(df, "Region"),
                "sales_by_category": [
                    {"name": str(category), "sales": _round(sales)}
                    for category, sales in category_sales.items()
                ],
                "highest_risk_subcategories": highest_risk_subcategories,
                "data_note": (
                    "Churn values are labels contained in the historical dataset. "
                    "This endpoint reports descriptive analysis and does not serve a live churn prediction model."
                ),
            }
        )
    except FileNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        return JsonResponse({"error": str(exc)}, status=500)
