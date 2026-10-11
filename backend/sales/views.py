
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from .forecasting import forecast_monthly_revenue
from .evaluation import evaluate_revenue_forecasts


def load_sales_data():
    """Load the existing Superstore dataset."""
    csv_path = Path(settings.BASE_DIR) / "data" / "Cleaned_Superstore.csv"

    if not csv_path.exists():
        return None

    return pd.read_csv(csv_path)


@require_GET
def sales_summary(request):
    """Return overall sales KPIs and sales grouped by category."""
    try:
        df = load_sales_data()

        if df is None:
            return JsonResponse(
                {"error": "Sales dataset not found."},
                status=404,
            )

        required = {"Sales", "Quantity", "Category"}
        missing = sorted(required - set(df.columns))

        if missing:
            return JsonResponse(
                {
                    "error": "Required columns are missing.",
                    "missing_columns": missing,
                },
                status=500,
            )

        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df = df.dropna(subset=["Sales", "Quantity", "Category"])

        category_sales = (
            df.groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        return JsonResponse({
            "total_records": int(len(df)),
            "total_revenue": round(float(df["Sales"].sum()), 2),
            "units_sold": int(df["Quantity"].sum()),
            "average_sale_per_record": round(float(df["Sales"].mean()), 2),
            "sales_by_category": [
                {
                    "category": str(category),
                    "revenue": round(float(revenue), 2),
                }
                for category, revenue in category_sales.items()
            ],
        })

    except (OSError, pd.errors.ParserError) as exc:
        return JsonResponse(
            {"error": f"Could not read sales dataset: {exc}"},
            status=500,
        )


@require_GET
def monthly_sales(request):
    """Return historical sales grouped by month."""
    try:
        csv_path = Path(settings.BASE_DIR) / "data" / "Sample - Superstore.csv"
        df = pd.read_csv(csv_path, encoding="cp1252") if csv_path.exists() else None

        if df is None:
            return JsonResponse(
                {"error": "Sales dataset not found."},
                status=404,
            )

        # The cleaned dataset may not contain dates.
        # Check for a date column before grouping by month.
        if "Order Date" not in df.columns:
            return JsonResponse(
                {
                    "error": "Monthly sales requires an Order Date column.",
                    "hint": "Use the original Superstore dataset with Order Date.",
                },
                status=400,
            )

        df["Order Date"] = pd.to_datetime(
            df["Order Date"], errors="coerce"
        )
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
        df = df.dropna(subset=["Order Date", "Sales"])

        monthly = (
            df.set_index("Order Date")["Sales"]
            .resample("MS")
            .sum()
            .round(2)
        )

        return JsonResponse({
            "months": len(monthly),
            "monthly_sales": [
                {
                    "month": month.strftime("%Y-%m"),
                    "revenue": float(revenue),
                }
                for month, revenue in monthly.items()
            ],
        })

    except (OSError, pd.errors.ParserError) as exc:
        return JsonResponse(
            {"error": f"Could not read sales dataset: {exc}"},
            status=500,
        )


@require_GET
def revenue_forecast(request):
    """Return forecasted monthly revenue."""
    try:
        months = int(request.GET.get("months", 3))
        forecasts = forecast_monthly_revenue(months)

        return JsonResponse({
            "forecast_horizon_months": months,
            "historical_data_ends": "2017-12",
            "forecast": forecasts,
            "note": (
                "Demonstration forecast based on historical data "
                "ending December 2017; not a current business prediction."
            ),
        })

    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except (OSError, pd.errors.ParserError) as exc:
        return JsonResponse(
            {"error": f"Could not read sales data: {exc}"},
            status=500,
        )


@require_GET
def revenue_evaluation(request):
    """Compare forecast accuracy against a naive baseline."""
    try:
        test_months = int(request.GET.get("test_months", 6))
        results = evaluate_revenue_forecasts(test_months)

        return JsonResponse(results)

    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except (OSError, pd.errors.ParserError) as exc:
        return JsonResponse(
            {"error": f"Could not evaluate sales data: {exc}"},
            status=500,
        )
