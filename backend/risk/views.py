from django.shortcuts import render

# Create your views here.

import math
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET

def load_risk_data():
    """Load risk results and attach SHAP contributions by original row index."""
    csv_path = Path(settings.BASE_DIR) / "data" / "risk_results.csv"

    if not csv_path.exists():
        return None, JsonResponse(
            {"error": "Risk results file not found"},
            status=404
        )

    df = pd.read_csv(csv_path)

    # Default explanation if SHAP results are unavailable.
    df["SHAP Explanation"] = "SHAP explanation not available"

    shap_path = Path(settings.BASE_DIR) / "data" / "shap_explanations.csv"

    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)

        if "Original Row Index" in shap_df.columns:
            shap_df = shap_df.set_index("Original Row Index")

            shap_columns = [
                col for col in shap_df.columns
                if col.startswith("SHAP_")
            ]

            # The risk CSV preserves the original dataset row order.
            # Use its row positions to match the SHAP indexes.
            df["Original Row Index"] = range(len(df))
            df = df.join(
                shap_df[shap_columns],
                on="Original Row Index"
            )

            # Build a readable explanation from the largest
            # absolute SHAP contribution for each explained record.
            for idx, row in df.iterrows():
                available = row[shap_columns].dropna()

                if not available.empty:
                    strongest = available.abs().idxmax()
                    feature_name = strongest.replace("SHAP_", "")

                    df.at[idx, "SHAP Explanation"] = (
                        f"Strongest contributing feature: {feature_name}"
                    )

    return df, None



def risk_records(request):
    df, error = load_risk_data()

    if error:
        return error

    required_columns = ["Risk Level", "Anomaly", "Risk Explanation"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return JsonResponse(
            {
                "error": "Required columns are missing from the CSV",
                "missing_columns": missing_columns
            },
            status=500
        )

    level = request.GET.get("level")

    if level:
        valid_levels = ["Low", "Medium", "High", "Critical"]

        if level not in valid_levels:
            return JsonResponse(
                {
                    "error": "Invalid risk level",
                    "valid_levels": valid_levels
                },
                status=400
            )

        df = df[df["Risk Level"] == level]

    try:
        page = int(request.GET.get("page", 1))
        page_size = int(request.GET.get("page_size", 20))
    except ValueError:
        return JsonResponse(
            {"error": "page and page_size must be integers"},
            status=400
        )

    if page < 1 or page_size < 1 or page_size > 100:
        return JsonResponse(
            {
                "error": (
                    "page must be at least 1 and page_size "
                    "must be between 1 and 100"
                )
            },
            status=400
        )

    total_records = len(df)
    start = (page - 1) * page_size
    records = df.iloc[start:start + page_size].to_dict(orient="records")

    return JsonResponse({
        "total_records": total_records,
        "page": page,
        "page_size": page_size,
        "returned_records": len(records),
        "results": records
    })


def risk_summary(request):
    df, error = load_risk_data()

    if error:
        return error

    required_columns = ["Risk Level", "Anomaly"]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        return JsonResponse(
            {"error": "Required columns are missing", "missing_columns": missing},
            status=500
        )

    risk_counts = (
        df["Risk Level"]
        .value_counts()
        .reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
    )

    return JsonResponse({
        "total_records": len(df),
        "normal_records": int((df["Anomaly"] == "Normal").sum()),
        "anomaly_records": int((df["Anomaly"] == "Anomaly").sum()),
        "risk_counts": {
            level: int(count) for level, count in risk_counts.items()
        }
    })


def risk_alerts(request):
    df, error = load_risk_data()

    if error:
        return error

    if "Risk Level" not in df.columns:
        return JsonResponse(
            {
                "error": "Required columns are missing",
                "missing_columns": ["Risk Level"]
            },
            status=500
        )

    alerts = df[df["Risk Level"].isin(["High", "Critical"])].copy()
    level = request.GET.get("level")

    if level:
        valid_levels = ["High", "Critical"]

        if level not in valid_levels:
            return JsonResponse(
                {
                    "error": "Invalid risk level",
                    "valid_levels": valid_levels
                },
                status=400
            )

        alerts = alerts[alerts["Risk Level"] == level]

    try:
        page = int(request.GET.get("page", 1))
        page_size = int(request.GET.get("page_size", 20))
    except ValueError:
        return JsonResponse(
            {"error": "page and page_size must be integers"},
            status=400
        )

    if page < 1 or page_size < 1 or page_size > 100:
        return JsonResponse(
            {
                "error": (
                    "page must be at least 1 and page_size "
                    "must be between 1 and 100"
                )
            },
            status=400
        )

    total_alerts = len(alerts)
    total_pages = math.ceil(total_alerts / page_size)
    start = (page - 1) * page_size
    records = alerts.iloc[start:start + page_size].to_dict(orient="records")

    return JsonResponse({
        "total_alerts": total_alerts,
        "total_pages": total_pages,
        "page": page,
        "page_size": page_size,
        "returned_alerts": len(records),
        "has_next": page < total_pages,
        "has_previous": page > 1,
        "results": records
    })

@require_GET
def business_risk_scores(request):
    """Return business risk scores grouped by sub-category."""
    import os
    import pandas as pd
    from django.conf import settings
    from django.http import JsonResponse

    file_path = os.path.join(
        settings.BASE_DIR,
        "data",
        "business_risk_scores.csv"
    )

    if not os.path.exists(file_path):
        return JsonResponse(
            {"error": "Business risk scores file not found."},
            status=404
        )

    df = pd.read_csv(file_path)
    records = df.replace({float("nan"): None}).to_dict(orient="records")

    return JsonResponse({
        "total_subcategories": len(records),
        "business_risk_scores": records
    })
