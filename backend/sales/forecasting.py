
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def forecast_monthly_revenue(months=3):
    """Forecast monthly revenue using historical Superstore sales."""
    if not 1 <= months <= 12:
        raise ValueError("months must be between 1 and 12")

    from pathlib import Path
    from django.conf import settings

    csv_path = (
        Path(settings.BASE_DIR)
        / "data"
        / "Sample - Superstore.csv"
    )

    df = pd.read_csv(csv_path, encoding="cp1252")
    df["Order Date"] = pd.to_datetime(
        df["Order Date"], errors="coerce"
    )
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    monthly = (
        df.set_index("Order Date")["Sales"]
        .resample("MS")
        .sum()
        .asfreq("MS", fill_value=0)
    )

    if len(monthly) < 6:
        raise ValueError("Not enough monthly data to forecast.")

    model = ExponentialSmoothing(
        monthly,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    ).fit()

    predictions = model.forecast(months)

    return [
        {
            "month": date.strftime("%Y-%m"),
            "forecast_revenue": round(float(value), 2),
        }
        for date, value in predictions.items()
    ]
