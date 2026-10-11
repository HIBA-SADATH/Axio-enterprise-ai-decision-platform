
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def evaluate_revenue_forecasts(test_months=6):
    """Compare Exponential Smoothing with a naive baseline."""
    if not 1 <= test_months <= 12:
        raise ValueError("test_months must be between 1 and 12")

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

    if len(monthly) <= test_months:
        raise ValueError("Not enough history for this test period.")

    train = monthly.iloc[:-test_months]
    actual = monthly.iloc[-test_months:]

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    ).fit()

    exponential_predictions = model.forecast(test_months)
    naive_predictions = pd.Series(
        train.iloc[-1],
        index=actual.index,
    )

    def metrics(predictions):
        errors = actual.to_numpy() - predictions.to_numpy()
        return {
            "mae": round(float(np.mean(np.abs(errors))), 2),
            "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 2),
        }

    return {
        "test_months": test_months,
        "training_months": len(train),
        "actual_period_start": actual.index[0].strftime("%Y-%m"),
        "actual_period_end": actual.index[-1].strftime("%Y-%m"),
        "models": {
            "exponential_smoothing": metrics(exponential_predictions),
            "naive_baseline": metrics(naive_predictions),
        },
        "monthly_results": [
            {
                "month": date.strftime("%Y-%m"),
                "actual_revenue": round(float(actual.loc[date]), 2),
                "exponential_smoothing": round(
                    float(exponential_predictions.loc[date]), 2
                ),
                "naive_baseline": round(
                    float(naive_predictions.loc[date]), 2
                ),
            }
            for date in actual.index
        ],
        "note": (
            "Evaluation uses a historical holdout period, not current sales."
        ),
    }
