import os
import json
import pandas as pd
import ollama

from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from .rag_assistant import build_index, answer_question
def home(request):
    return JsonResponse({
        "project": "Enterprise AI Decision Intelligence Platform",
        "module": "Member 4 - AI Decision Intelligence",
        "status": "Backend running successfully",
        "features": [
            "Business Analytics",
            "AI Executive Copilot",
            "RAG Document Assistant",
            "What-If Simulator",
            "Decision Recommendation Engine"
        ]
    })


def business_analytics(request):
    file_path = os.path.join(
        settings.BASE_DIR,
        "copilot",
        "data",
        "Cleaned_Superstore.csv"
    )

    if not os.path.exists(file_path):
        return JsonResponse({
            "error": "Dataset not found. Check the CSV filename in copilot/data/"
        }, status=404)

    try:
        df = pd.read_csv(file_path)

        required_columns = [
            "Sales", "Profit", "Quantity",
            "Discount", "Category", "Sub-Category"
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return JsonResponse({
                "error": "Missing required columns",
                "columns": missing
            }, status=400)

        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
        df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
        df = df.dropna(subset=required_columns)

        category_summary = (
            df.groupby("Category")
            .agg(
                sales=("Sales", "sum"),
                profit=("Profit", "sum"),
                quantity=("Quantity", "sum")
            )
            .reset_index()
        )

        subcategory_summary = (
            df.groupby("Sub-Category")
            .agg(
                sales=("Sales", "sum"),
                profit=("Profit", "sum")
            )
            .reset_index()
            .sort_values("sales", ascending=False)
            .head(5)
        )

        return JsonResponse({
            "total_records": int(len(df)),
            "total_sales": round(float(df["Sales"].sum()), 2),
            "total_profit": round(float(df["Profit"].sum()), 2),
            "total_quantity": int(df["Quantity"].sum()),
            "average_discount": round(float(df["Discount"].mean()), 4),
            "category_performance": category_summary.to_dict("records"),
            "top_subcategories_by_sales": subcategory_summary.to_dict("records")
        })

    except Exception as e:
        return JsonResponse({
            "error": str(e)
        }, status=500)
@csrf_exempt

def ai_copilot(request):
    if request.method != "POST":
        return JsonResponse({
            "error": "Use POST with a question field."
        }, status=405)

    import json

    try:
        body = json.loads(request.body)
        question = body.get("question", "").strip()

        if not question:
            return JsonResponse({
                "error": "Please enter a business question."
            }, status=400)

        # Get real analytics from the existing API function
        analytics_response = business_analytics(request)
        analytics = json.loads(analytics_response.content)

        if "error" in analytics:
            return JsonResponse(analytics, status=500)

        prompt = f"""
You are an AI Executive Copilot for a business intelligence platform.
Answer the user's question using the business data below.
Do not invent facts or claim the data contains dates, customer IDs,
marketing budgets, or other unavailable fields.
Clearly label recommendations as suggestions.

BUSINESS ANALYTICS:
{json.dumps(analytics, indent=2)}

USER QUESTION:
{question}

Give a clear, concise, business-focused answer.
"""

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful business analytics assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return JsonResponse({
            "question": question,
            "answer": response["message"]["content"],
            "data_used": "Sample Superstore business analytics"
        })

    except json.JSONDecodeError:
        return JsonResponse({
            "error": "Invalid JSON request body."
        }, status=400)

    except Exception as e:
        return JsonResponse({
            "error": str(e)
        }, status=500)
@csrf_exempt
def what_if_simulator(request):
    if request.method != "POST":
        return JsonResponse({
            "error": "Use POST with discount_change_percent."
        }, status=405)

    try:
        body = json.loads(request.body)
        change_percent = float(
            body.get("discount_change_percent", 0)
        )

        if not -100 <= change_percent <= 100:
            return JsonResponse({
                "error": "Discount change must be between -100 and 100 percent."
            }, status=400)

        file_path = os.path.join(
            settings.BASE_DIR,
            "copilot",
            "data",
            "Cleaned_Superstore.csv"
        )

        df = pd.read_csv(file_path)

        for col in ["Sales", "Profit", "Discount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Sales", "Profit", "Discount"])

        # Only use rows with valid discount rates.
        df = df[
            (df["Discount"] >= 0) &
            (df["Discount"] < 1)
        ].copy()

        if df.empty:
            return JsonResponse({
                "error": "No usable records with valid discount rates."
            }, status=400)

        # Baseline historical totals.
        baseline_sales = float(df["Sales"].sum())
        baseline_profit = float(df["Profit"].sum())

        # Estimate sales value before discount and hold it constant.
        pre_discount_sales = (
            df["Sales"] / (1 - df["Discount"])
        )

        # Estimate costs from historical sales and profit.
        estimated_costs = df["Sales"] - df["Profit"]

        # Apply a relative change to each existing discount.
        # Example: 20% discount with a 10% reduction becomes 18%.
        new_discount = (
            df["Discount"] * (1 + change_percent / 100)
        ).clip(lower=0, upper=0.95)

        scenario_sales = float(
            (pre_discount_sales * (1 - new_discount)).sum()
        )

        scenario_profit = float(
            scenario_sales - estimated_costs.sum()
        )

        return JsonResponse({
            "discount_change_percent": change_percent,
            "average_baseline_discount": round(
                float(df["Discount"].mean()), 4
            ),
            "average_scenario_discount": round(
                float(new_discount.mean()), 4
            ),
            "baseline_sales": round(baseline_sales, 2),
            "scenario_sales_estimate": round(scenario_sales, 2),
            "baseline_profit": round(baseline_profit, 2),
            "scenario_profit_estimate": round(scenario_profit, 2),
            "estimated_profit_change": round(
                scenario_profit - baseline_profit, 2
            ),
            "assumption": (
                "Accounting-based scenario estimate: reported sales are "
                "assumed to reflect existing discounts. Pre-discount "
                "sales, quantities, and estimated costs are held constant. "
                "This does not predict customer demand or prove the causal "
                "effect of changing discounts."
            )
        })

    except json.JSONDecodeError:
        return JsonResponse({
            "error": "Invalid JSON request body."
        }, status=400)

    except (ValueError, TypeError):
        return JsonResponse({
            "error": "Provide a valid numeric discount_change_percent."
        }, status=400)

    except FileNotFoundError:
        return JsonResponse({
            "error": "Cleaned_Superstore.csv was not found."
        }, status=404)

    except Exception as error:
        return JsonResponse({
            "error": str(error)
        }, status=500)

@csrf_exempt
def rag_build(request):
    if request.method != "POST":
        return JsonResponse(
            {"error": "Use POST to build the document index."},
            status=405
        )

    try:
        chunk_count = build_index()

        if chunk_count == 0:
            return JsonResponse({
                "message": "No PDF or TXT documents found.",
                "chunks_indexed": 0
            })

        return JsonResponse({
            "message": "Document index built successfully.",
            "chunks_indexed": chunk_count
        })

    except Exception as error:
        return JsonResponse({"error": str(error)}, status=500)


@csrf_exempt
def rag_ask(request):
    if request.method != "POST":
        return JsonResponse(
            {"error": "Use POST to ask a document question."},
            status=405
        )

    try:
        data = json.loads(request.body)
        question = data.get("question", "").strip()

        if not question:
            return JsonResponse(
                {"error": "Please provide a question."},
                status=400
            )

        return JsonResponse(answer_question(question))

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    except Exception as error:
        return JsonResponse({"error": str(error)}, status=500)

def decision_recommendations(request):
    """Generate business recommendations from available sales data."""

    if request.method != "GET":
        return JsonResponse(
            {"error": "Use GET to retrieve recommendations."},
            status=405
        )

    file_path = os.path.join(
        settings.BASE_DIR,
        "copilot",
        "data",
        "Cleaned_Superstore.csv"
    )

    try:
        df = pd.read_csv(file_path)

        required_columns = [
            "Sales", "Profit", "Discount",
            "Category", "Sub-Category"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            return JsonResponse({
                "error": "Missing required columns",
                "columns": missing
            }, status=400)

        for col in ["Sales", "Profit", "Discount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required_columns)

        if df.empty:
            return JsonResponse({
                "error": "No usable business records found."
            }, status=400)

        # Analyze category profitability
        category_summary = (
            df.groupby("Category")
            .agg(
                sales=("Sales", "sum"),
                profit=("Profit", "sum")
            )
            .reset_index()
            .sort_values("profit", ascending=False)
        )

        # Analyze individual subcategories
        subcategory_summary = (
            df.groupby("Sub-Category")
            .agg(
                sales=("Sales", "sum"),
                profit=("Profit", "sum"),
                average_discount=("Discount", "mean")
            )
            .reset_index()
        )

        subcategory_summary["profit_margin_percent"] = (
            subcategory_summary["profit"]
            / subcategory_summary["sales"].replace(0, float("nan"))
            * 100
        )

        subcategory_summary["profit_margin_percent"] = (
            subcategory_summary["profit_margin_percent"].fillna(0)
        )

        recommendations = []

        # Recommendation 1: Focus on the most profitable category
        best_category = category_summary.iloc[0]

        recommendations.append({
            "type": "Growth Opportunity",
            "focus": str(best_category["Category"]),
            "observation": (
                f"This category generated the highest total profit: "
                f"{best_category['profit']:.2f}."
            ),
            "recommendation": (
                "Evaluate opportunities to promote its profitable products "
                "and maintain their performance."
            )
        })

        # Recommendation 2: Investigate loss-making subcategories
        loss_products = (
            subcategory_summary[
                subcategory_summary["profit"] < 0
            ]
            .sort_values("profit")
            .head(3)
        )

        for _, row in loss_products.iterrows():
            recommendations.append({
                "type": "Profitability Risk",
                "focus": str(row["Sub-Category"]),
                "profit": round(float(row["profit"]), 2),
                "average_discount": round(
                    float(row["average_discount"]), 4
                ),
                "recommendation": (
                    "Review pricing, product costs, and discount practices "
                    "before deciding on corrective action."
                )
            })

        # Recommendation 3: Identify high-sales, low-margin products
        high_sales_low_margin = (
            subcategory_summary[
                (subcategory_summary["sales"] > 0)
                & (subcategory_summary["profit_margin_percent"] < 5)
            ]
            .sort_values("sales", ascending=False)
            .head(3)
        )

        for _, row in high_sales_low_margin.iterrows():
            recommendations.append({
                "type": "Margin Improvement",
                "focus": str(row["Sub-Category"]),
                "sales": round(float(row["sales"]), 2),
                "profit_margin_percent": round(
                    float(row["profit_margin_percent"]), 2
                ),
                "recommendation": (
                    "Investigate costs and pricing because strong sales "
                    "do not necessarily produce strong profit margins."
                )
            })

        return JsonResponse({
            "engine": "Rule-Based Decision Recommendation Engine",
            "highest_profit_category": {
                "category": str(best_category["Category"]),
                "profit": round(float(best_category["profit"]), 2),
                "sales": round(float(best_category["sales"]), 2)
            },
            "recommendations": recommendations,
            "method": (
                "Recommendations are generated from historical sales, "
                "profit, discount, category, and subcategory metrics."
            ),
            "limitation": (
                "These are descriptive recommendations, not causal "
                "predictions. The dataset does not provide order dates "
                "or customer IDs."
            )
        })

    except FileNotFoundError:
        return JsonResponse({
            "error": "Cleaned_Superstore.csv was not found."
        }, status=404)

    except Exception as error:
        return JsonResponse({
            "error": str(error)
        }, status=500)

def discount_scenario_optimizer(request):
    """Rank subcategories using simplified discount scenarios."""

    if request.method != "GET":
        return JsonResponse({
            "error": "Use GET to retrieve discount optimization results."
        }, status=405)

    file_path = os.path.join(
        settings.BASE_DIR,
        "copilot",
        "data",
        "Cleaned_Superstore.csv"
    )

    try:
        df = pd.read_csv(file_path)

        required_columns = [
            "Sales", "Profit", "Discount", "Sub-Category"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            return JsonResponse({
                "error": "Missing required columns",
                "columns": missing
            }, status=400)

        for col in ["Sales", "Profit", "Discount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required_columns)

        # Use records with valid discount rates.
        df = df[
            (df["Discount"] >= 0) &
            (df["Discount"] < 1)
        ].copy()

        if df.empty:
            return JsonResponse({
                "error": "No usable records with valid discount rates."
            }, status=400)

        # Approximate pre-discount sales and costs from historical data.
        df["pre_discount_sales"] = (
            df["Sales"] / (1 - df["Discount"])
        )
        df["estimated_cost"] = df["Sales"] - df["Profit"]

        # These are relative changes to existing discounts, not
        # percentage-point changes.
        candidate_changes = [-20, -10, 0, 10, 20]
        ranked_results = []

        for subcategory, group in df.groupby("Sub-Category"):
            baseline_sales = float(group["Sales"].sum())
            baseline_profit = float(group["Profit"].sum())

            pre_discount_sales = group["pre_discount_sales"]
            estimated_costs = group["estimated_cost"]

            scenarios = []

            for change in candidate_changes:
                scenario_discount = (
                    group["Discount"] * (1 + change / 100)
                ).clip(lower=0, upper=0.95)

                scenario_sales = float(
                    (
                        pre_discount_sales *
                        (1 - scenario_discount)
                    ).sum()
                )

                scenario_profit = float(
                    scenario_sales - estimated_costs.sum()
                )

                scenarios.append({
                    "relative_discount_change_percent": change,
                    "scenario_sales_estimate": round(
                        scenario_sales, 2
                    ),
                    "scenario_profit_estimate": round(
                        scenario_profit, 2
                    ),
                    "estimated_profit_improvement": round(
                        scenario_profit - baseline_profit, 2
                    )
                })

            # Select the scenario with the highest estimated profit.
            best_scenario = max(
                scenarios,
                key=lambda item: item["scenario_profit_estimate"]
            )

            ranked_results.append({
                "subcategory": str(subcategory),
                "baseline_sales": round(baseline_sales, 2),
                "baseline_profit": round(baseline_profit, 2),
                **best_scenario
            })

        ranked_results.sort(
            key=lambda item: item["estimated_profit_improvement"],
            reverse=True
        )

        return JsonResponse({
            "engine": "Rule-Based Discount Scenario Optimizer",
            "scenarios_tested": candidate_changes,
            "ranking": ranked_results,
            "assumptions": [
                "Discount changes are relative percentages of existing discounts.",
                "Historical pre-discount sales and estimated costs are held constant.",
                "Customer demand and sales volume do not respond to discount changes.",
                "Results are scenario estimates, not causal predictions or ML forecasts."
            ]
        })

    except FileNotFoundError:
        return JsonResponse({
            "error": "Cleaned_Superstore.csv was not found."
        }, status=404)

    except Exception as error:
        return JsonResponse({
            "error": str(error)
        }, status=500)
