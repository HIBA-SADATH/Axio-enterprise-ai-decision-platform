
from django.urls import path
from . import views

urlpatterns = [
    path("summary/", views.sales_summary, name="sales_summary"),
    path("monthly/", views.monthly_sales, name="monthly_sales"),
    path("forecast/", views.revenue_forecast, name="revenue_forecast"),
    path("evaluation/", views.revenue_evaluation, name="revenue_evaluation"),
]
