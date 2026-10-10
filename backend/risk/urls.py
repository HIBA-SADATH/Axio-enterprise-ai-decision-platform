
from django.urls import path
from . import views

urlpatterns = [
    path("records/", views.risk_records, name="risk_records"),
    path("summary/", views.risk_summary, name="risk_summary"),
    path("alerts/", views.risk_alerts, name="risk_alerts"),
    path("business-scores/", views.business_risk_scores, name="business_risk_scores"),
]

