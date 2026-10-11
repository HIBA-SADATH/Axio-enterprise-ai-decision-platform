
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("analytics/", views.business_analytics, name="business_analytics"),
    path("ai/", views.ai_copilot, name="ai_copilot"),
    path("what-if/", views.what_if_simulator, name="what_if"),
    path("rag/build/", views.rag_build, name="rag_build"),
    path("rag/ask/", views.rag_ask, name="rag_ask"),
    path("recommendations/",views.decision_recommendations,name="decision_recommendations"),
    path("optimizer/",views.discount_scenario_optimizer,name="discount_scenario_optimizer"
),
  
]
