from django.urls import path

from . import views

urlpatterns = [
    path("summary/", views.customer_summary, name="customer_summary"),
]
