from django.urls import path
from assessment import views

urlpatterns = [
    path("", views.assess, name="assess"),
    path("about/", views.about, name="about"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
