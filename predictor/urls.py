from django.urls import path
from predictor.views import home, sources

urlpatterns = [
    path("", home, name="home"),
    path("sources/", sources, name="sources"),
]
