from django.urls import path
from predictor.views import home, about, stats, api, archive

urlpatterns = [
    path("", home, name="home"),
    path("about/", about, name="about"),
    path("stats/", stats, name="stats"),
    path("api/", api, name="api"),
    path("archive/", archive, name="archive"),
]
