from django.urls import path
from predictor.views import home, sources, stats, api, archive

urlpatterns = [
    path("", home, name="home"),
    path("sources/", sources, name="sources"),
    path("stats/", stats, name="stats"),
    path("api/", api, name="api"),
    path("archive/", archive, name="archive"),
]
