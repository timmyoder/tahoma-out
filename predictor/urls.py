from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns


from predictor.views import (home, about, stats, APIViewSetAll,
                             APIViewSetAllOut, archive, last_prediction, api)


# API endpoints
api_patterns = format_suffix_patterns([
    path('api', last_prediction, name='api-last'),
    path('api/', last_prediction, name='api-last'),
    path('api/all/', APIViewSetAll.as_view({'get': 'list'}), name='api-all'),
    path('api/all-out/', APIViewSetAllOut.as_view({'get': 'list'}), name='api-all-out')
])

urlpatterns = [
    path("", home, name="home"),
    path("about/", about, name="about"),
    path("stats/", stats, name="stats"),
    path("archive/", archive, name="archive"),
    path('api-root/', api, name='api-explain'),
] + api_patterns

