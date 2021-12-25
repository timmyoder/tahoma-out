from django.shortcuts import render
from predictor.prediction import Predictor


def home(request):

    predictor = Predictor()

    return render(request, 'home.jinja2', {"predictor": predictor})


def sources(request):
    return render(request, 'sources.jinja2')
