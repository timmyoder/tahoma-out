from django.shortcuts import render

from predictor.models import Photo


def home(request):
    latest_prediction = Photo.objects.latest('id')

    predictions = [latest_prediction.pred_all,
                   latest_prediction.pred_base,
                   latest_prediction.pred_none,
                   latest_prediction.pred_tip]
    formatted_prediction = [f'{p*100:.2f}%' for p in predictions]
    return render(request, 'home.jinja2', {'photo_model': latest_prediction,
                                           'formatted_predictions': formatted_prediction})


def about(request):
    return render(request, 'about.jinja2')


def stats(request):
    return render(request, 'stats.jinja2')


def archive(request):
    return render(request, 'archive.jinja2')


def api(request):
    return render(request, 'api.jinja2')
