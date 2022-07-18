import datetime as dt

from django.shortcuts import render
from django.utils.timezone import make_aware

from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view

from predictor.models import Photo, Plot
from predictor.forms import ExampleForm
from predictor.stats_server import Stats
from predictor.serializers import PredictionSerializer


def home(request):
    latest_prediction = Photo.objects.latest('id')

    predictions = [latest_prediction.pred_all,
                   latest_prediction.pred_base,
                   latest_prediction.pred_none]
    formatted_prediction = [f'{p * 100:.2f}%' for p in predictions]
    return render(request, 'home.jinja2', {'photo_model': latest_prediction,
                                           'formatted_predictions': formatted_prediction})


def about(request):
    return render(request, 'about.jinja2')


def stats(request):
    heatmap = Plot.objects.filter(label='heatmap').first()
    statistics = Stats()
    return render(request, 'stats.jinja2', {'stats': statistics,
                                            'heatmap': heatmap})


def archive(request):
    context = {}
    if request.method == 'POST':
        form = ExampleForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data['display_date']
            start_time = form.cleaned_data['start_time']
            end_time = form.cleaned_data['end_time']

            if start_time is None:
                start_dt = dt.datetime(date.year, date.month, date.day)
            else:
                start_dt = dt.datetime.combine(date, start_time)

            if end_time is None:
                next_day = date + dt.timedelta(days=1)
                end_dt = dt.datetime(next_day.year, next_day.month, next_day.day)

            else:
                end_dt = dt.datetime(date.year, date.month, date.day, end_time.hour)

            aware_start_dt = make_aware(start_dt)
            aware_end_dt = make_aware(end_dt)
            results = Photo.objects.filter(datetime__gte=aware_start_dt,
                                           datetime__lte=aware_end_dt
                                           )
            if len(results) == 0:
                results = 3.14
            context.update({'results': results})

    else:
        form = ExampleForm()

    context['form'] = form

    return render(request, 'archive.jinja2', context)


def api(request):
    return render(request, 'api.jinja2')


class APIViewSetAll(viewsets.ModelViewSet):
    """
    API endpoint that returns all of the predictions
    """
    queryset = Photo.objects.all().order_by('-datetime')
    serializer_class = PredictionSerializer
    http_method_names = ['get']


class APIViewSetAllOut(viewsets.ModelViewSet):
    """
    API endpoint that returns predictions when She was 'All the way out'
    """
    queryset = Photo.objects.filter(winner='All the way out').order_by('-datetime')
    serializer_class = PredictionSerializer
    http_method_names = ['get']


@api_view(('GET',))
def last_prediction(request):
    """
    API endpoint that returns the last prediction
    """
    latest = Photo.objects.latest('id')
    serializer_class = PredictionSerializer(latest)
    return Response(serializer_class.data, status=200)
