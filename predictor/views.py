import datetime as dt

from django.shortcuts import render
from django.utils.timezone import make_aware

from predictor.models import Photo, Plot
from predictor.forms import ExampleForm
from predictor.stats_server import Stats


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
    heatmap = Plot.objects.filter(label='heatmap')
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
                end_dt = dt.datetime(date.year, date.month, date.day + 1)
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
