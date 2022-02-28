from django.shortcuts import render

from django.utils.timezone import make_aware

from predictor.models import Photo
from predictor.forms import SpecificDateForm


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
    context = {}
    if request.method == 'POST':
        form = SpecificDateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data['display_date']
            import datetime as dt
            date = dt.datetime(date.year, date.month, date.day)
            aware_date = make_aware(date)
            results = Photo.objects.filter(datetime__year=aware_date.year,
                                           datetime__month=aware_date.month,
                                           datetime__day=aware_date.day)
            context.update({'results': results})

    else:
        form = SpecificDateForm()

    context['form'] = form

    return render(request, 'archive.jinja2', context)


def api(request):
    return render(request, 'api.jinja2')
