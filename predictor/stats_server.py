from predictor.models import Photo
from pytz import timezone


def date_formatter(date_to_format):
    pst = timezone('US/Pacific')
    date_to_format = date_to_format.astimezone(pst)
    date_time = date_to_format.strftime('%b. %-d, %Y, %-I:%M')
    am_pm = date_to_format.strftime('%p').lower()
    am_pm = f'{am_pm[0]}.{am_pm[1]}.'
    return f'Photo pulled {date_time} {am_pm}'


class Stats:
    def __init__(self):
        self.images = Photo.objects.filter(winner='All the way out').order_by('-id').all()
        self.last_out = Photo.objects.filter(winner='All the way out').order_by('-id').first()
        self.image_urls = [photo.image.url for photo in self.images]
        self.labels = [date_formatter(photo.datetime) for photo in self.images]

