from pytz import timezone
import pandas as pd

from predictor.models import Photo


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
        self.max_out_date = None
        self.max_out_time = None
        self.most_out()

    def most_out(self):
        query = Photo.objects.filter(winner='Not out')
        all_winners = pd.DataFrame.from_records(query.values())
        all_winners['date'] = pd.to_datetime(all_winners['datetime']).dt.date

        out_by_day = all_winners.groupby('date').count()

        self.max_out_date = out_by_day.index[out_by_day['id'].argmax()]
        max_out_count = out_by_day.loc[self.max_out_date, 'id']
        max_out_total_minutes = max_out_count * 10
        max_out_hours = max_out_total_minutes // 60
        if max_out_hours < 1:
            hrs_str = ""
        elif max_out_hours == 1:
            hrs_str = f'{max_out_hours} hour '
        else:
            hrs_str = f'{max_out_hours} hours and '
        max_out_minutes = max_out_total_minutes % 60

        if max_out_minutes == 0:
            min_str = ''
        else:
            min_str = f'{max_out_minutes} minutes'

        self.max_out_time = f"{hrs_str}{min_str}"


if __name__ == '__main__':

    s = Stats()
