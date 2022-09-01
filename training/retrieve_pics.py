import shutil
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import pandas as pd
import requests
from loguru import logger

from config import PHOTOS_DIR, PHOTO_LOG, MEDIA_DIR, FOUR_PREDICTIONS

cam_url = 'https://cdn.tegna-media.com/king/weather/waterfront.jpg'

base_url = 'https://spaceneedledev.com/panocam/assets'

logger.add(PHOTO_LOG)

START_DATE = dt.datetime(year=2020, month=1, day=1)
END_DATE = dt.datetime(year=2022, month=6, day=1)


def get_pic_url(year, month, day, hour_min):
    return f'{base_url}/{year}/{month}/{day}/{year}_{month}{day}_{hour_min}00/slice8.jpg'


def datetime_formatter(input_dt):
    year = input_dt.strftime('%Y')
    month = input_dt.strftime('%m')
    day = input_dt.strftime('%d')
    hour_min = input_dt.strftime('%H%M')
    return year, month, day, hour_min


def download_single_image(year, month, day, hour_min):
    label = f'{year}_{month}_{day}_{hour_min}'
    photo_path = PHOTOS_DIR / f'{label}.png'

    url = get_pic_url(year, month, day, hour_min)
    if 400 < int(hour_min) < 2200:
        try:
            response = requests.get(url, stream=True)
            logger.info(f'retrieving {url}')
            if response.status_code == 404:
                logger.debug(f'{photo_path.name} NOT downloaded. 404')
                return False

            with open(photo_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            logger.success(f'{photo_path.name} downloaded')
            del response
        except requests.exceptions.ConnectionError as error:
            logger.debug(error)


def training_data_downloader(start, end):
    all_dates = pd.date_range(start, end, freq='10min').tolist()
    formatted_dates = [datetime_formatter(date_) for date_ in all_dates]
    formatted_dates.reverse()

    with ThreadPoolExecutor(max_workers=100) as executor:
        executor.map(lambda args: download_single_image(*args), formatted_dates)


def get_current_image():
    live_path = MEDIA_DIR / 'live.png'

    if FOUR_PREDICTIONS:
        url = cam_url
    else:
        now = dt.datetime.now()
        last_time = now - dt.timedelta(minutes=now.minute % 10,
                                       seconds=now.second,
                                       microseconds=now.microsecond)
        last_time = last_time - dt.timedelta(minutes=20)
        url = get_pic_url(*datetime_formatter(last_time))
    try:
        response = requests.get(url, stream=True)
        logger.info(f'retrieving live photo')

        with open(live_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        logger.success(f'live photo downloaded')
        del response
    except requests.exceptions.ConnectionError as error:
        logger.debug(error)


if __name__ == '__main__':
    training_data_downloader(start=START_DATE, end=END_DATE)
