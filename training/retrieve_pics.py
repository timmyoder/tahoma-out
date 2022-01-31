import shutil
import time
import datetime as dt

import requests
from loguru import logger

from config import PHOTOS_DIR, PHOTO_LOG, MEDIA_DIR

cam_url = 'https://cdn.tegna-media.com/king/weather/waterfront.jpg'

logger.add(PHOTO_LOG)


def training_data_downloader():
    while True:
        minute = dt.datetime.now().minute
        hour = dt.datetime.now().hour
        date = dt.datetime.now().date()
        label = f'{date}_{hour}_{minute}'
        photo_path = PHOTOS_DIR / f'{label}.png'

        if 4 < hour < 22:
            try:
                response = requests.get(cam_url, stream=True)
                logger.info(f'retrieving {photo_path.name}')

                with open(photo_path, 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
                logger.success(f'{photo_path.name} downloaded')
                del response
            except requests.exceptions.ConnectionError as error:
                logger.debug(error)
                time.sleep(60*3)
                continue
        time.sleep(60 * 8)


def get_current_image():
    live_path = MEDIA_DIR / 'live.png'
    try:
        response = requests.get(cam_url, stream=True)
        logger.info(f'retrieving live photo')

        with open(live_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        logger.success(f'live photo downloaded')
        del response
    except requests.exceptions.ConnectionError as error:
        logger.debug(error)
