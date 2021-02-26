import shutil
import time
import datetime as dt

import requests
from loguru import logger

# todo download smaller image
from PIL import Image
import pyvips

from config import PHOTOS_DIR, PHOTO_LOG

cam_url = 'https://cdn.tegna-media.com/king/weather/waterfront.jpg'

# todo: add multiple logs, move to config.py
logger.add(PHOTO_LOG)

while True:
    time.sleep(60 * 2)
    hour = dt.datetime.now().hour
    date = dt.datetime.now().date()
    if hour < 8 | hour > 19:
        continue
    label = f'{date}_{hour}'
    photo_path = PHOTOS_DIR / f'{label}.png'
    response = requests.get(cam_url, stream=True)

    with open(photo_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    logger.success(f'{photo_path.name} downloaded')

    del response
