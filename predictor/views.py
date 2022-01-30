from django.shortcuts import render
from predictor.prediction import Predictor

import numpy as np
import shutil

from config import PHOTOS_DIR, MEDIA_DIR
from retrieve_pics import get_current_image


def render_random(request):
    all_pics = [pic for pic in PHOTOS_DIR.rglob('*.png')]

    rand_pic = np.random.choice(all_pics)
    to_file = MEDIA_DIR / 'random_picture.png'

    shutil.copy(str(rand_pic), str(to_file))

    predictor = Predictor(image=rand_pic, model_name='simple_cnn.h5')
    predictor.load_image()
    predictor.predict()

    return render(request, 'home.jinja2', {'image': rand_pic.name,
                                           "predictor": predictor})


def render_live(request):
    get_current_image()
    live_im = MEDIA_DIR / 'live.png'
    predictor = Predictor(image=live_im, model_name='simple_cnn.h5')
    predictor.load_image()
    predictor.predict()

    return render(request, 'home.jinja2', {'image': live_im.name,
                                           "predictor": predictor})


def home(request):
    return render_live(request)


def sources(request):
    return render(request, 'sources.jinja2')
