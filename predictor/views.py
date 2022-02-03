from django.shortcuts import render
from predictor.prediction import Predictor

import numpy as np
import shutil

from config import PHOTOS_DIR, MEDIA_DIR
from training.retrieve_pics import get_current_image

model = 'mobilenetV2_fine_tuned.h5'


def render_random(request):
    all_pics = [pic for pic in PHOTOS_DIR.rglob('*.png')]

    rand_choice = np.random.choice(all_pics)
    rand_pic = MEDIA_DIR / 'random_picture.png'

    shutil.copy(str(rand_choice), str(rand_pic))

    predictor = Predictor(model_name=model, prediction_image=rand_pic)
    predictor.load_image()
    predictor.predict()

    return render(request, 'home.jinja2', {'image': rand_pic.name,
                                           "predictor": predictor})


def render_live(request):
    get_current_image()
    live_im = MEDIA_DIR / 'live.png'
    predictor = Predictor(model_name=model, prediction_image=live_im)
    predictor.load_image()
    predictor.predict()

    return render(request, 'home.jinja2', {'image': live_im.name,
                                           "predictor": predictor})


def home(request):
    return render_live(request)
    # return render_random(request)


def about(request):
    return render(request, 'about.jinja2')


def stats(request):
    return render(request, 'stats.jinja2')


def archive(request):
    return render(request, 'archive.jinja2')


def api(request):
    return render(request, 'api.jinja2')
