from django.shortcuts import render
from predictor.prediction import Predictor

import numpy as np
import shutil

from config import PHOTOS_DIR, STATIC_DIR


def home(request):
    all_pics = [pic for pic in PHOTOS_DIR.rglob('*.png')]

    rand_pic = np.random.choice(all_pics)
    to_file = STATIC_DIR / 'current_pic.png'

    shutil.copy(str(rand_pic), str(to_file))

    predictor = Predictor(image=rand_pic)
    predictor.load_image()
    predictor.predict()

    return render(request, 'home.jinja2', {'image': rand_pic,
                                           "predictor": predictor})


def sources(request):
    return render(request, 'sources.jinja2')
