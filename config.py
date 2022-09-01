import pathlib

ROOT = pathlib.Path(__file__).parent
MODELS_DIR = ROOT / 'models'
PHOTOS_DIR = ROOT / 'data' / 'raw_images_needle'
SRC = ROOT / 'src'
LOGS = ROOT / 'logs'

STATIC_DIR = ROOT / 'predictor' / 'static'
PREDICTOR_DIR = ROOT / 'predictor'
MEDIA_DIR = ROOT / 'media'

PHOTO_LOG = LOGS / 'p.log'
MODEL_LOG = LOGS / 'model.log'

FOUR_PREDICTIONS = True
MODEL = 'mobilenetV2_fine_tuned.h5'
