import pathlib

ROOT = pathlib.Path(__file__).parent
MODELS_DIR = ROOT / 'models'
PHOTOS_DIR = ROOT / 'data' / 'raw_images'
SRC = ROOT / 'src'
LOGS = ROOT / 'logs'

EDIT_DIR = PHOTOS_DIR / 'edited'

PHOTO_LOG = LOGS / 'p.log'
