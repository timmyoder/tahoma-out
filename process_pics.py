from PIL import Image

from config import EDIT_DIR, PHOTOS_DIR

photo = PHOTOS_DIR / '2021-02-27_13_55.png'

for photo in PHOTOS_DIR.glob('*.png'):
    im = Image.open(photo)
    width, height = im.size  # 1000 x 563

    # cut left side of image so the image is square
    im = im.crop((width-563, 0, width, height))

    small_size = 256, 256
    im.thumbnail(small_size)

    small_file = EDIT_DIR / photo.name
    im.save(small_file, 'png')



