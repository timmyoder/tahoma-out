original_image_size = (1080, 512)
resize_image_size = (128, 128)
mobile_net_image_size = (128, 128)


if __name__ == '__main__':
    from config import PHOTOS_DIR
    from PIL import Image
    directory = PHOTOS_DIR / 'training' / 'all_the_way'
    test_image = directory / '2020_04_17_1450.png'

    # Opens a image in RGB mode
    im = Image.open(test_image)

    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size

    # # Cropped image of above dimension
    # # (It will not change original image)
    im1 = im.crop((0, 0, width, width))
    # # Shows the image in image viewer
    im1.resize((128, 128))
    im1.show()
