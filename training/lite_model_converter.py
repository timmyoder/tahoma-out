import tensorflow as tf
from tensorflow import image
from tensorflow.keras.models import load_model

from training.image_sizes import mobile_net_image_size, resize_image_size

from config import MODELS_DIR


def load_saved_model(model_name):
    model = load_model(MODELS_DIR / f'{model_name}.h5',
                       custom_objects={'image': image,
                                       'mobile_net_image_size': mobile_net_image_size,
                                       'resize_image_size': resize_image_size})
    to_convert_dir = MODELS_DIR / 'to_convert' / model_name
    to_convert_dir.mkdir()
    tf.saved_model.save(model, str(to_convert_dir))


def save_model(model_name):
    # Convert the model
    model_dir = MODELS_DIR / 'to_convert' / model_name
    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_dir))
    tflite_model = converter.convert()

    # Save the model
    with open(MODELS_DIR / 'lite' / f'{model_name}.tflite', 'wb') as out_file:
        out_file.write(tflite_model)


if __name__ == '__main__':
    model_to_convert = 'mobilenetV2_fine_tuned'
    load_saved_model(model_to_convert)
    save_model(model_to_convert)
