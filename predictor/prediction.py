import numpy as np

from tensorflow import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from training.image_sizes import mobile_net_image_size, resize_image_size

from config import MODELS_DIR, MEDIA_DIR


class Predictor:
    def __init__(self, model_name='simple_sequential_small.h5',
                 prediction_image=MEDIA_DIR / 'dne.png'):
        self.prediction = None
        self.model_name = model_name
        self.model = load_model(MODELS_DIR / 'production' / self.model_name,
                                custom_objects={'image': image,
                                                'mobile_net_image_size': mobile_net_image_size,
                                                'resize_image_size': resize_image_size})
        self.current_image = prediction_image
        self.current_image_array = None
        self.labels = ['All the way out',
                       'Just the bottom',
                       'Not out',
                       'Only the tip']
        self.formatted_predictions = None
        self.winner = None

    def load_image(self):
        image = load_img(self.current_image)
        input_arr = img_to_array(image)
        self.current_image_array = np.array([input_arr])  # Convert single image to a batch.

    def predict(self):
        self.prediction = self.model.predict(self.current_image_array)

        formatted = [f'{p*100:.2f}%' for p in self.prediction.reshape(-1)]
        self.formatted_predictions = formatted
        self.winner = self.labels[self.prediction.argmax()]


