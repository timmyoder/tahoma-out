import datetime as dt

import cloudinary.uploader as uploader

from predictor.models import Photo
from training.retrieve_pics import get_current_image
from config import MEDIA_DIR
from predictor.prediction import Predictor

LIVE_MODEL = 'mobilenetV2_fine_tuned.h5'


class Archiver(Predictor):
    def __init__(self, model_name=LIVE_MODEL, image=MEDIA_DIR / 'live.png'):
        super().__init__(model_name=model_name, prediction_image=image)

    def load_and_predict(self):
        self.load_image()
        self.predict()

    @staticmethod
    def update_image():
        get_current_image()

    def upload_prediction(self):
        minute = dt.datetime.now().minute
        hour = dt.datetime.now().hour
        date = dt.datetime.now().date()
        label = f'{date}_{hour}_{minute}'

        predictions = [p for p in self.prediction.reshape(-1)]

        pred_all = predictions[0]
        pred_base = predictions[1]
        pred_tip = predictions[3]
        pred_none = predictions[2]

        winner_id = self.prediction.argmax()

        new_photo = Photo.objects.create(name=label,
                                         image=uploader.upload_resource(self.current_image),
                                         winner=self.winner,
                                         winner_id=winner_id,
                                         pred_all=pred_all,
                                         pred_base=pred_base,
                                         pred_none=pred_none,
                                         pred_tip=pred_tip,
                                         model=self.model_name
                                         )

    def archive(self):
        self.update_image()
        self.load_and_predict()
        self.upload_prediction()
