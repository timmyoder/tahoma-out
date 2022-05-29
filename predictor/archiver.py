import datetime as dt

import cloudinary.uploader as uploader
from PIL import Image

from predictor.models import Photo, Plot
from predictor.prediction import Predictor
from predictor.plotting import Heatmap
from training.retrieve_pics import get_current_image
from training.image_sizes import original_image_size
from config import MEDIA_DIR

LIVE_MODEL = 'mobilenetV2_fine_tuned.h5'


class Archiver(Predictor):
    def __init__(self, model_name=LIVE_MODEL, image=MEDIA_DIR / 'live.png',
                 output_im_size=original_image_size):
        super().__init__(model_name=model_name, prediction_image=image)
        self.output_im_size = output_im_size

    def load_and_predict(self):
        self.load_image()
        self.predict()

    def update_image(self):
        get_current_image()
        self.resize_image()

    def resize_image(self):
        im = Image.open(self.current_image)
        height, width = self.output_im_size
        new_image = im.resize((width, height))
        new_image.save(self.current_image)

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

    @staticmethod
    def upload_heatmap():
        Heatmap()
        Plot.objects.update_or_create(label='heatmap',
                                      heatmap_plot=uploader.upload_resource(
                                          MEDIA_DIR / 'heatmap_plot.png',
                                          public_id="plots/heatmap",
                                          overwrite=True,
                                          resource_type='image',
                                          invalidate=True, )
                                      )

    def archive(self):
        self.update_image()
        self.load_and_predict()
        self.upload_prediction()
        self.upload_heatmap()
