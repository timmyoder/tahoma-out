from predictor.models import Photo


class Stats:
    def __init__(self):
        self.last_out = Photo.objects.filter(winner='Not out').latest('id')

