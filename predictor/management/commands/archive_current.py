from django.core.management.base import BaseCommand

from predictor.archiver import Archiver


class Command(BaseCommand):

    def handle(self, *args, **options):
        # pull down live picture and store prediction to database
        archiver = Archiver()
        archiver.archive()
