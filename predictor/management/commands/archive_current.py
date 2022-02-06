from django.core.management.base import BaseCommand

from predictor.archiver import Archiver


class Command(BaseCommand):

    def handle(self, *args, **options):
        # load NYT deaths and cases
        archiver = Archiver()
        archiver.archive()
