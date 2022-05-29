from rest_framework import serializers

from predictor.models import Photo


class PredictionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Photo
        fields = ['winner', 'datetime', 'name',
                  'pred_all', 'pred_tip', 'pred_base', 'pred_none']
