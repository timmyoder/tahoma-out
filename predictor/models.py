from django.db import models
from cloudinary.models import CloudinaryField


class Photo(models.Model):
    datetime = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=50)
    image = CloudinaryField('image')
    winner = models.CharField(null=True, max_length=20)
    winner_id = models.IntegerField(null=True)
    pred_all = models.FloatField(null=True)
    pred_tip = models.FloatField(null=True)
    pred_base = models.FloatField(null=True)
    pred_none = models.FloatField(null=True)
    model = models.CharField(null=True, max_length=50)


# todo: finish switching the databases
# class Predictions(models.Model):
#     datetime = models.DateTimeField(auto_now_add=True)
#     name = models.CharField(max_length=50)
#     url = models.URLField()
#     winner = models.CharField(max_length=20)
#     winner_id = models.IntegerField()
#     pred_all = models.FloatField(null=True)
#     pred_kinda = models.FloatField(null=True)
#     pred_none = models.FloatField(null=True)
#     model = models.CharField(null=True, max_length=50)


class Plot(models.Model):
    label = models.CharField(max_length=50, primary_key=True)
    datetime = models.DateTimeField(auto_now_add=True)
    heatmap_plot = CloudinaryField('heatmap_plot')

