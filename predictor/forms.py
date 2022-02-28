from django import forms
from django.forms.widgets import NumberInput

from predictor.models import Photo


class SpecificDateForm(forms.Form):
    display_date = forms.DateField(widget=NumberInput(attrs={'type': 'date'}))

