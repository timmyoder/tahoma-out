from django import forms


class DatePickerInput(forms.DateInput):
    input_type = 'date'


class TimePickerInput(forms.TimeInput):
    input_type = 'time'


class ExampleForm(forms.Form):
    display_date = forms.DateField(widget=DatePickerInput)
    start_time = forms.TimeField(widget=TimePickerInput, required=False)
    end_time = forms.TimeField(widget=TimePickerInput, required=False)
