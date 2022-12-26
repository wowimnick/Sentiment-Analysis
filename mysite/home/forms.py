from django import forms
  
# creating a form 
class HomeForm(forms.Form):
    checkbox2 = forms.BooleanField(label = 'checkboxid')