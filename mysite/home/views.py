from django.shortcuts import render
from django.http import JsonResponse
import torch
import torchtext
import matplotlib
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd
from .model import BRNN
from wordcloud import WordCloud
import spacy

matplotlib.use('Agg')
plt.style.use('ggplot')

# Load the model
model = BRNN(25002, 1)
cp = torch.load('home/model.pt', map_location=torch.device('cpu'))
model.load_state_dict(cp)
model.eval()
nlp = spacy.load("en_core_web_sm")

# Load the vocabulary
TEXT = torchtext.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
TEXT.vocab = torch.load('home/vocab.pt')

def home(request):
    print(request.GET.get('text', ''))
    if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and request.method == "GET":
        input_text = request.GET.get('text', '')
        if input_text:
            predictions = modelevaluate(input_text)
            # Truncate the decimals
            predictions = round(predictions, 2)
        else:
            predictions = 'N/A'
        return JsonResponse({'predictions': predictions})

    return render(request, "main.html")

# Input: Dataset
# Output: An array of sentiment positivity per each message
def modelevaluate(input):
    input = TEXT.preprocess(input)
    input = [TEXT.vocab.stoi[i] for i in input]
    input = torch.tensor(input).to('cpu')
    input = input.unsqueeze(1)
    output = model(input)
    return torch.sigmoid(output).item()

def error_page(request):
    return render(request, "error.html", {"error_message": "No tweets could be found. Please try again with a different search term or settings."})
