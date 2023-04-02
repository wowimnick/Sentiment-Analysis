from django.shortcuts import render
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd
import snscrape.modules.twitter as sntwitter
import re
from wordcloud import WordCloud
import nltk
from wordcloud import STOPWORDS
from langdetect import detect
matplotlib.use('Agg')
plt.style.use('ggplot')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], 
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)])
        print("Success: GPU memory limit set to 256MB")
    except RuntimeError as e:
        print(e)

# The model is loaded as soon as the main page is loaded
model = tf.keras.models.load_model('home/model')

def home(request):
    return render(request, "main.html")

def results(request):
    # Get user input and parameters
    input = request.GET['text']
    max = request.GET['amount']
    startdate = request.GET.get('startdate')
    enddate = request.GET.get('enddate')

    # Checks if a time range is present, if so includes the dates in the twitter search query
    if startdate and enddate:
        input += f' since:{startdate} until:{enddate}'

    output = []
    for tweet in sntwitter.TwitterSearchScraper(input).get_items():
        if len(output) >= int(max):
            break
        output.append([tweet.content])

    # If no tweets were found, show an error page
    if not output:
        return error_page(request)

    # Gathers data in a Pandas dataframe, assigns it only one column which lists the scraped tweets individually
    data = cleandata(pd.DataFrame(output, columns=["text"]))
    predictions = modelevaluate(data)

    # Creates a seperate list of sentiment results for the individual tweets, if the result < 5 then it get put in the lessthan array.
    # otherwise it gets put in the larger array.
    # The array which is larger is the result of the user query
    positive = [p for p in predictions if p > 5]
    negative = [n for n in predictions if n < 5]
    if len(positive) > len(negative):
        ans = "POSITIVE"
    else:
        ans = "NEGATIVE"
    biggerthanlen = len(positive)
    lessthanlen = len(negative)

    c = sorted(collections.Counter(predictions).items()) 
    pos_num, freq = zip(*sorted(c.items()))

    colors = ['#FF6464', '#FF9164', '#FFC564', '#FFDE64', '#FFEE64',
              '#F8FF64', '#E7FF64', '#CFFF64', '#B3FF64', '#82FF64']

    # Creates the bar chart displaying Positivity in respect to Frequency 
    plt.bar(pos_num, freq, color=colors,  edgecolor='#922ffe')
    plt.title("Positivity", color='white')
    plt.xlabel("Positivity Amount")
    plt.ylabel("Frequency")
    plt.savefig('../mysite/static/frequency.png',
                bbox_inches='tight', pad_inches=0, transparent=True)
    plt.clf()


    # Creates a pie chart displaying the same data as the bar chart

    plt.pie(freq, labels=pos_num, autopct='%.0f%%', pctdistance=0.85,
            colors=colors, shadow=True, startangle=90, textprops={'color': "#4c4a4f"})
    plt.title('Frequency of Positivity Occurences', color='white')

    centre_circle = plt.Circle((0, 0), 0.70, fc='#0f0817')
    plt.gcf().gca().add_artist(centre_circle)
    plt.savefig('../mysite/static/pie.png', transparent=True)
    plt.clf()


    # Creates a wordcloud plot, showing the frequency of words in the user query results

    plt.imshow(WordCloud(background_color="#0f0817", max_words=1000, height=570, width=900,
               stopwords=STOPWORDS).generate_from_frequencies(formatwordcloud(data)), interpolation="bilinear")
    plt.axis("off")
    plt.savefig('../mysite/static/wordcloud.png',
                bbox_inches='tight', pad_inches=0, transparent=True)
    plt.clf()

    return render(request, "results.html", {'ans': ans, 'biggerthanlen': biggerthanlen, 'lessthanlen': lessthanlen})

# Input: Dataset
# Output: An array of sentiment positivity per each message
def modelevaluate(data):
    predictions = model.predict(data)
    predictions = np.round_(predictions, decimals=1)
    predictions = predictions.flatten()
    return [int(str(i)[2:]) for i in predictions]

# Input: Dataset
# Output: Tokenizes the sentences into words and creates a dictionary with the frequency of each word
def formatwordcloud(data):
    tokens = data.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
    token_counts = collections.Counter(tokens)
    token_counts = sorted(token_counts.items())
    return {word: count for word, count in sorted(token_counts.items())}

# Input: Dataset
# Output: Cleans the dataset of emojis, uppercase, punctuation, links, tags, hashtags, nulls, and empty rows
def cleandata(inputdataset):
    emoj = re.compile(u"[\U0001F600-\U0010ffff]+", re.UNICODE)
    inputdataset['text'] = inputdataset['text'].apply(lambda x: x.lower().replace(emoj, '') if x.isascii() else x)
    inputdataset['text'] = inputdataset['text'].replace(
        '@[a-zA-Z0-9$-_@.&+!*\(\),]*|#[a-zA-Z0-9$-_@.&+!*\(\),]*|https?://[a-zA-Z0-9$-_@.&+!*\(\),]*|[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', regex=True)
    inputdataset['text'] = inputdataset['text'].str.replace('/n', ' ')
    inputdataset = inputdataset.replace('nan', pd.NA).dropna()
    inputdataset = inputdataset[inputdataset['text'].str.len() >= 20]
    return inputdataset


def error_page(request):
    return render(request, "error.html", {"error_message": "No tweets could be found. Please try again with a different search term or settings."})
