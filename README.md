# Sentiment-Analysis

NOTE >> THIS PROJECT IS DEPRECATED AND NO LONGER WORKS AS CREATED. Twitter has blocked scrapers from functioning, which this application requires, Twitter now needs an account and user token to search. However, the project with simplified into a basic sentiment analysis web app.
Model was also updated to use a PyTorch model instead of a TensorFlow model.



This GitHub repository contains the code for a sentiment analysis web-interface built with Django.

The sentiment analysis model is developed using the BiLSTM algorithm, trained and tested on the publicly available IMDB dataset.

The code for the scraping, preprocessing, and evaluation of tweets can be found in the 'mysite/home/views.py' file, which is responsible for handling the deployment of the pre-trained model to the web-interface. 

The web page is hosted on a VPS with basic NGINX and UWSGI configuration.

The code imports required libraries and loads the pre-trained model on website load, then gathers, cleans, and sorts the dataset created from Twitter. The final results are displayed in the form of 3 images of varying graphs, which are stored in the static folder for later use. The HTML files for the home page and results page, along with their respective CSS files, can be found in the templates and static/CSS folders respectively. 

This model achieved an accuracy of ~84% on the validation set. This web-interface can help identify and address negative customer feedback, leading to an improvement in customer retention and overall customer loyalty.

![image](https://github.com/wowimnick/Sentiment-Analysis/blob/main/mysite/static/menu.png?raw=true)
