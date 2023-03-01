# Sentiment-Analysis

This GitHub repository contains the code for a sentiment analysis web-interface built with Django. 

The code for the scraping, preprocessing, and evaluation of tweets can be found in the '/home/views.py' file, which is responsible for handling the deployment of the pre-trained model to the web-interface. 

The web page is hosted on a VPS with basic NGINX and UWSGI configuration.

The code imports required libraries and loads the pre-trained model on website load, then gathers, cleans, and sorts the dataset created from Twitter. The final results are displayed in the form of 3 images of varying graphs, which are stored in the static folder for later use. The HTML files for the home page and results page, along with their respective CSS files, can be found in the templates and static/CSS folders respectively. 

This model achieved an accuracy of ~84% on the validation set. This web-interface can help identify and address negative customer feedback, leading to an improvement in customer retention and overall customer loyalty.

![image](https://user-images.githubusercontent.com/65257805/222081438-11e0cc2b-942c-44dd-8c03-f9b46f66faf5.png)
