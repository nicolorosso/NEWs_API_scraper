
# importing json module that will help converting the datastructures to JSON strings.
import json
# importing requests module to interact with APIs
import requests
# importing pandas for data manipulation 
import pandas as pd
# importing classes form datatime to deal with the date 
from datetime import date, timedelta
# importing nltk and downloading stopwords to later remove them
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#set the credentials to access data from the news API for the first dataframe. 

url = 'https://newsapi.org/v2/everything?'
# selecting the topic of interest 
topic = ['q=bitcoin&', 'q=finance&', 'q=metaverse&', 'q=recession&'] 
# selecting as range of dates the past month (max. period available)
x = date.today()    
y = x - timedelta(days=30)
date1 = f'from={y}&to={x}&'
sort = 'sortBy=popularity'
apikey= '&apiKey=fa422c784ca843a0bb09c0a9381a4abf'

# Access the API and storing the results in a json file 
news = []
for i in topic:
  news.append(requests.get(url+i+date1+sort+apikey).json())
# Normalize JSON data into a flat table
df_news = pd.json_normalize(news, record_path='articles')

#set the credentials to access data from the news API for the second dataframe.
url = 'https://newsapi.org/v2/everything?'
# selecting different topics of interest
topic = ['q=war&', 'q=election&', 'q=covid&', 'q=Trump&'] 
# selecting the same range of dates 
date2 = f'from={y}&to={x}&'
sort = 'sortBy=popularity'
apikey= '&apiKey=fa422c784ca843a0bb09c0a9381a4abf'

# Access the API and storing the results in a json file 
news2 = []
for i in topic:
  news2.append(requests.get(url+i+date2+sort+apikey).json())

# Normalize JSON data into a flat table
df_news2 = pd.json_normalize(news2, record_path='articles')

# Concatenate the two different dataframes to create a single one bigger 
df_final = pd.concat([df_news, df_news2], ignore_index=True)

"""DATA CLEANING FOR FUTURE MODELING"""

import re 
import string 

# lowercase for every words, Removing words within parenthesis, punctuations and words containing numbers 
def clean_text_round1(text): 
  text = text.lower()
  text = re.sub('\[.*?\]', "", text)
  text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
  text = re.sub("\w*\d\w*", "", text)
  return text

# Creation of a new column with cleaned text 
round1 = lambda x: clean_text_round1(x)
df_final['cleaned_text'] = df_final.description.apply(round1)

# Importing stop words in english
stop_words= set(stopwords.words('english'))

# Creation of a new column with cleaned text and no stop words 
df_final['cleaned_final'] = df_final['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

"""SENTIMENT ANALYSIS"""

#Installation of transformer library and importing the pipeline class from it
!pip install -q transformers
from transformers import pipeline

#Calling the pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Creation of a new column with sentiment analysis's reults 
df_final['sentiment'] = df_final.description.apply(lambda x: sentiment_pipeline(x))

# Unpacking the sentiment and score in the sentiment column in two distinct columns: sentiment and score
df_final['sentiment'], df_final['score'] = df_final.sentiment.apply(lambda x: x[0]['label']), df_final.sentiment.apply(lambda x: x[0]['score'])

"""DEFINING THE FINAL DATASET"""

# Converting the dtype of columns publishedAt from object to datetime
df_final.publishedAt = df_final.publishedAt.astype('datetime64')

# Adding a column with the count of words of the description for each row
df_final['description_length'] = df_final['description'].apply(lambda x: len(x.split(' ')))

#drop un-necessary columns
df_final = df_final.drop('cleaned_text', 1)
df_final = df_final.drop('source.id', 1)

#fill the nan values in the 'author' column
df_final['author'] = df_final['author'].fillna('author not indicated')

#save the final work in a csv file
df_final.to_csv("df_notizie.csv", index=False)

df_final


