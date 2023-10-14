from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import re
#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')

#news = 'India won the ICC 2023 world cup'
#news = 'Bala Subramaniam gets the best singer award'
#news = 'Intel starts its manufacturing hub in Sompeta'
#news = 'India\'s currency values beats the US dollors and touches the record high value'
#news = 'Dhiraj gets elected as the youngest Prime Minister'



def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)

    # general
    phrase = re.sub("n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# This collection of Stopwords was taken from Internet
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

def preprocess_text(sentence):
    '''
    Basic Sentence Preprocessing
    '''
    sentence = decontracted(sentence)
    sentence = re.sub(r'\s', ' ', sentence) # Removing new line Characters
    sentence = re.sub(r' +', ' ', sentence) # Removing extra spaces
    sentence = re.sub('[^A-Za-z]+', ' ', sentence) # Removing non-letters
    sentence = ' '.join(e for e in sentence.split() if e.lower() not in stopwords) # Removing Stopwords
    return sentence.lower().strip()



lemmatizer = WordNetLemmatizer()

#def lematize_sentence(sentence):
#    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence)])

st.title("Get your NEWS tagged!")
news = st.text_input('Please enter the NEWS') 

if not news:
  st.stop()

news = preprocess_text(news)
#news = lematize_sentence(news)

vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

news = vectorizer.transform([news])

model = pickle.load(open('news_tagger.sav', 'rb'))

prediction = model.predict(news)[0]

mappings = {0:'Business', 1:'Entertainment', 2:'Politics', 3:'Sports', 4:'Technology'}

category = None 
category = mappings.get(prediction)

st.write('The NEWS is from ', category)