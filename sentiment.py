from fastapi import FastAPI
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import CountVectorizer

import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

app = FastAPI()

@app.get('/')
def get_root():
    return {'message': 'This is the sentiment analysis app'}

@app.get('/sentiment_analysis/')
async def query_sentiment_analysis(text: str):
    return analyze_sentiment(text)

def analyze_sentiment(text):
    model = pickle.load(open("rff-model.pkl", "rb"))
    v = pickle.load(open("v.pkl", "rb"))
    result = model.predict(v.transform([text]))
    sent = ''
    if (result == 'negative'):
        sent = 'negative'
    elif (result =='positive'):
        sent = 'positive'

    # Format and return results
    return {'sentiment': sent}
