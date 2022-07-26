from fastapi import APIRouter, Response, status, Query
from enum import Enum
from typing import Optional

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

router = APIRouter(
    prefix='/sentiment_analysis',
    tags=['sentiment_analysis']
)

@router.get(
  '/',
  summary='This is a sentiment analysis app',
  description='This API runs a pretrained Random Forest Classifier on airline sentiment data and predicts a sentiment for a user given text',
  response_description="A sentiment which can be either positive or negative.",
  status_code=status.HTTP_200_OK
  )
async def query_sentiment_analysis( response: Response, text: str = Query(min_length=10, max_length=200)):
    response.status_code = status.HTTP_200_OK
    return analyze_sentiment(text)

def analyze_sentiment(text):
    model = pickle.load(open("rfc-model.pkl", "rb"))
    v = pickle.load(open("vectorizer.pkl", "rb"))
    result = model.predict(v.transform([text]))
    sent = ''
    if (result == 'negative'):
        sent = 'negative'
    elif (result =='positive'):
        sent = 'positive'

    # Format and return results
    return {'sentiment': sent}