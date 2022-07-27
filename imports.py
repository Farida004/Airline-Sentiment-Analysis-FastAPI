
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle

import re
import nltk
from wordcloud import WordCloud
nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,classification_report