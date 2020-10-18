#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm

from datetime import datetime
import re
from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

import re
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import plotly.express as px


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def preprocess(val):
    nltk.download('stopwords')
    stop = stopwords.words('english')
    input_txt = {'review_text' : [val]}
    val = pd.DataFrame(input_txt, columns=['review_text'])

    val['review_text'] = val['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    
    st = PorterStemmer()
    val['review_text'] = val['review_text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    
    regex = r'\d+'
    val['review_text'] = val['review_text'].apply(lambda x:" ".join(re.sub(regex, 'numbers', word) for word in x.split()))
    
    vectorizer = TfidfVectorizer()
    xtrain = pickle.load(open('C:/CS/Python/SpoilerAlert/model/train.pkl', 'rb'))
    vectorizer.fit_transform(xtrain)
    val = vectorizer.transform(list(val['review_text']))
    
    return val

def result(s):
    pred = preprocess(s)
    model = pickle.load(open('C:/CS/Python/SpoilerAlert/model/model.pkl', 'rb'))
    return model.predict(pred)
