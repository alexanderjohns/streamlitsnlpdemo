import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model


def text_cleaning(text):
    text = text.lower()
    # split into tokens by white space
    tokens = text.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def dummy_fun(doc):
    return doc



df = pd.read_csv("train.csv")
df['sentiment'] = df['sentiment'].replace({'negative' : 0, 'positive' : 1})
df['review'] = df['review'].apply(text_cleaning)


cv_trigrams = CountVectorizer(analyzer='word',
                              tokenizer=dummy_fun,
                              preprocessor=dummy_fun,
                              max_features =100000,
                              token_pattern=None, ngram_range =(1,3))

tfidf_bigrams =  TfidfVectorizer(analyzer='word',
                              tokenizer=dummy_fun,
                              preprocessor=dummy_fun,
                              max_features =100000,
                              token_pattern=None, ngram_range=(1,2))

tfidf_trigrams =  TfidfVectorizer(analyzer='word',
                              tokenizer=dummy_fun,
                              preprocessor=dummy_fun,
                              max_features =100000,
                              token_pattern=None, ngram_range=(1,3))



X1 = cv_trigrams.fit_transform(df['review'])
X2 = tfidf_bigrams.fit_transform(df['review'])
X3 = tfidf_trigrams.fit_transform(df['review'])

lr_1 = pickle.load(open("lr_1.sav", 'rb'))
lr_2 = pickle.load(open("lr_2.sav", 'rb'))
nb_1 = pickle.load(open("nb_1.sav", 'rb'))
nb_2 = pickle.load(open('nb_2.sav', 'rb'))
svc_1 = pickle.load(open('svc_1.sav', 'rb'))
svc_2 = pickle.load(open('svc_2.sav', 'rb'))



st.title("Sentiment Analysis")
st.header("Enter Text Below")

sentence = st.text_input("Input Text Here:")

if sentence:
    sentence = text_cleaning(sentence)
    arr = np.array([sentence])
    x_cv3 = cv_trigrams.transform(arr)
    x_tf2 = tfidf_bigrams.transform(arr)
    x_tf3 = tfidf_trigrams.transform(arr)

    svc_1_result = svc_1.predict(x_tf2)
    svc_2_result = svc_2.predict(x_tf3)
    lr_1_result = lr_1.predict_proba(x_tf3)
    lr_2_result = lr_2.predict_proba(x_cv3)
    nb_1_result = nb_1.predict_proba(x_tf3)
    nb_2_result = nb_2.predict_proba(x_cv3)


    st.write("SVC_1:")
    st.write(svc_1_result)
    st.write("SVC_2:")
    st.write(svc_1_result)
    st.write("LR_1:")
    st.write(lr_1_result)
    st.write("LR_2:")
    st.write(lr_2_result)
    st.write("NB_1:")
    st.write(nb_1_result)
    st.write("NB_2:")
    st.write(nb_2_result)
