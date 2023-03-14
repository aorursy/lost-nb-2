#!/usr/bin/env python
# coding: utf-8

# In[1]:



del testimport numpy as np
import seaborn as sns
import pandas as pd
import itertools 
import csv
import collections
import matplotlib.pyplot as plt
import re
import nltk
from bs4 import BeautifulSoup
from gensim.models import word2vec

from sklearn.manifold import TSNE


sns.set_context("paper")
get_ipython().run_line_magic('matplotlib', 'inline')

RES_DIR = "../input/"
# Load train data (skips the content column)
def load_train_data():
    categories = ['cooking', 'robotics', 'travel', 'crypto', 'diy', 'biology']
    train_data = []
    for cat in categories:
        data = pd.read_csv("{}{}.csv".format(RES_DIR, cat), usecols=['id', 'title', 'tags','content'])
        data['category'] = cat
        train_data.append(data)
    
    return pd.concat(train_data)
data = load_train_data()
print(data.head())


data=data.sample(3000)


# In[2]:


uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""

# This could take a while
data["title"] = data["title"].map(stripTagsAndUris)
data["content"] = data["content"].map(stripTagsAndUris)
import string
print(data.head())

def removePunctuation(x):
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    #x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)

# point questionmarks etc
data["title"] = data["title"].map(removePunctuation)
data["content"] = data["content"].map(removePunctuation)
#data = clean_dataframe(data)


# In[3]:


from gensim.parsing import PorterStemmer
global_stemmer = PorterStemmer()
 
class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """
 
    #This reverse lookup will remember the original forms of the stemmed
    #words
    word_lookup = {}
 
    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """
 
        #Stem the word
        stemmed = global_stemmer.stem(word)
 
        #Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)
 
        return stemmed
 
    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """
 
        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word
        
#data['title']=data['title'].replace('?',' ?')
data.head(5)# Summary about tags


# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

data['tot']=data['title']+' '+data['content']
#create sklearn pipeline, fit all, and predit test data
clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
                ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
                ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
                ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
clf.fit(data['tot'], data['tags'])



# In[5]:


test = pd.read_csv("../input/test.csv")
test['tags'] = ''
test['category'] = 'physics'

data=test.sample(3000)
data["title"] = data["title"].map(removePunctuation)
data["content"] = data["content"].map(removePunctuation)
data['tot']=data['title']+' '+data['content']
t_labels = clf.predict(data['tot'])
print(t_labels)

