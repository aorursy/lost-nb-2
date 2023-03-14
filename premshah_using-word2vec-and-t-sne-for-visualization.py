#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[4]:


data = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
data.head()
data_test.head()


# In[5]:


STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")
    
    for col in ['question1', 'question2']:
        data[col] = data[col].apply(clean_sentence)
    
    return data

data = clean_dataframe(data)
data.head(5)


# In[6]:


data_test = clean_dataframe(data_test)
data_test.head()


# In[ ]:


def build_corpus(data):
    corpus = []
    
    for col in ['question1','question2']:
         for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
     
    return corpus

corpus = build_corpus(data)        
corpus[0:2]


# In[ ]:


corpus_test = build_corpus(data_test)
corpus_test[0:2]


# In[ ]:


def build_corpus_q(data):
    corpus = []
    for sentence in data.iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
     
    return corpus

corpus_q1 = build_corpus_q(data['question1'])
corpus_q2 = build_corpus_q(data['question2'])


# In[ ]:


corpus_test_q1 = build_corpus_q(data_test['question1'])
corpus_test_q2 = build_corpus_q(data_test['question2'])


# In[ ]:


from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
model.wv['trump']


# In[ ]:


model_test = word2vec.Word2Vec(corpus_test, size=100, window=20, min_count=200, workers=4)
model_test.wv['trump']


# In[ ]:


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[ ]:


tsne_plot(model)


# In[ ]:


tsne_plot(model_test)


# In[ ]:


# A more selective model
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=1000, workers=4)
tsne_plot(model)


# In[ ]:


model.most_similar('india')


# In[ ]:


def get_tsne_vector(model):
    "Creates a TSNE model and use it with Word2Vec to find how similar both questions are"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    return labels, x, y


# In[ ]:


words, X, Y = get_tsne_vector(model)


# In[ ]:


words_test, X_test, Y_test = get_tsne_vector(model_test)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
tfidf_q1 = vectorizer.fit_transform(corpus_q1)
tfidf_q2 = vectorizer.fit_transform(corpus_q2)


# In[ ]:


tfidf_test_q1 = vectorizer.fit_transform(corpus_test_q1)
tfidf_test_q2 = vectorizer.fit_transform(corpus_test_q2)


# In[ ]:


def word2vec_and_tfidf(data,words,X,Y,tfidf_q1,tfidf_q2):
    for index in data:
        val1x, val1y = findVal(data.iloc(index)['question1'],words,X,Y,tfidf_q1)
        val2x, val2y = findVal(data.iloc(index)['question2'],words,X,Y,tfidf_q2)
        


# In[ ]:


def findVal(ques,words,X,Y,tfidf):
    var i = 0
    valx = []
    valy =[]
    for wrd in list(ques.split(" ")):
        var index = words[wrd]
        valx.append(tfidf[i]*X[index])
        valy.append(tfidf[i]&Y[index])
        i++
    
    return valx, valy

