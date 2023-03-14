#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from tensorflow.keras.callbacks import EarlyStopping


# In[3]:


df=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")
df.head()


# In[4]:


def clean_description(description):
    document = re.sub(r'\W', ' ', str(description))
    document = re.sub(' +', ' ', document)        
    document = document.lower()
    document = re.sub('\(', " ", document)
    document = re.sub('\)', " ", document)
    document = re.sub('-', "", document)
    document = re.sub('&', "", document)
    document = re.sub('&', "", document)
    document = re.sub('|', "", document)
    document = re.sub('\/', " ", document)
    document = re.sub("\'", "", document)
    document = re.sub('\"', "", document)
    document = re.sub(',', "", document)
    document = re.sub('[0-9]', "", document)
    document = document.split()
    document=' '.join( [w for w in document if len(w)>1 and (w.lower() not in STOPWORDS)] )
    return document


# In[5]:


df['question_text']=df['question_text'].apply(lambda x: clean_description(x))


# In[6]:


MAX_NB_WORDS = 50000
# Max number of words in each query.
MAX_SEQUENCE_LENGTH = 400
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['question_text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[7]:


X = tokenizer.texts_to_sequences(df['question_text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# In[8]:


Y = pd.get_dummies(df['target']).values


# In[9]:


def createmodel():
  model = Sequential()
  model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
  model.add(SpatialDropout1D(0.2))
  model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  return model


# In[10]:


model = createmodel()

epochs = 2
batch_size = 1028

history = model.fit(X, Y, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[11]:


df_test=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
df_test['question_text']=df_test['question_text'].apply(lambda x: clean_description(x))

tokenizertest = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizertest.fit_on_texts(df_test['question_text'].values)

X_test = tokenizer.texts_to_sequences(df_test['question_text'].values)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
ynew = model.predict_classes(X_test)


# In[12]:


df_submission=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
df_submission["prediction"]=ynew
df_submission=df_submission[["qid","prediction"]]
df_submission.to_csv("submission.csv",index=False)

