#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from collections import defaultdict
import os
import re
import warnings
import random

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

print(os.listdir("../input"))

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', -1)
warnings.filterwarnings('ignore')


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set_style('darkgrid')
sns.set_palette('deep')
figSize = (10, 6)

percentiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# Any results you write to the current directory are saved as output.


# In[2]:


# *** Read Data ***
trainDf = pd.read_csv('../input/train.csv')
testDf  = pd.read_csv('../input/test.csv')


# In[3]:


# *** Get a View of Data ***
trainDf.head()


# In[4]:


# *** Train and Test Shape ***
print('Train Data - Number of Rows : {} Number of Columns : {}'.format(trainDf.shape[0], trainDf.shape[1]))
print('Test Data  - Number of Rows : {} Number of Columns : {}'.format(testDf.shape[0], testDf.shape[1]))


# In[5]:


# *** Check Data Types ***
dTypeDist = defaultdict(int)
for aDict in trainDf.dtypes.reset_index().rename(columns = {'index' : 'colName', 0 : 'dType'}).to_dict(orient='records'):
    dTypeDist[str(aDict['dType']).replace('dtype(', '').replace(')', '')] += 1

dTypeDistDf = pd.DataFrame.from_dict(dTypeDist, orient = 'index').                  reset_index().                  rename(columns = {'index' : 'dType', 0 : 'colCount'})
                 
#print(dTypeDistDf)

fig, ax = plt.subplots(figsize = figSize)
graph = sns.barplot(x = dTypeDistDf['dType'], y = dTypeDistDf['colCount'])


# In[6]:


# *** Check Missing Values ***
missingValsDf = trainDf.isnull().sum().reset_index().                         rename(columns = {'index' : 'colName', 0 : '#MissingValues'})

fig, ax = plt.subplots(figsize = (20, 8))
graph = sns.lineplot(x = 'colName', y = '#MissingValues', data = missingValsDf)
xtks = plt.xticks(rotation = 90)


# In[7]:


# *** Unique Values ***
uniqValsDf = trainDf.nunique().                     reset_index().                     rename(columns = {'index' : 'colName', 0 : '#UniqValues'})

fig, ax = plt.subplots(figsize = (20, 8))
graph = sns.lineplot(x = 'colName', y = '#UniqValues', data = uniqValsDf)
xtks = plt.xticks(rotation = 90)


# In[8]:


targetDistDf = trainDf['target'].describe(percentiles=percentiles).                                  reset_index().                                  rename(columns = {'index' : 'Metric', 'target' : 'Value'})
targetDistDf = targetDistDf.ix[1:, :]

fig, ax = plt.subplots(figsize = figSize)
graph = sns.barplot(x = targetDistDf['Metric'], y = targetDistDf['Value'], color = 'cadetblue')


# In[9]:


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


stop_words = set(stopwords.words('english'))  


# In[10]:


def cleanText(aStentence):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    text = aStentence.lower()
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r",", " commas ", text)
    text = re.sub(r"\.", " fullStop ", text)
    text = re.sub(r"!", " exclamationmark ", text)
    text = re.sub(r"\?", " questionmark ", text)
    text = re.sub(r"'", " singleQoute ", text)
    text = re.sub(r'"', " doubleQoute ", text)
    text = re.sub(r'\n', " newLine ", text)
    text = re.sub("[^A-za-z]"," ", text)
    wordTokens = word_tokenize(text)
    text = ' '.join([CONTRACTION_MAP[aWord] if aWord in CONTRACTION_MAP else aWord for aWord in wordTokens])
    text = ' '.join([aWord for aWord in text.split() if aWord not in stop_words])
    return text


# In[11]:


trainDf['comment_text'] = trainDf['comment_text'].map(lambda x : ' '.join([w.lower() for w in cleanText(x).split() if w!='']))


# In[12]:


trainDf['uniformNumber'] = trainDf['comment_text'].map(lambda x : random.uniform(0, 1))


# In[13]:


train_sentences = list(trainDf[trainDf['uniformNumber'] <= 0.7]['comment_text'])
train_labels = list(trainDf[trainDf['uniformNumber'] <= 0.7]['target'])

validation_sentences = list(trainDf[trainDf['uniformNumber'] > 0.7]['comment_text'])
validation_labels = list(trainDf[trainDf['uniformNumber'] > 0.7]['target'])


# In[14]:


vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
#word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)


# In[15]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[16]:


num_epochs = 3
history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(validation_padded, validation_labels), verbose=2)


# In[17]:


testDf['comment_text'] = testDf['comment_text'].map(lambda x : ' '.join([w.lower() for w in cleanText(x).split() if w!='']))


# In[18]:


test_sequences = list(testDf['comment_text'])
test_sequences = tokenizer.texts_to_sequences(test_sequences)
test_padded = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length)


# In[19]:


test_predictions = model.predict(test_padded).flatten()


# In[20]:


test_predictions


# In[21]:


subDf  = pd.read_csv('../input/sample_submission.csv')


# In[22]:


subDf['prediction'] = pd.Series(test_predictions)


# In[23]:


subDf.head()


# In[24]:


testDf.head()


# In[25]:


subDf.to_csv('submission.csv', index = False)


# In[ ]:




