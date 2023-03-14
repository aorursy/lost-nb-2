#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_zip_file = '/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip'
test_zip_file = '/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip'


 
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:



df_train = pd.read_csv(train_zip_file,sep='\t')
df_train.head()


# In[3]:



df_test = pd.read_csv(test_zip_file, sep='\t')
df_test.head()


# In[4]:


import string
from nltk.corpus import stopwords


# In[5]:


def cleanup_message(message):
    messsage_pun =  [ char for char in message  if char not in string.punctuation]
    message_punc_join = ''.join(messsage_pun)
    message_punc_join_clean = [ word for word in message_punc_join.split() if word.lower() not in stopwords.words('english')]
    return message_punc_join_clean


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer=cleanup_message)
train_vector = vectorizer.fit_transform(df_train['Phrase'])


# In[7]:


from sklearn.naive_bayes import MultinomialNB


# In[8]:


multinomialnb = MultinomialNB()


# In[9]:


labels = df_train['Sentiment'].values


# In[10]:


multinomialnb.fit(train_vector,labels)


# In[11]:


test_vector = vectorizer.transform(df_test['Phrase'])


# In[12]:


y1 =  multinomialnb.predict(test_vector)


# In[13]:


y1


# In[14]:


np.savetxt("outpu.csv", y1, delimiter=",")


# In[ ]:




