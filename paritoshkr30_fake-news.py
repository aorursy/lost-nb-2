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


df = pd.read_csv("../input/fake-news/train.csv")


# In[3]:


df.head()


# In[4]:


df = df.dropna()


# In[5]:


X = df.drop('label' , axis =1)


# In[6]:


y = df['label']


# In[7]:


X.shape


# In[8]:


import tensorflow as tf


# In[9]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


# In[10]:


voc_size = 5000


# In[11]:


messages  = X.copy()


# In[12]:


messages['title'][1]


# In[13]:


messages.reset_index(inplace=True)


# In[14]:


import nltk 
import re
from nltk.corpus import stopwords


# In[15]:


nltk.download('stopwords')


# In[16]:


### Dataset Preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[17]:


corpus


# In[18]:


onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr


# In[19]:


sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[20]:


embedded_docs[0]


# In[21]:


## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[22]:


len(embedded_docs),y.shape


# In[23]:


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


# In[24]:


X_final.shape,y_final.shape


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


# In[26]:


### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=64)


# In[27]:


from tensorflow.keras.layers import Dropout
## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[28]:


y_pred=model.predict_classes(X_test)


# In[29]:


from sklearn.metrics import confusion_matrix


# In[30]:


confusion_matrix(y_test,y_pred)


# In[31]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




