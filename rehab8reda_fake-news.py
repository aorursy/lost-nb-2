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


# In[4]:


##import libraries
import tensorflow as tf
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io 
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K 
K.clear_session()


# In[5]:


tf.__version__[0]


# In[54]:


##import data
train=pd.read_csv('/kaggle/input/fake-news/train.csv')
test=pd.read_csv('/kaggle/input/fake-news/test.csv')
submit=pd.read_csv('/kaggle/input/fake-news/submit.csv')


# In[55]:


train.text=train.text.astype(str)
test.text=test.text.astype(str)


# In[56]:


train.head()


# In[58]:


##check null 
## text ,label don't have null values
train.isnull().sum()


# In[59]:


total_data=pd.concat([train,test])
total_data.head()


# In[60]:


## tokenize data
tokenizer= Tokenizer()
tokenizer.fit_on_texts(total_data['text'])
word_index=tokenizer.word_index
vocab_size=len(word_index)
print(vocab_size)


# In[61]:


train_sequences=tokenizer.texts_to_sequences(train['text'])
train_paded_sequences=pad_sequences(train_sequences,maxlen=500,padding='post',truncating='post')

test_sequences=tokenizer.texts_to_sequences(test['text'])
test_paded_sequences=pad_sequences(test_sequences,maxlen=500,padding='post',truncating='post')


# In[79]:


## split to get validation data 
from sklearn.model_selection import train_test_split
train_paded_sequences,valid_paded_sequences,y_train,y_valid=train_test_split(train_paded_sequences,train['label'].values,test_size=.2)


# In[36]:


## get pretrained embedding 
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt     -O /tmp/glove.6B.100d.txt')


# In[62]:


embedding_index={}
with open('/tmp/glove.6B.100d.txt') as f:
    for line in f:
        values=line.split()
        word=values[0]
        coef=np.array(values[1:],dtype='float32')
        embedding_index[word]=coef
        
print(len(coef))
embedding_matrix=np.zeros((vocab_size+1,100))
for word ,i in word_index.items():
    embedding_vector=embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector
        
        
        
        
        


# In[63]:


## build model architecture 
model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size+1,100,weights=[embedding_matrix],trainable=False))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv1D(64, 5, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=4))
model.add(tf.keras.layers.LSTM(20, return_sequences=True))
model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# In[64]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[80]:


history = model.fit(train_paded_sequences, y_train, epochs=5, batch_size=100, validation_data=[valid_paded_sequences,y_valid])


# In[81]:


# Visualize the results:

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[86]:


predictions=model.predict_classes(test_paded_sequences)


# In[87]:


predictions


# In[ ]:




