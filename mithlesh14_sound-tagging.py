#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import gc 
import time 
from tqdm import tqdm, tqdm_notebook; tqdm.pandas

from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split

import tensorflow as tf 
from keras import backend
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


seed = 1321
np.random.seed(seed)
tf.set_random_seed(seed)


# In[3]:


from pathlib import Path


# In[4]:


data_train_curated =Path('../input/train_curated.csv')
data_test = Path()
data_train_noisy = Path('../input/train_noisy.csv')


# In[5]:


data_train_curated = pd.read_csv('../input/train_curated.csv')
data_test = pd.read_csv('../input/sample_submission.csv')
data_train_noisy = pd.read_csv('../input/train_noisy.csv')


# In[6]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[7]:


data_train_curated.shape


# In[8]:


data_test.sample(10)


# In[9]:


print("Number of class in training data",len(set(data_train_curated.labels)))
print("Number of class in test data", len(set(data_test.columns[1:])))


# In[10]:


data_train_curated.info()


# In[11]:


data_train_curated.head()


# In[12]:


data_train_noisy.info()


# In[13]:


data_train_noisy.head()


# In[14]:


catagory_train = data_train_curated.groupby(['labels']).count()
catagory_train.columns = ['counts']
print(len(catagory_train))


# In[15]:


category_group = data_train_curated.groupby(['labels']).count()
category_group.columns = ['counts']
print(len(category_group))


# In[16]:


plt.figure(figsize = (28,8))
temp = data_train_curated['labels'].value_counts()
x = temp.index
y = temp.values
sns.barplot(x,y)
plt.xlabel('Catagory',color = 'Red',size =15)
plt.ylabel('No of sample', color = 'red', size =15)


# In[17]:


import IPython.display as ipd
sound = ('../input/train_curated/0019ef41.wav')
ipd.Audio(sound)


# In[18]:


from scipy.io import wavfile
rate, data = wavfile.read(sound)
plt.plot(data , '-',color ='r')


# In[19]:


#zooming the wave
plt.figure(figsize=(10,5))
plt.plot(data[:300],'.',color ='b');plt.plot(data[:300],'-',color ='r')


# In[20]:


import wave

data_train_curated['frames'] = data_train_curated['fname'].apply(lambda x: wave.open('../input/train_curated/' + x).getnframes())
data_test['frames'] = data_test['fname'].apply(lambda x: wave.open('../input/test/' + x).getnframes())


# In[21]:


data_train_curated.head()


# In[22]:


plt.figure(figsize = (18,9))
plt.hist(bins= 150  , color ='r', x =data_train_curated['frames'], rwidth= 0.6, );
plt.xlabel('No of frames', color = 'white',size = 20)
plt.ylabel('counts of fname ', color = 'white', size =20)


# In[23]:


data_train_curated[data_train_curated['frames']>2500000]


# In[24]:


sound = ('../input/train_curated/77b925c2.wav')
ipd.Audio(sound)


# In[25]:


from wordcloud import *
wordcloud = WordCloud(width = 1000, height = 600, 
                background_color ='black', 
                min_font_size = 5).generate(''.join(data_train_curated.labels)) 
  
# plot the WordCloud image                        
plt.figure(figsize = (13, 12), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[26]:


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data - 0.5
    


# In[27]:


from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)
from keras.utils import Sequence, to_categorical
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras import backend as K


# In[28]:


class Config(object):
    def __init__(self,
                 sampling_rate=18000, audio_duration=2, 
                 n_classes=len(category_group),
                 use_mfcc=False, n_folds=10, learning_rate=0.0002, 
                 max_epochs=20, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)
    


# In[29]:


#Thanks to keras ;)

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, config,data_dirs,data_dir, list_IDs, labels=None, 
                 batch_size=64, preprocessing_fn=lambda x: x):
        'Initialization'
        self.config = config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# In[30]:


#to get dummy 1D dummy model
def get_dummy(config):
    nclass = config.n_classes
    input_length = config.audio_length
    inp = Input(shape =(input_length,1))
    x = GlobalMaxPool1D()(inp)
    out = Dense(nclass, activation=softmax)(x)
    model = models.Model(inputs = inp, outputs = out)
    optimizer = optimizers.Adam(config.learning_rate)
    model.compile(optimizer = optimizer, loss = losses.categorical_crossentropy,metrics=['Accuracy'])
    return model


# In[31]:


def_conv_model(config):
    nclass = config.n_classes
    input_length = config.audio_length
    inp = Input(shape = (input_length, 1))
    x = Convolution1D(16,9,activation=relu, padding = 'valid')(inp)
    x = Convolution1D(16,9, activation= relu, padding = 'valid')(x)
    x = MaxPool1D(16)(x)
    x =Dropout(rate = 0.1)(x)
    


# In[32]:


Stay Tuned

