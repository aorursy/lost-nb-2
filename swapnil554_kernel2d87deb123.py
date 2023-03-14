#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


from zipfile import ZipFile
# Create a ZipFile Object and load sample.zip in it
# Create a ZipFile Object and load sample.zip in it
with ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zipObj:
   # Extract all the contents of zip file in different directory
   zipObj.extractall('temp')


# In[3]:


with ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as zipObj:
   # Extract all the contents of zip file in different directory
   zipObj.extractall('temp')


# In[4]:


categories = []
import os
for dirname, _, filenames in os.walk('temp'):
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        if category == 'cat':
            categories.append(0)


# In[5]:


import cv2
import glob


# In[6]:


import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# In[7]:


from PIL import Image


# In[8]:


dim = (150,150)
filelist_trainx = sorted(glob.glob('temp/train/*.jpg'), key=numericalSort)
#filelist_trainx.sort()
x_train = np.array([np.array(cv2.resize(cv2.imread(fname), dim, interpolation = cv2.INTER_AREA)) for fname in filelist_trainx])


# In[9]:


dim = (150,150)
filelist_testx = sorted(glob.glob('temp/test1/*.jpg'), key=numericalSort)
#filelist_trainx.sort()
x_test = np.array([np.array(cv2.resize(cv2.imread(fname), dim, interpolation = cv2.INTER_AREA)) for fname in filelist_testx])


# In[10]:


x_test.shape


# In[11]:


test_categories = []
import os
for dirname, _, filenames in os.walk('temp/test1'):
    for filename in filenames:
        print(filename)
        category = filename.split('.')[0]
        test_categories.append(category)


# In[12]:


np.unique(test_categories)


# In[13]:


import tensorflow as tf
from keras.layers import Conv2D,Dense,Dropout
from keras.applications import InceptionV3


# In[14]:


get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
  

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


# In[15]:


pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)


# In[16]:


img_input = pre_trained_model.input


# In[17]:


pre_trained_model.load_weights(local_weights_file)


# In[18]:


for layer in pre_trained_model.layers:
  layer.trainable = False
  


# In[19]:


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# In[20]:


from keras.optimizers import Adam
from keras.models import Model
import keras
# Flatten the output layer to 1 dimension
x = keras.layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = keras.layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = keras.layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
preds = keras.layers.Dense(1, activation='sigmoid')(x)           

model = Model(img_input,preds) 

model.compile(optimizer = Adam(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])


# In[21]:


y_train = categories


# In[22]:


model.fit(x_train,y_train,epochs=30,batch_size=1000)


# In[23]:


preds = model.predict(x_test)
model.save_weights("model.h5")


# In[24]:


predictions = []
for i in range(preds.shape[0]):
    if(preds[i]<=0.5):
        predictions.append(0)
    else:
        predictions.append(1)
    


# In[25]:


df = pd.DataFrame()
df["id"] = test_categories
df["label"] = predictions


# In[26]:


df.to_csv('submission.csv',index=False)

