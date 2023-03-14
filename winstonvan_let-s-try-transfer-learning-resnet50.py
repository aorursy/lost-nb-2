#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D


# In[ ]:


train = pd.read_json('../input/dont-call-me-turkey/train.json')
test = pd.read_json('../input/dont-call-me-turkey/test.json')


# In[ ]:


num_classes = 2
resnet_weights_path = '../input/resnetweights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' # MANUALLY ADD


# In[ ]:


print(os.listdir("../input/resnetweights"))


# In[ ]:


model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = False

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


from keras.preprocessing.sequence import TimeseriesGenerator

data_generator = TimeseriesGenerator(data=trainx, targets=50, length=10)


# In[ ]:


train_generator = data_generator.

