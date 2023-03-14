#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.layers import Conv2D, MaxPool2D,      Dropout, Dense, Input, concatenate,          GlobalAveragePooling2D, AveragePooling2D,    Flatten
from keras.optimizers import Adam
from keras.models import Model

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


train_df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
test_df = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')
sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
train_df.head()


# In[3]:


test_df.head()


# In[4]:


sample_sub_df.head()


# In[5]:


import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from tqdm.auto import tqdm
from glob import glob
import time, gc
import cv2


get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# In[6]:


y_grapheme_root=train_df["grapheme_root"]
y_vowel_diacritic=train_df["vowel_diacritic"]
y_cons_diacritic=train_df["consonant_diacritic"]


# In[7]:


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[np.reshape(Y,-1)]
    return Y


# In[8]:


Y_root = convert_to_one_hot(y_grapheme_root, y_grapheme_root.max()+1).T
Y_cons = convert_to_one_hot(y_cons_diacritic,y_cons_diacritic.max()+1).T
Y_vowel = convert_to_one_hot(y_vowel_diacritic, y_vowel_diacritic.max()+1).T


# In[9]:


IMG_SIZE=64
N_CHANNELS=1


# In[10]:


kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)


# In[11]:


def inception_module(X,filters,stage,block):
    conv_name_base = 'inception' + str(stage) + block + '_branch'
    #bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1x1, F3x3_reduce, F3x3, F5x5_reduce, F5x5, F_pool_project  = filters
    
    conv_1x1=Conv2D( F1x1, (1, 1), padding='same', activation='relu', name = conv_name_base + '2a')(X)
    
    conv_3x3_reduce= Conv2D( F3x3_reduce, (1, 1), padding='same', activation='relu')(X)
    conv_3x3= Conv2D( F3x3, (3, 3), padding='same', activation='relu', name = conv_name_base + '2b')(conv_3x3_reduce)
    
    conv_5x5_reduce= Conv2D( F5x5_reduce, (1, 1), padding='same', activation='relu')(X)
    conv_5x5= Conv2D( F5x5, (5, 5), padding='same', activation='relu', name = conv_name_base + '2c')(conv_5x5_reduce)
    
    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(X)
    pool_proj = Conv2D(F_pool_project, (1, 1), padding='same', activation='relu', name = conv_name_base + '2d')(pool_proj)
    
    output=concatenate([conv_1x1,conv_3x3, conv_5x5, pool_proj ],  axis=3)
    print(output.shape)
    
    return output
    


# In[12]:


def GoogleNet(input_shape = (IMG_SIZE, IMG_SIZE, 1), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2')(X_input)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    
    x = inception_module(x,[64,96,128,16,32,32],stage=1,block='a')
    
    x = inception_module(x,[128,128,192,32,96,64],stage=1,block='b')
    
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    
    x = inception_module(x,[192,96,208,16,58,64],stage=1,block='c')
    
    '''
    
    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1_head_root = Dense(classes[0], activation='softmax', name='auxilliary_output_1' + str(classes[0]))(x1)
    x1_head_cons = Dense(classes[1], activation='softmax', name='auxilliary_output_1' + str(classes[1]))(x1)
    x1_head_vowel = Dense(classes[2], activation='softmax', name='auxilliary_output_1' + str(classes[2]))(x1)
    '''
    
    
    x = inception_module(x,[160,112,224,24,64,64],stage=2,block='a')
    x = inception_module(x,[128,128,256,24,64,64],stage=2,block='b')
    x = inception_module(x,[112,144,288,32,64,64],stage=2,block='c')
    
    '''
    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2_head_root = Dense(classes[0], activation='softmax', name='auxilliary_output_2' + str(classes[0]))(x2)
    x2_head_cons = Dense(classes[1], activation='softmax', name='auxilliary_output_2' + str(classes[1]))(x2)
    x2_head_vowel = Dense(classes[2], activation='softmax', name='auxilliary_output_2' + str(classes[2]))(x2)
    
    '''
    
    x = inception_module(x,[256,160,320,32,128,128],stage=3,block='a')
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    x = inception_module(x,[256,160,320,32,128,128],stage=3,block='b')
    x = inception_module(x,[384,192,384,48,128,128],stage=3,block='c')
    
    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = Dropout(0.4)(x)

    head_root = Dense(classes[0], activation='softmax', name='fc' + str(classes[0]))(x)
    head_cons = Dense(classes[1], activation='softmax', name='fc' + str(classes[1]))(x)
    head_vowel = Dense(classes[2], activation='softmax', name='fc' + str(classes[2]))(x)
                     
    
    # Create model
    model = Model(inputs = X_input, outputs = [head_root, head_vowel, head_cons], name='ResNet50')

    return model


# In[13]:


model = GoogleNet(input_shape = (IMG_SIZE, IMG_SIZE, 1), classes = [168,11,7])


# In[14]:


import math 
from keras.optimizers import SGD 

initial_lrate = 0.01
sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)


# In[15]:


model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


# In[16]:


batch_size = 256
epochs = 25


# In[17]:


HEIGHT = 137
WIDTH = 236
SIZE = 128


# In[18]:


Y_root=Y_root.T
Y_cons =Y_cons.T
Y_vowel=Y_vowel.T
def resize(df, size=64, need_progress_bar=True):
    resized = {}
    resize_size=64
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0 
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    else:
        for i in range(df.shape[0]):
            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0 
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized


# In[19]:


model.summary()


# In[20]:


for i in range(4):
    train_df_=pd.DataFrame()
    train_df_= pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df, on='image_id')
    print(train_df_.shape)
    X_train = train_df_.drop(['image_id','grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis=1)
    X_train=resize(X_train)/255
    print(X_train.shape)
    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    print(X_train.shape)
    model.fit(X_train,{'fc168': Y_root[i*50210:(i+1)*50210,:], 'fc11': Y_vowel[i*50210:(i+1)*50210,:], 'fc7': Y_cons[i*50210:(i+1)*50210,:]},batch_size=batch_size,epochs = epochs)


# In[21]:


del train_df,Y_root,Y_cons ,Y_vowel
print(train_df_.shape)
del train_df_


# In[22]:


preds_dict = {  
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[] # model predictions placeholder
row_id=[] # row_id place holder


# In[23]:


for i in range(4):
    test_df_= pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 
    test_df_.set_index('image_id', inplace=True)
    X_test=resize(test_df_)/255
    print(X_test.shape)
    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    print(X_test.shape)
    preds=model.predict(X_test)
    #print(preds)
    for i, p in enumerate(preds_dict):
        preds_dict[p] = np.argmax(preds[i], axis=1)
        
    for k,id in enumerate(test_df_.index.values):  
        for i,comp in enumerate(components):
            id_sample=id+'_'+comp
            row_id.append(id_sample)
            target.append(preds_dict[comp][k])


# In[24]:


df_sample = pd.DataFrame(
    {
        'row_id': row_id,
        'target':target
    },
    columns = ['row_id','target'] 
)
df_sample.to_csv('submission.csv',index=False)
df_sample.head()


# In[ ]:




