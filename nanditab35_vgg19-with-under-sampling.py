#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import matplotlib.pyplot as plt
import gc
import scipy.io as sio
#import cv2
#import imutils
from PIL import Image
import tensorflow as tf
import tensorflow.image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("/kaggle/input"))

path = '/kaggle/input/'

# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('env', 'JOBLIB_TEMP_FOLDER=/tmp')
gc.collect()


# In[2]:



df_train_lbl = pd.read_csv(path + 'train.csv')
df_test_lbl = pd.read_csv(path + 'test.csv')

m_tr = np.shape(df_train_lbl)[0]
m_te = np.shape(df_test_lbl)[0]

print(m_tr)

no_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==0])[0])/m_tr)
print(no_dr_ratio

mild_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==1])[0])/m_tr)
print(mild_dr_ratio)

moderate_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==2])[0])/m_tr)
print(moderate_dr_ratio)

severe_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==3])[0])/m_tr)
print(severe_dr_ratio)

proliferative_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==4])[0])/m_tr)
print(proliferative_dr_ratio)


# In[3]:


# Under-sampling of dataset
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

train_path = path + 'train_images/'
test_path = path + 'test_images/'

all_images = glob.glob(train_path + '*.png')

print(np.shape(df_train_lbl))
sampled_train_lbl = pd.DataFrame(columns = df_train_lbl.columns)
rus = RandomUnderSampler(random_state=0)
sampled_train_lbl, y_resampled = rus.fit_resample(df_train_lbl, df_train_lbl['diagnosis'])

print(sorted(Counter(y_resampled).items()))
print(np.shape(sampled_train_lbl))

sampled_m_tr = np.shape(sampled_train_lbl)[0]

gc.collect()


# In[4]:


def image_resize_tf(img_path, image_dim):
    filename = tf.placeholder(tf.string, name="inputFile")
    fileContent = tf.read_file(filename, name="loadFile")
    image = tf.image.decode_png(fileContent, name="decodePng")
    
    resize_nearest_neighbor = tf.image.resize_images(image, size=[image_dim,image_dim], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    sess = tf.Session()
    feed_dict={filename: img_path}
    with sess.as_default():
        actual_resize_nearest_neighbor = resize_nearest_neighbor.eval(feed_dict)
        #plt.imshow(actual_resize_nearest_neighbor)
    return actual_resize_nearest_neighbor


# In[5]:


resized_img = image_resize_tf("../input/train_images/875d2ffcbf47.png", 224)


# In[6]:


# shuffling the data
from random import shuffle

idx_arr = [i for i in range(0,sampled_m_tr)]
shuffle(idx_arr)
m_train_validate = int(sampled_m_tr*0.7)
m_validate = sampled_m_tr - m_train_validate
idx_train = idx_arr[:m_train_validate]
idx_validate = idx_arr[m_train_validate:]


# In[ ]:


# resizing training images
img_arr_train = np.ndarray(shape=(m_train_validate, 224, 224, 3))
lbl_train = np.ndarray(shape=(m_train_validate, 5))
#one_hot_targets = np.eye(nb_classes)[targets]
idx = 0
k = 0
for idx in range(0,np.shape(sampled_train_lbl)[0]):
    if idx in idx_train:
        name = sampled_train_lbl[idx][0] + '.png'
        lbl_train[k,:] = np.eye(5)[int(sampled_train_lbl[idx][1])].T
        img = image_resize_tf(train_path + name, 224)
        #print(img)
        img_arr_train[k,:,:,:] = img
        k += 1
print(np.shape(img_arr_train))
print(np.shape(lbl_train))


# In[7]:


# resizing validating images
img_arr_validate = np.ndarray(shape=(m_validate, 224, 224, 3))
lbl_validate = np.ndarray(shape=(m_validate, 5))
idx = 0
k = 0
for idx in range(0,np.shape(sampled_train_lbl)[0]):
    if idx in idx_validate:
        name = sampled_train_lbl[idx][0] + '.png'
        lbl_validate[k,:] = np.eye(5)[int(sampled_train_lbl[idx][1])].T
        img = image_resize_tf(train_path + name, 224)
        #print(img)
        img_arr_validate[k,:,:,:] = img
        k += 1

print(np.shape(img_arr_validate))
print(np.shape(lbl_validate))


# In[8]:


# vgg19 code
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import Adam

input_shape = (224, 224, 3)

#Instantiate an empty model
model = Sequential([
Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
Conv2D(64, (3, 3), activation='relu', padding='same'),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(128, (3, 3), activation='relu', padding='same'),
Conv2D(128, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Flatten(),
Dense(4096, activation='relu'),
Dense(4096, activation='relu'),
#Dense(1000, activation='relu'),
Dense(5, activation='softmax'),    
])

model.summary()

# Compile the model
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
sgd = SGD(lr=0.0001, momentum=0.9)
#adm = Adam()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=["accuracy"])


# In[9]:


#img_arr_train = img_arr_train/255
#img_arr_validate = img_arr_validate/255

# contering the the image array for training

k = 0
for k in range(0,np.size(img_arr_train,1)):
    img_train = img_arr_train[k,:,:,:]
    img_train_scaled = np.asarray(img_train)
    mean1, std1 = img_train_scaled.mean(), img_train_scaled.std()
    img_train_scaled = (img_train_scaled - mean1)/std1
    img_arr_train[k,:,:,:] = img_train_scaled
    
print(np.shape(img_arr_train))


# In[10]:


# contering the the image array for validating

k = 0
for k in range(0,np.size(img_arr_validate,1)):
    img_validate = img_arr_validate[k,:,:,:]
    img_validate_scaled = np.asarray(img_validate)
    mean1, std1 = img_validate_scaled.mean(), img_validate_scaled.std()
    img_validate_scaled = (img_validate_scaled - mean1)/std1
    img_arr_validate[k,:,:,:] = img_validate_scaled
    
print(np.shape(img_arr_validate))


# In[11]:


#lbls = np.array(sampled_train_lbl['diagnosis']).reshape(sampled_m_tr,1)
history = model.fit(x=img_arr_train,y=lbl_train,validation_data=(img_arr_validate, lbl_validate),batch_size=64,epochs=200,verbose=1) 


# In[12]:


#img_arr_train
del img_arr_train
del img_arr_validate
del df_train_lbl
del df_test_lbl
gc.collect()


# In[13]:


#test_images = glob.glob(test_path + '*.png')
df_sample_sub = pd.read_csv(path + 'sample_submission.csv')
m_test = np.shape(df_sample_sub)[0]
test_images = np.ndarray(shape=(m_test, 224, 224, 3))
k = 0

for row in df_sample_sub.iterrows():
    name = row[1]['id_code'] + '.png'
    img = image_resize_tf(test_path + name, 224)
    img_test_scaled = np.asarray(img)
    mean1, std1 = img_test_scaled.mean(), img_test_scaled.std()
    img_test_scaled = (img_test_scaled - mean1)/std1
    test_images[k,:,:,:] = img_test_scaled
    k += 1


# In[14]:


scores = model.predict_proba(test_images)
y_test = np.argmax(scores,axis=1)
print(y_test)


# In[15]:


df_sample_sub['diagnosis'] = y_test
df_sample_sub['diagnosis'].astype('int64')


# In[16]:


os.chdir("/kaggle/working/")
df_sample_sub.to_csv('submission.csv', index=False)

