#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import random # random numbers
import shutil # zip a folder

import numpy as np # linear algebra
import zipfile # unzip files
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm
from datetime import datetime
from IPython.display import FileLink

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# In[3]:


from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, sparse_categorical_accuracy, SparseCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[4]:


ls ../input/dogs-vs-cats-redux-kernels-edition/


# In[5]:


zip_ref = zipfile.ZipFile("../input/dogs-vs-cats-redux-kernels-edition/train.zip", "r")
filelist = zip_ref.filelist[1:]
print(f'{zip_ref.filelist[0].filename} is removed')
random.shuffle(filelist)


# In[6]:


len(filelist)


# In[7]:


count_cats = 0
count_dogs = 0
trainN = int(len(filelist)*0.9)
valN = int(len(filelist)*0.1)

for file_path in tqdm(filelist[0:trainN]):
    if file_path.filename.endswith(".jpg"):
        file_name = file_path.filename[6:]
        if file_name.startswith("cat"):
            zip_ref.extract(file_path.filename, path='train/cats/'+file_name)
            count_cats += 1
        elif file_name.startswith("dog"):
            zip_ref.extract(file_path.filename, path='train/dogs/'+file_name)
            count_dogs += 1

for file_path in tqdm(filelist[trainN:]):
    if file_path.filename.endswith(".jpg"):
        file_name = file_path.filename[6:]
        if file_name.startswith("cat"):
            zip_ref.extract(file_path.filename, path='valid/cats/'+file_name)
            count_cats += 1
        elif file_name.startswith("dog"):
            zip_ref.extract(file_path.filename, path='valid/dogs/'+file_name)
            count_dogs += 1


# In[8]:


get_ipython().system('ls train/cats/ | wc -l')
get_ipython().system('ls train/dogs/ | wc -l')
get_ipython().system('ls valid/cats/ | wc -l')
get_ipython().system('ls valid/dogs/ | wc -l')


# In[9]:


def get_shallow_cnn(input_shape=(28, 28, 1)):
  model = Sequential()
  model.add(Conv2D(16, 3, activation=relu, input_shape=input_shape))
  model.add(MaxPool2D())
  model.add(Conv2D(32, 3, activation=relu))
  model.add(MaxPool2D())
  model.add(Flatten())
  model.add(Dense(128, activation=relu))
  model.add(Dense(2, activation='softmax'))
  return model


# In[10]:


def get_shallow_cnn_regu(input_shape=(28, 28, 1), decay_rate=1e-3, dropout=0.1):
    model = Sequential()
    model.add(Conv2D(16, 3, activation=relu, input_shape=input_shape,
                     kernel_regularizer=l2(decay_rate)))
    model.add(Dropout(dropout))
    model.add(MaxPool2D())
    model.add(Conv2D(32, 3, activation=relu, kernel_regularizer=l2(decay_rate)))
    model.add(Dropout(dropout))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(128, activation=relu, kernel_regularizer=l2(decay_rate)))
    model.add(Dense(2, activation='softmax'))
    return model


# In[11]:


def get_shallow2_cnn(input_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(16, 3, activation=relu, input_shape=input_shape))
    model.add(MaxPool2D())
    model.add(Conv2D(32, 3, activation=relu))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, activation=relu))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, activation=relu))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(128, activation=relu))
    model.add(Dense(2, activation='softmax'))
    return model


# In[12]:


def get_shallow2_cnn_regu(input_shape=(28, 28, 1), decay_rate=1e-3, dropout=0.1):
    model = Sequential()
    model.add(Conv2D(16, 3, activation=relu, input_shape=input_shape,
                     kernel_regularizer=l2(decay_rate)))
    model.add(Dropout(dropout))
    model.add(MaxPool2D())
    model.add(Conv2D(32, 3, activation=relu, kernel_regularizer=l2(decay_rate)))
    model.add(Dropout(dropout))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, activation=relu, kernel_regularizer=l2(decay_rate)))
    model.add(Dropout(dropout))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, activation=relu, kernel_regularizer=l2(decay_rate)))
    model.add(Dropout(dropout))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(128, activation=relu))
    model.add(Dense(2, activation='softmax'))
    return model


# In[13]:


batch_size = 32
image_gen = ImageDataGenerator(rescale=1/255.0)
image_gen_train = image_gen.flow_from_directory("train", target_size=(64, 64),
                                                batch_size=batch_size, shuffle=True)
image_gen_valid = image_gen.flow_from_directory("valid", target_size=(64, 64),
                                                batch_size=batch_size, shuffle=True)


# In[14]:


batch_size = 32
image_gen_augument = ImageDataGenerator(rescale=1/255.0, rotation_range=10,
                                       horizontal_flip=True, zoom_range=0.2)
image_gen_augument_train = image_gen_augument.flow_from_directory("train", target_size=(64, 64),
                                                batch_size=batch_size, shuffle=True)
image_gen_augument_valid = image_gen_augument.flow_from_directory("valid", target_size=(64, 64),
                                                batch_size=batch_size, shuffle=True)


# In[15]:


for image1, image2 in zip(image_gen_train, image_gen_augument_train):
    plt.imshow(image1[0][0])
    plt.figure()
    plt.imshow(image2[0][0])
    break


# In[16]:


steps_per_epoch = image_gen_train.n//batch_size
validation_steps = image_gen_valid.n//batch_size


# In[17]:


print(steps_per_epoch, validation_steps)


# In[18]:


model1 = get_shallow_cnn((64, 64, 3))
model1.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                      'AUC'])


# In[19]:


log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history1 = model1.fit(image_gen_train, validation_data=image_gen_valid,
               steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback],
               epochs=10, validation_steps=validation_steps)


# In[20]:


model1.save("shallow_cnn")
shutil.make_archive("shallow_cnn", 'zip', "shallow_cnn")
FileLink('shallow_cnn.zip')


# In[21]:


train_loss_results = history1.history['loss']
valid_loss_results = history1.history['val_loss']
train_accuracy_results = history1.history['binary_accuracy']
valid_accuracy_results = history1.history['val_binary_accuracy']
train_auc_results = history1.history['auc']
valid_auc_results = history1.history['val_auc']


# In[22]:


plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, sharex=True, figsize=(18, 5))

axes[0].set_xlabel("Epochs", fontsize=14)
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].set_title('Loss vs epochs')
axes[0].plot(train_loss_results, label='train_loss')
axes[0].plot(valid_loss_results, label='valid_loss')
axes[0].legend()

axes[1].set_title('Accuracy vs epochs')
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epochs", fontsize=14)
axes[1].plot(train_accuracy_results, label='train_accuracy')
axes[1].plot(valid_accuracy_results, label='valid_accuracy')
axes[1].legend()

axes[2].set_title('AUC vs epochs')
axes[2].set_ylabel("AUC", fontsize=14)
axes[2].set_xlabel("Epochs", fontsize=14)
axes[2].plot(train_accuracy_results, label='train_auc')
axes[2].plot(valid_accuracy_results, label='valid_auc')
axes[2].legend()

plt.show()


# In[23]:


model2 = get_shallow_cnn_regu((64, 64, 3), decay_rate=1e-3, dropout=0.1)
model2.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                      'AUC'])


# In[24]:


log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history2 = model2.fit(image_gen_train, validation_data=image_gen_valid,
               steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback],
               epochs=30, validation_steps=validation_steps)


# In[25]:


model2.save("shallow_cnn_reg")
shutil.make_archive("shallow_cnn_reg", 'zip', "shallow_cnn_reg")
FileLink('shallow_cnn_reg.zip')


# In[26]:


get_ipython().system('ls shallow_cnn_reg')


# In[27]:


train_loss_results = history2.history['loss']
valid_loss_results = history2.history['val_loss']
train_accuracy_results = history2.history['binary_accuracy']
valid_accuracy_results = history2.history['val_binary_accuracy']
train_auc_results = history2.history['auc']
valid_auc_results = history2.history['val_auc']


# In[28]:


plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, sharex=True, figsize=(18, 5))

axes[0].set_xlabel("Epochs", fontsize=14)
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].set_title('Loss vs epochs')
axes[0].plot(train_loss_results, label='train_loss')
axes[0].plot(valid_loss_results, label='valid_loss')
axes[0].legend()

axes[1].set_title('Accuracy vs epochs')
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epochs", fontsize=14)
axes[1].plot(train_accuracy_results, label='train_accuracy')
axes[1].plot(valid_accuracy_results, label='valid_accuracy')
axes[1].legend()

axes[2].set_title('AUC vs epochs')
axes[2].set_ylabel("AUC", fontsize=14)
axes[2].set_xlabel("Epochs", fontsize=14)
axes[2].plot(train_accuracy_results, label='train_auc')
axes[2].plot(valid_accuracy_results, label='valid_auc')
axes[2].legend()

plt.show()


# In[29]:


model3 = get_shallow2_cnn_regu((64, 64, 3), decay_rate=1e-3, dropout=0.1)
model3.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                      'AUC'])


# In[30]:


log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history3 = model3.fit(image_gen_train, validation_data=image_gen_valid,
               steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback],
               epochs=30, validation_steps=validation_steps)


# In[31]:


model3.save("shallow2_cnn_reg")
shutil.make_archive("shallow2_cnn_reg", 'zip', "shallow2_cnn_reg")
FileLink('shallow2_cnn_reg.zip')


# In[32]:


train_loss_results = history3.history['loss']
valid_loss_results = history3.history['val_loss']
train_accuracy_results = history3.history['binary_accuracy']
valid_accuracy_results = history3.history['val_binary_accuracy']
train_auc_results = history3.history['auc']
valid_auc_results = history3.history['val_auc']


# In[33]:


plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, sharex=True, figsize=(18, 5))

axes[0].set_xlabel("Epochs", fontsize=14)
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].set_title('Loss vs epochs')
axes[0].plot(train_loss_results, label='train_loss')
axes[0].plot(valid_loss_results, label='valid_loss')
axes[0].legend()

axes[1].set_title('Accuracy vs epochs')
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epochs", fontsize=14)
axes[1].plot(train_accuracy_results, label='train_accuracy')
axes[1].plot(valid_accuracy_results, label='valid_accuracy')
axes[1].legend()

axes[2].set_title('AUC vs epochs')
axes[2].set_ylabel("AUC", fontsize=14)
axes[2].set_xlabel("Epochs", fontsize=14)
axes[2].plot(train_accuracy_results, label='train_auc')
axes[2].plot(valid_accuracy_results, label='valid_auc')
axes[2].legend()

plt.show()


# In[34]:


model4 = get_shallow2_cnn_regu((64, 64, 3), decay_rate=1e-3, dropout=0.1)
model4.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                      'AUC'])


# In[35]:


log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history4 = model4.fit(image_gen_augument_train, validation_data=image_gen_augument_valid,
               steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback],
               epochs=30, validation_steps=validation_steps)


# In[36]:


model4.save("shallow2_cnn_reg_aug")
shutil.make_archive("shallow2_cnn_reg_aug", 'zip', "shallow2_cnn_reg_aug")
FileLink('shallow2_cnn_reg_aug.zip')


# In[37]:


train_loss_results = history4.history['loss']
valid_loss_results = history4.history['val_loss']
train_accuracy_results = history4.history['binary_accuracy']
valid_accuracy_results = history4.history['val_binary_accuracy']
train_auc_results = history4.history['auc']
valid_auc_results = history4.history['val_auc']


# In[38]:


plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, sharex=True, figsize=(18, 5))

axes[0].set_xlabel("Epochs", fontsize=14)
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].set_title('Loss vs epochs')
axes[0].plot(train_loss_results, label='train_loss')
axes[0].plot(valid_loss_results, label='valid_loss')
axes[0].legend()

axes[1].set_title('Accuracy vs epochs')
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epochs", fontsize=14)
axes[1].plot(train_accuracy_results, label='train_accuracy')
axes[1].plot(valid_accuracy_results, label='valid_accuracy')
axes[1].legend()

axes[2].set_title('AUC vs epochs')
axes[2].set_ylabel("AUC", fontsize=14)
axes[2].set_xlabel("Epochs", fontsize=14)
axes[2].plot(train_accuracy_results, label='train_auc')
axes[2].plot(valid_accuracy_results, label='valid_auc')
axes[2].legend()

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




