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

import os
import cv2
from tqdm import tqdm
from time import time
from collections import Counter
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
dataset_base_path = "../input"
train_data_path = os.path.join(dataset_base_path, 'train')
test_data_path = os.path.join(dataset_base_path, 'test')
TENSORBOARD_LOGS_PATH = './tensorboard_logs'
IMAGE_SIZE = 50
import shutil
# shutil.rmtree("cat_dogs_checkpoints")
os.listdir(".")
# !df -h
os.mkdir("cat_dogs_checkpoints")
# os.mkdir("TENSORBOARD_LOGS_PATH")
# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip ngrok-stable-linux-amd64.zip')
LOG_DIR = TENSORBOARD_LOGS_PATH# Here you have to put your log directory
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# In[3]:


# Load data
# df = pd.read_csv('../input/sampleSubmission.csv')
train_images = os.listdir(train_data_path)
test_images = os.listdir(test_data_path)


# In[4]:


# Prepare train data
def prepare_train_data(train_images_names):
    train_images = list()
    train_labels = list()
    for image_name in tqdm(train_images_names):
        image = cv2.imread(os.path.join(train_data_path, image_name))
        try:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        except:
            print(image_name)
            continue
        label = 1 if image_name.split(".")[0] == "cat" else 0
        train_images.append(image)
        train_labels.append(label)
    train_images = np.stack(train_images)
    train_labels = np.stack(train_labels)
    np.save("train_images.npy", train_images)
    np.save("train_labels.npy", train_labels)
    return train_images, train_labels
        


# In[5]:


# Prepare test data
def prepare_test_data(test_images_names):
    test_images = list()
    test_labels = list()
    for image_name in tqdm(test_images_names):
        image = cv2.imread(os.path.join(test_data_path, image_name))
        try:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        except:
            print(image_name)
            continue
        image_num = image_name.split(".")[0]
        test_images.append(image)
        test_labels.append(image_num)
    test_images = np.stack(test_images)
    test_image_id = np.stack(test_labels)
    np.save("test_images.npy", test_images)
    np.save("test_image_id.npy", test_image_id)
    return test_images, test_images


# In[6]:


prepare_train_data(train_images)
prepare_test_data(test_images)


# In[7]:


dataset_images = np.load('train_images.npy')
dataset_labels = np.load('train_labels.npy')
# np.random.shuffle(train_dataset)
# Generate random indexes for train / val / test split


# In[8]:


print(dataset_images.shape)
dataset_images = dataset_images / 255.0
dataset_image_orig = dataset_images


# In[9]:


NUM_TRAIN_DATA = 21000
X_train = dataset_images[:NUM_TRAIN_DATA, :] # 21000 training samples 
X_val = dataset_images[NUM_TRAIN_DATA:NUM_TRAIN_DATA + 2000, :] # 4000 val samples
X_test = dataset_images[NUM_TRAIN_DATA + 2000:]


# In[10]:


Y_train, Y_val, Y_test = dataset_labels[:NUM_TRAIN_DATA], dataset_labels[NUM_TRAIN_DATA:NUM_TRAIN_DATA + 2000], dataset_labels[NUM_TRAIN_DATA + 2000:]
Y_test.shape


# In[11]:


def show_images_horizontally(images, labels=[], lookup_label=None,
                            figsize=(15, 7)):

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure, imshow, axis
    print(labels[0])
    fig = figure(figsize=figsize)
    for i in range(images.shape[0]):
        fig.add_subplot(1, images.shape[0], i + 1)
        if lookup_label:
            plt.title(lookup_label[labels[i]])
        imshow(images[i], cmap='Greys_r')
        axis('off')


# In[12]:


show_images_horizontally(X_val[:5], Y_val[:5], lookup_label={1:"cat", 0:"dog"})


# In[13]:


model = Sequential()
# Add model layers
model.add(Conv2D(64, (3,3), strides=(1,1),  padding='same', activation='relu', input_shape=(50, 50, 3), name="block1_conv1"))
model.add(Conv2D(64, (3,3), strides=(1,1),  padding='same', activation='relu', name="block1_conv2"))
model.add(MaxPool2D((2,2), padding='same', name='max_pooling1'))
model.add(Conv2D(128, (3,3), strides=(1,1),  padding='same', activation='relu',name="block2_conv1"))
model.add(Conv2D(128, (3,3), strides=(1,1),  padding='same', activation='relu',name="block2_conv2"))
model.add(MaxPool2D((2,2), padding='same', name='max_pooling2'))
model.add(Conv2D(256, (3,3), strides=(1,1),  padding='same', activation='relu',name="block3_conv1"))
model.add(Conv2D(256, (3,3), strides=(1,1),  padding='same', activation='relu',name="block3_conv2"))
model.add(Conv2D(256, (3,3), strides=(1,1),  padding='same', activation='relu',name="block3_conv3"))
model.add(MaxPool2D((2,2), padding='same', name='max_pooling3'))
model.add(Conv2D(512, (3,3), strides=(1,1),  padding='same', activation='relu',name="block4_conv1"))
model.add(Conv2D(512, (3,3), strides=(1,1),  padding='same', activation='relu',name="block4_conv2"))
model.add(Conv2D(512, (3,3), strides=(1,1),  padding='same', activation='relu',name="block4_conv3"))
# model.add(MaxPool2D((2,2), padding='same', name='max_pooling4'))
# model.add(Conv2D(512, (3,3), strides=(1,1),  padding='same', activation='relu',name="block5_conv1"))
# model.add(Conv2D(512, (3,3), strides=(1,1),  padding='same', activation='relu',name="block5_conv2"))
# model.add(Conv2D(512, (3,3), strides=(1,1),  padding='same', activation='relu',name="block5_conv3"))
# model.add(MaxPool2D((2,2), padding='same', name='max_pooling5')) We might not need this max pool
model.add(Flatten(name="Flatten"))
model.add(Dense(128, activation='relu', name="fc1"))
model.add(Dense(128, activation='relu', name="fc2"))
model.add(Dense(1, activation='sigmoid', name="predictions"))
model.summary()


# In[14]:


# tensorboard = TensorBoard(log_dir=TENSORBOARD_LOGS_PATH + "/{}".format(time()))
file_path = 'cat_dogs_checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.compile(optimizer=optimizers.Adamax(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, callbacks=[checkpoint], batch_size=32, shuffle=True)


# In[15]:


model = load_model('cat_dogs_checkpoints/weights-improvement-50-0.88.hdf5')
# !cp cat_dogs_checkpoints/weights-improvement-37-0.88.hdf5 .
# !ls cat_dogs_checkpoints


# In[16]:


get_ipython().system('ls')
res = model.predict(X_test)
res_squeezed = np.squeeze(res)
y_pred = (res_squeezed > 0.5) * 1
sum(Y_test == y_pred) / Y_test.shape[0]


# In[17]:


test_images_sub = np.load('test_images.npy')
test_images_id = np.load('test_image_id.npy')


# In[18]:


# test_images_sub.shape
res = model.predict(test_images_sub)
res_squeezed = np.squeeze(res)
y_pred = (res_squeezed > 0.5) * 1


# In[19]:


import csv
with open('submissions.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(["id", "label"])
    for i, result in enumerate(y_pred): 
        writer.writerow([test_images_id[i], result])


# In[20]:




