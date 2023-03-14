#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('V1')


# In[2]:


import os
import shutil
from tensorflow import keras
import cv2
import random
import numpy as np
#import seaborn as sns
#%matplotlib inline 
#import matplotlib.pyplot as plt
#from matplotlib import ticker
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd



# Global variables
NUM_CLASSES = 2
TRAIN_DIR_ORIGIN = '../input/train'
TRAIN_DIR = './train_data'
TEST_DIR_ORIGIN = '../input/test'
TEST_DIR = './test_data'
CHANNELS = 3

# Global hyperparameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_EPOCHS = 5
STEPS_PER_EPOCH_TRAINING = 25
STEPS_PER_EPOCH_VALIDATION = 125
RESNET50_POOLING = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
MODEL_OPTIMIZER = 'sgd'
LOSS_FCT = 'categorical_crossentropy'
METRICS = ['accuracy']
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100
BATCH_SIZE_TESTING = 1 # More would make the submission needing to be formatted
VALIDATION_SPLIT = 0.25
DROP_IMAGES = [dog.11731.jpg,
dog.4334.jpg,
cat.4688.jpg,
cat.11222.jpg,
cat.1450.jpg,
cat.2159.jpg,
cat.3822.jpg,
cat.4104.jpg,
cat.5355.jpg,
cat.7194.jpg,
cat.7920.jpg,
cat.9250.jpg,
cat.9444.jpg,
cat.9882.jpg,
dog.11538.jpg,
dog.11724.jpg,
dog.8507.jpg,
cat.2939.jpg,
cat.3216.jpg,
cat.4833.jpg,
cat.7968.jpg,
cat.8470.jpg,
dog.10161.jpg,
dog.10190.jpg,
dog.11186.jpg,
dog.1308.jpg,
dog.1895.jpg,
dog.9188.jpg,
cat.5351.jpg,
cat.5418.jpg,
cat.9171.jpg,
dog.10747.jpg,
dog.2614.jpg,
dog.4367.jpg,
dog.8736.jpg,
cat.7377.jpg,
dog.12376.jpg,
dog.1773.jpg,
cat.10712.jpg,
cat.11184.jpg,
cat.7564.jpg,
cat.8456.jpg,
dog.10237.jpg,
dog.1043.jpg,
dog.1194.jpg,
dog.5604.jpg,
dog.9517.jpg,
cat.11565.jpg,
dog.10797.jpg,
dog.2877.jpg,
dog.8898.jpg]


# In[3]:


os.makedirs(os.path.join(TRAIN_DIR, 'dogs'), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'cats'), exist_ok=True)

try:
    os.listdir(TRAIN_DIR_ORIGIN)
except FileNotFoundError:
    train_dir_tmp = './tmp'
    os.makedirs(train_dir_tmp, exist_ok=True)
    shutil.unpack_archive(TRAIN_DIR_ORIGIN + '.zip', train_dir_tmp, 'zip')
    train_dir_tmp = os.path.join(train_dir_tmp, os.path.basename(TRAIN_DIR_ORIGIN))
    print('Unpacked train set')
else:
    train_dir_tmp = TRAIN_DIR_ORIGIN
    print('There was no need to unpack train set')
    
print("Copying the training images...")
for i, f in enumerate(os.listdir(train_dir_tmp)):
    if 'cat' in f:
        shutil.copy(os.path.join(train_dir_tmp, f), os.path.join(TRAIN_DIR, 'cats', f))
    elif 'dog' in f:
        shutil.copy(os.path.join(train_dir_tmp, f), os.path.join(TRAIN_DIR, 'dogs', f))
    if (i + 1) % 1000 == 0:
        print('Copied', i + 1, 'train images')


def pad(s):
    return s.split('.')[0].zfill(5) + '.' + s.split('.')[1]
        
os.makedirs(TEST_DIR, exist_ok=True)
try:
    os.listdir(TEST_DIR_ORIGIN)
except FileNotFoundError:
    shutil.unpack_archive(TEST_DIR_ORIGIN + '.zip', TEST_DIR, 'zip')
    print('Unpacked test set')
    test_img_dir = os.path.join(TEST_DIR, os.listdir(TEST_DIR)[0])
    for i, f in enumerate(os.listdir(test_img_dir)):
        os.rename(os.path.join(test_img_dir, f), os.path.join(test_img_dir, pad(f)))
        if (i + 1) % 1000 == 0:
            print('Renamed', i + 1, 'test images')
else:
    print('There was no need to unpack test set')
    for i, f in enumerate(os.listdir(TEST_DIR_ORIGIN)):
        shutil.copy(os.path.join(TEST_DIR_ORIGIN, f), os.path.join(TEST_DIR, 'all', pad(f)))
        if (i + 1) % 1000 == 0:
            print('Copied', i + 1, 'test images')



# In[4]:


train_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     validation_split=VALIDATION_SPLIT)

train_generator = train_image_gen.flow_from_directory(TRAIN_DIR,
                                                      target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                      batch_size=BATCH_SIZE_TRAINING,
                                                      seed=0,
                                                      subset='training',
                                                      class_mode='categorical',
                                                      shuffle=True)

val_generator = train_image_gen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE_VALIDATION,
                                                    seed=0,
                                                    subset='validation',
                                                    class_mode='categorical',
                                                    shuffle=True)

test_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_image_gen.flow_from_directory(TEST_DIR,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE_TESTING,
                                                    class_mode=None,
                                                    seed=0,
                                                    shuffle=False)


# In[5]:


model = Sequential()
model.add(ResNet50(include_top=False, pooling=RESNET50_POOLING, weights='imagenet'))
model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))

# Indicate whether the first layer should be trained/changed or not.
model.layers[0].trainable = False

model.compile(optimizer=MODEL_OPTIMIZER,
              loss=LOSS_FCT, 
              metrics=METRICS)


# In[6]:


model.fit_generator(train_generator,
                    steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
                    epochs=NUM_EPOCHS,
                    validation_data=val_generator,
                    validation_steps=STEPS_PER_EPOCH_VALIDATION)


# In[7]:


model.save_weights('model_weights.h5')
model.save('model_keras.h5')


# In[8]:


predictions = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)


# In[9]:


pred_labs = list(map(lambda x: x[1] / (x[0] + x[1]), predictions))


# In[10]:


counter = range(1, len(test_generator) + 1)
solution = pd.DataFrame({"id": counter, "label": pred_labs})
solution.to_csv("submission.csv", index = False)

