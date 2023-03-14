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


print(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/"))


# In[3]:


import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


get_ipython().system('unzip -q ../input/dogs-vs-cats-redux-kernels-edition/train.zip')


# In[5]:


get_ipython().system('unzip -q ../input/dogs-vs-cats-redux-kernels-edition/test.zip')


# In[6]:


filenames = os.listdir("/kaggle/working/test")
for filename in filenames:
    test_df = pd.DataFrame({
    'filename': filenames
})

test_df.index = test_df.index + 1
test_df.head()


# In[7]:


print(os.listdir("/kaggle/working"))


# In[8]:


filenames = os.listdir("/kaggle/working/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
df.head()


# In[9]:


sns.countplot(df['category'])


# In[10]:


df['category'] = df['category'].astype(str)


# In[11]:


train_df, validate_df = train_test_split(df, test_size=0.1)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()


# In[12]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]


# In[13]:


print(total_train)
print(total_validate)


# In[14]:


train_batches = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1) \
    .flow_from_dataframe(
    train_df, 
    "/kaggle/working/train", 
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(224, 224),
    batch_size=124)

valid_batches = ImageDataGenerator(rescale=1./255)     .flow_from_dataframe(
    validate_df, 
    "/kaggle/working/train", 
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(224, 224),
    batch_size=124)

test_batches = ImageDataGenerator(rescale=1./255)     .flow_from_dataframe(
    test_df, 
    "/kaggle/working/test", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    batch_size=124,
    target_size=(224, 224),
    shuffle=False
)


# In[15]:


assert train_batches.n == 22500
assert valid_batches.n == 2500


# In[16]:


imgs, labels = next(train_batches)


# In[17]:


# This function will plot images in the form of a grid with 1 row and 10 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[18]:


plotImages(imgs)
print(labels[0:10])


# In[19]:


model= tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = (224,224,3)),
     tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Dropout(.25),
     tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(1, activation='sigmoid')]
)


# In[20]:


model.summary()


# In[21]:


model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[22]:


model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=2,
          verbose=2
)


# In[23]:


train_batches1 = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)     .flow_from_dataframe(
    train_df, 
    "/kaggle/working/train", 
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(224, 224),
    batch_size=124)

valid_batches1 = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)     .flow_from_dataframe(
    validate_df, 
    "/kaggle/working/train", 
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(224, 224),
    batch_size=124)

test_batches1 = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)     .flow_from_dataframe(
    test_df, 
    "/kaggle/working/test", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    batch_size=124,
    target_size=(224, 224),
    shuffle=False
)


# In[24]:


imgs, labels = next(train_batches1)
plotImages(imgs)
print(labels[0:10])


# In[25]:


vgg16_model = tf.keras.applications.vgg16.VGG16()


# In[26]:


vgg16_model.summary()


# In[27]:


model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)


# In[28]:


model.summary()


# In[29]:


for layer in model.layers:
    layer.trainable = False


# In[30]:


model.add(Dense(units=1, activation='sigmoid'))


# In[31]:


model.summary()


# In[32]:


model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[33]:


model.fit(x=train_batches1,
          steps_per_epoch=len(train_batches1),
          validation_data=valid_batches1,
          validation_steps=len(valid_batches1),
          epochs=3,
          verbose=2)


# In[34]:


results = model.predict(test_batches)


# In[35]:


test_df['category'] = np.where(results > 0.5, 1,0)


# In[36]:


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)

