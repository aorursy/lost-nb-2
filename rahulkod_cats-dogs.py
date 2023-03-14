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


try:
    get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass


# In[3]:


import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[4]:


tf.__version__, keras.__version__


# In[5]:


print(os.listdir('../input/dogs-vs-cats/'))


# In[6]:


from zipfile import ZipFile
with ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zf:
    zf.extractall('destination_path/')


# In[7]:


# print(os.listdir('./destination_path/train'))


# In[8]:


filenames = os.listdir('./destination_path/train')
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)
        
# laod data into a dataframe
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# In[9]:


df[:10]


# In[10]:


df.info()


# In[11]:


import pathlib
import random

from tensorflow.keras.preprocessing.image import load_img


# In[12]:


plt.figure(figsize=(12, 12))
for i in range(0, 9):
    plt.subplot(3, 3, i+1)
    sample_image = random.choice(filenames)
    image = load_img("./destination_path/train/"+sample_image)
    plt.imshow(image)
    plt.title(sample_image)
    plt.axis('off')


# In[13]:


df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 


# In[14]:


new_df = df.groupby('category').apply(lambda x: x.sample(2000)).reset_index(drop=True)


# In[15]:


new_df.info()


# In[16]:


new_df['category'].value_counts()


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


train_df, val_df = train_test_split(new_df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


# In[19]:


# imports

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Dropout, Dense, Flatten, MaxPooling2D 
from tensorflow.keras.layers import BatchNormalization


# In[20]:


# constants

image_width, image_height = 150, 150
image_channels = 3
image_size = (image_width, image_height)


# In[21]:


# Define model 

model = Sequential([
    Conv2D(32, 3, use_bias=False, input_shape = (image_width, image_height, image_channels)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
    
    Conv2D(32, 3, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
    
    Conv2D(64, 3, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
    
    # fcn
    Flatten(),
    Dense(256, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, use_bias=False),
    Activation('relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# In[22]:


# compile model

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])


# In[23]:


model.summary()


# In[24]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[25]:


early_stop = EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001, verbose=1)

callbacks = [early_stop, reduce_lr]


# In[26]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[27]:


batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   shear_range=0.1,
                                   horizontal_flip=True,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)

val_datagen = ImageDataGenerator(rescale=1./255)


# In[28]:


# flow from Dataframe

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    './destination_path/train/',
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=image_size,
                                                    class_mode='binary',
                                                    batch_size=batch_size)


# In[29]:


valid_generator = val_datagen.flow_from_dataframe(val_df,
                                                 './destination_path/train/',
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=image_size,
                                                    class_mode='binary',
                                                  batch_size=batch_size)


# In[30]:


example_df = train_df.sample(n=1).reset_index(drop = True)

example_generator = train_datagen.flow_from_dataframe(example_df,
                                                       './destination_path/train/',
                                                       x_col = 'filename',
                                                       y_col = 'category',
                                                       target_size = image_size,
                                                       class_mode = 'categorical')


# In[31]:


plt.figure(figsize=(12, 12))
for i in range(0, 9):
    plt.subplot(3, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[32]:


total_train = train_df.shape[0]
print(total_train)
total_val = val_df.shape[0]
print(total_val)


# In[33]:


device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))


# In[34]:


tf.keras.backend.clear_session()


# In[35]:


epochs = 30

history = model.fit_generator(train_generator,
                              epochs=epochs,
                              validation_data=valid_generator,
                              validation_steps=total_val//batch_size,
                              steps_per_epoch=total_train//batch_size,
                              callbacks=callbacks)


# In[36]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc = 'upper right')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1.0])
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
# plt.ylim([0.5,1.0])
plt.title('Training and Validation Loss')


# In[37]:


loss, accuracy = model.evaluate_generator(valid_generator, steps=total_val//batch_size)


# In[38]:


loss, accuracy


# In[39]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D


# In[40]:


base_model = VGG16(weights='imagenet', input_shape = (image_width, image_height, image_channels),
                   include_top = False)


# In[41]:


# freeze vgg16

len(base_model.layers)


# In[42]:


base_model.summary()


# In[43]:


# freeze the first 15 layers

for layer in base_model.layers[:15]:
    layer.trainable = False
    
for layer in base_model.layers[15:]:
    layer.trainable = True

base_model.summary()


# In[44]:


# Add a classification head

new_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])


# In[45]:


new_model.compile(loss='binary_crossentropy',
             optimizer = 'rmsprop',
             metrics = ['accuracy'])


# In[46]:


new_model.summary()


# In[47]:


history_new = new_model.fit_generator(train_generator,
                              epochs=30,
                              validation_data=valid_generator,
                              validation_steps=total_val//batch_size,
                              steps_per_epoch=total_train//batch_size,
                              callbacks=callbacks)


# In[48]:


vgg_loss, vgg_accuracy = new_model.evaluate_generator(valid_generator, steps=total_val//batch_size)

vgg_loss, vgg_accuracy


# In[49]:


acc = history_new.history['accuracy']
val_acc = history_new.history['val_accuracy']

loss = history_new.history['loss']
val_loss = history_new.history['val_loss']

plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')


# In[ ]:




