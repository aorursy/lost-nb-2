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
'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# Any results you write to the current directory are saved as output.


# In[2]:


import os
from glob import glob
import random
import time
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed

from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import cv2
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files       
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


# In[3]:


dataset = pd.read_csv('../input/driver_imgs_list.csv')
dataset.head(6)


# In[4]:


# Load the dataset previously downloaded from Kaggle
NUMBER_CLASSES = 10

def get_cv2_image(path, img_rows, img_cols, color_type=3):
    # Loading as Grayscale image
    if color_type == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    # Reduce size
    img = cv2.resize(img, (img_rows, img_cols)) 
    return img

# Training
def load_train(img_rows, img_cols, color_type=3):
    train_images = [] 
    train_labels = []
    # Loop over the training folder 
    for classed in tqdm(range(NUMBER_CLASSES)):
        files = glob(os.path.join('..', 'input', 'train', 'c' + str(classed), '*.jpg'))
        for file in files:
            img = get_cv2_image(file, img_rows, img_cols, color_type)
            train_images.append(img)
            train_labels.append(classed)
    return train_images, train_labels 

def read_and_normalize_val_data(img_rows, img_cols, color_type):
    X, labels = load_train(img_rows, img_cols, color_type)
    y = np_utils.to_categorical(labels, 10)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    x_train = np.array(x_train, dtype=np.uint8).reshape(-1,img_rows,img_cols,color_type)
    x_val = np.array(x_val, dtype=np.uint8).reshape(-1,img_rows,img_cols,color_type)
    
    return x_train, x_val, y_train, y_val

# Validation
def load_test(size=200000, img_rows=64, img_cols=64, color_type=3):
    path = os.path.join('..', 'input', 'test', '*.jpg')
    files = sorted(glob(path))
    X_test, X_test_id = [], []
    total = 0
    files_size = len(files)
    for file in tqdm(files):
        if total >= size or total >= files_size:
            break
        file_base = os.path.basename(file)
        img = get_cv2_image(file, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(file_base)
        total += 1
    return X_test, X_test_id

def read_and_normalize_sampled_test_data(size, img_rows, img_cols, color_type=3):
    test_data, test_ids = load_test(size, img_rows, img_cols, color_type)
    
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(-1,img_rows,img_cols,color_type)
    
    return test_data, test_ids


# In[5]:


img_rows = 224
img_cols = 224
color_type = 3


# In[6]:


x_train, x_val, y_train, y_val = read_and_normalize_val_data(img_rows, img_cols, color_type)
print('Train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')


# In[7]:


test_files, test_targets = read_and_normalize_sampled_test_data(200, img_rows, img_cols, color_type)
print('Test shape:', test_files.shape)
print(test_files.shape[0], 'Test samples')


# In[8]:


names = [item[17:19] for item in sorted(glob("../input/train/*/"))]
test_files_size = len(np.array(glob(os.path.join('..', 'input', 'test', '*.jpg'))))

print('There are %s total images.\n' % (test_files_size + len(x_train) + len(x_val)))
print('There are %d training images.' % len(x_train))
print('There are %d total training categories.' % len(names))
print('There are %d validation images.' %len(x_val))
print('There are %d test images.'% test_files_size)


# In[9]:


activity_map = {'c0': 'Safe driving', 
                'c1': 'Texting - right', 
                'c2': 'Talking on the phone - right', 
                'c3': 'Texting - left', 
                'c4': 'Talking on the phone - left', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'}


# In[10]:


from keras.applications import VGG16

conv_base = VGG16(weights="imagenet", include_top=False,input_shape=(224,224,3))
for layer in conv_base.layers:
    layer.trainable = False
        


# In[11]:


def conv_model():      
    x = conv_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x=Dropout(0.5)(x)
    predictions = Dense(10, activation = 'softmax')(x)

    model = Model(input =conv_base.input, output = predictions)
    
    return model


# In[12]:


model = conv_model()

model.summary()

model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])


# In[13]:


train_datagen = ImageDataGenerator(rescale = 1.0/255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True, 
                                   validation_split = 0.2)

test_datagen = ImageDataGenerator(rescale=1.0/ 255, validation_split = 0.2)


# In[14]:


training_generator = train_datagen.flow_from_directory('../input/train', 
                                                 target_size = (img_rows, img_cols), 
                                                 batch_size = 100,
                                                 shuffle=True,
                                                 class_mode='categorical', subset="training")

validation_generator = test_datagen.flow_from_directory('../input/train', 
                                                   target_size = (img_rows, img_cols), 
                                                   batch_size = 100,
                                                   shuffle=False,
                                                   class_mode='categorical', subset="validation")


# In[15]:


history= model.fit_generator(training_generator,
                        steps_per_epoch
                             
                             
                             
                             
                             
                             
                             =100,
                        epochs = 100, 
                        verbose = 1,
                        validation_data = validation_generator,
                        validation_steps = 50)


# In[16]:


import matplotlib.pyplot as plt
history_dict = history.history 
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss'] 
 
epochs = range(1, len(loss_values) + 1) 
 
plt.plot(epochs, loss_values, 'bo', label='Training loss')   
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')   
plt.title('Training and validation loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend() 
 
plt.show()


# In[17]:


acc = history_dict['acc']  
val_acc = history_dict['val_acc'] 
 
plt.plot(epochs, acc, 'bo', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
 
plt.show()


# In[18]:


# Evaluate the performance of the new model
score = model.evaluate_generator(validation_generator, 100)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[19]:


conv_base.trainable = True 
set_trainable = False 
for layer in conv_base.layers: 
    if layer.name == 'block5_conv1':
        set_trainable = True     
    if set_trainable:         
        layer.trainable = True     
    else:
        layer.trainable = False 


# In[20]:


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc']) 
 
history = model.fit_generator( train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50) 


# In[21]:


import matplotlib.pyplot as plt
history_dict = history.history 
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss'] 
 
epochs = range(1, len(loss_values) + 1) 
 
plt.plot(epochs, loss_values, 'bo', label='Training loss')   
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')   
plt.title('Training and validation loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend() 
 
plt.show()


# In[22]:


acc = history_dict['acc']  
val_acc = history_dict['val_acc'] 
 
plt.plot(epochs, acc, 'bo', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
 
plt.show()


# In[23]:


# Evaluate the performance of the new model
score = model.evaluate_generator(validation_generator, 100)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[24]:


model.save("weights.h5")

