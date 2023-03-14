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
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, LeakyReLU, Dropout, Dense, Flatten, MaxPool2D
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau


root = '../input'

#read the data im train.file
df_train = pd.read_csv(os.path.join(root, 'train.csv'))
#print(df_train.head(2))

#没有空缺值
df_train_id = df_train.pop('id')
df_train_label = df_train.pop('has_cactus')
df_train_label = to_categorical(df_train_label, num_classes=2)
del df_train

# #先读取一个图片看看情况 图片是一个32*32的结果
# pic_path = os.path.join(root, 'train\\' + df_train_id[0])
# img = load_img(pic_path, grayscale=True)
# img = img_to_array(img).reshape(32, 32)
# print(img.shape)
# plt.imshow(img)
# plt.show()

#读取所有的图片
img_data = np.empty((len(df_train_id), 32, 32, 1))
for index, id in enumerate(df_train_id):
    #print('reading the No.%s image' %index)
    #img = load_img(os.path.join(root, 'train\\'+id))    #0.7537 w/o grayscale
    img = load_img(os.path.join(root, 'train/train/' + id), grayscale=True)
    img = img_to_array(img)
    img_data[index] = img / 255.0

#print(img_data.shape)

#split to train and test data
img_train, img_test, label_train, label_test = train_test_split(img_data, df_train_label, test_size=0.1, random_state=912)

#Build the model
model = Sequential()

model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 1), use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Convolution2D(filters=96, kernel_size=(3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(filters=96, kernel_size=(3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
model.fit(img_train, label_train, batch_size=128, epochs=150, validation_data=(img_test, label_test), verbose=2, callbacks=[learning_rate_reduction])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(img_train, label_train, batch_size=128, epochs=500, validation_data=(img_test, label_test), verbose=2)

# datagen = ImageDataGenerator(
#     rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
# )

# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# model.fit_generator(datagen.flow(img_train, label_train, batch_size=128), steps_per_epoch=img_train.shape[0]/128, epochs=200, validation_data=(img_test, label_test), verbose=2, callbacks=[learning_rate_reduction])


submission_file = pd.read_csv(os.path.join(root, 'sample_submission.csv'))
test_id = submission_file['id']
#读取test的图片
img_data = np.empty((len(test_id), 32, 32, 1))
for index, id in enumerate(test_id):
    #print('reading the No.%s image' %index)
    img = load_img(os.path.join(root, 'test/test/' + id), grayscale=True)
    img = img_to_array(img)
    img_data[index] = img / 255.0

y_pred = np.around(model.predict(img_data))
submission_file['has_cactus'] = y_pred
submission_file.to_csv('submission.csv', index=False)

