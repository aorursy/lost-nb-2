#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df=pd.read_csv("../input/train.csv")
df.head()


# In[3]:


print("Number of samples: ",len(df))
print("Number of Labels: ",np.unique(df.has_cactus))


# In[4]:


sns.distplot(df.has_cactus)


# In[5]:


from PIL import Image 
from skimage.transform import resize
train=pd.read_csv("../input/train.csv")
train_images=[]
path="../input/train/train/"
for i in train.id:
    image=plt.imread(path+i)
    train_images.append(image)


# In[6]:


train_images=np.asarray(train_images)
X=train_images
y=train.has_cactus
print("Labels: ",y.shape)
print("images: ",X.shape)


# In[7]:


plt.imshow(X[2])


# In[8]:


import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D,     BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam



def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def dense_net(filters, growth_rate, classes, dense_block_size, layers_in_block):
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(classes, activation='softmax')(dense_x)

    return Model(input_img, output)


# In[9]:


from keras.utils import np_utils
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
Cat_test_y = np_utils.to_categorical(y_test)
y_train=np_utils.to_categorical(y_train)

print("X_train shape : ",X_train.shape)
print("y_train shape : ",y_train.shape)
print("X_test shape : ",X_test.shape)
print("y_test shape : ",y_test.shape)


# In[10]:


dense_block_size = 3
layers_in_block = 4

growth_rate = 12
classes = 2
model = dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block)
model.summary()


# training
batch_size = 32
epochs = 10
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
history=model.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, shuffle=True,validation_data=(X_test, Cat_test_y))


# In[11]:


# set the matplotlib backend so figures can be saved in the background
# plot the training loss and accuracy
import sys
import matplotlib
print("Generating plots...")
sys.stdout.flush()
matplotlib.use("Agg")
matplotlib.pyplot.style.use("ggplot")
matplotlib.pyplot.figure()
N = epochs 
matplotlib.pyplot.plot(np.arange(0, N), history.history["loss"], label="train_loss")
matplotlib.pyplot.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
matplotlib.pyplot.plot(np.arange(0, N), history.history["acc"], label="train_acc")
matplotlib.pyplot.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
matplotlib.pyplot.title("Cactus Image Classification")
matplotlib.pyplot.xlabel("Epoch #")
matplotlib.pyplot.ylabel("Loss/Accuracy")
matplotlib.pyplot.legend(loc="lower left")
matplotlib.pyplot.savefig("plot.png")


# In[12]:


from sklearn import metrics
label_pred = model.predict(X_test)

pred = []
for i in range(len(label_pred)):
    pred.append(np.argmax(label_pred[i]))

Y_test = np.argmax(Cat_test_y, axis=1) # Convert one-hot to index

print(metrics.classification_report(Y_test, pred))


# In[13]:


from sklearn import metrics
label_pred = model.predict(X_test)

pred = []
for i in range(len(label_pred)):
    pred.append(np.argmax(label_pred[i]))

Y_test = np.argmax(Cat_test_y, axis=1) # Convert one-hot to index

print(metrics.accuracy_score(Y_test, pred))


# In[14]:


#training all data on this model
model = dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block)
batch_size = 32
epochs = 20
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
y= np_utils.to_categorical(y)
history=model.fit(X,y, epochs=epochs, batch_size=batch_size, shuffle=True,verbose=0)


# In[15]:


from PIL import Image 
from skimage.transform import resize
sample=pd.read_csv("../input/sample_submission.csv")
test_images=[]
path="../input/test/test/"
for i in sample.id:
    image=plt.imread(path+i)
    test_images.append(image)


# In[16]:


# prediction
test_images=np.asarray(test_images)
test_images=test_images.reshape(test_images.shape[0],32,32,3)
pred1=model.predict(test_images)
pred = []
for i in range(len(test_images)):
    pred.append(np.argmax(pred1[i]))


results = pd.DataFrame({"id" : sample.id, "has_cactus": pred})
results.to_csv("submission.csv", index = False)


# In[ ]:





# In[ ]:





# In[ ]:




