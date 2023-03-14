#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train_dataset=pd.read_csv('../input/Kannada-MNIST/train.csv')
test_dataset=pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[3]:


print("Train Dataset",train_dataset.shape)
print("Test Dataset", test_dataset.shape)


# In[4]:


train_dataset.head()


# In[5]:


test_dataset.head()


# In[6]:


x = train_dataset.iloc[:, 1:].values.astype('float32') / 255
y = train_dataset.iloc[:, 0] # labels


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.05, random_state=1234) 


# In[8]:


X_train.shape


# In[9]:


X_train=X_train.reshape(-1,28,28,1)
X_test= X_test.reshape(-1,28,28,1)


# In[10]:


print("X_train_shape ", X_train.shape)
print("X_test_shape ", X_test.shape)
print("Y_train_shape ", Y_train.shape)
print("Y_test_shape ", Y_test.shape)


# In[11]:


plt.imshow(X_train[i][:,:,0])
print("Y_train ", Y_train)


# In[12]:


w_grd = 15
L_grd =15

fig , axes = plt.subplots(L_grd, w_grd, figsize = (25, 25))
axes = axes.ravel()

n_train = len(X_train)

for i in np.arange(0, L_grd * w_grd):
    index = np.random.randint(0, n_train)
    axes[i].imshow(X_train[index][:,:,0])
    axes[i].axis('off')


# In[13]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[14]:


import keras
Y_train = keras.utils.to_categorical(Y_train)
Y_test = keras.utils.to_categorical(Y_test)


# In[15]:


#X_train = X_train /255
#X_test = X_test /255


# In[16]:


Input_shape = X_train.shape[1:]
Input_shape


# In[17]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# In[18]:


cnn_model = Sequential()
cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5),
                     activation = 'relu', input_shape= Input_shape))
cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5),
                     activation = 'relu'))
cnn_model.add(MaxPooling2D(2, 2))
cnn_model.add(Dropout(0.3))


#Another CNN with 64 , 64
cnn_model.add(Conv2D(filters = 64, kernel_size = (5,5),
                     activation = 'relu'))
cnn_model.add(Conv2D(filters = 64, kernel_size = (5,5),
                     activation = 'relu'))
cnn_model.add(MaxPooling2D(2, 2))
cnn_model.add(Dropout(0.2))

#Flatten
cnn_model.add(Flatten())

#Dense
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = 256, activation = 'relu'))

cnn_model.add(Dense(units = 10, activation = 'softmax'))


# In[19]:


cnn_model.compile(loss = 'categorical_crossentropy', optimizer= keras.optimizers.rmsprop(lr = 0.001), metrics = ['accuracy'])


# In[20]:


history = cnn_model.fit(X_train, Y_train, batch_size= 64, epochs= 2, shuffle=True)


# In[21]:


evalution = cnn_model.evaluate(X_test, Y_test)
print("Accuracy ", evalution[1])


# In[22]:


predicted_classes = cnn_model.predict_classes(X_test)
predicted_classes
#Y_test = Y_test.argmax(1)
Y_test


# In[23]:


y_pre_test=cnn_model.predict(X_test)
y_pre_test=np.argmax(y_pre_test,axis=1)
Y_test=np.argmax(Y_test,axis=1)


# In[24]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pre_test)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot= True)


# In[25]:


#test_id = test_dataset.id

#test_dataset = test_dataset.drop('id',axis=1)
#test_dataset = test_dataset/255
#test_dataset = test_dataset.values.reshape(-1,28,28,1)
test_dataset=pd.read_csv('../input/Kannada-MNIST/test.csv')
raw_test_id=test_dataset.id
test_dataset=test_dataset.drop("id",axis="columns")
test_dataset=test_dataset / 255
test=test_dataset.values.reshape(-1,28,28,1)
test.shape


# In[26]:


Y_prediction = cnn_model.predict(test)    
Y_prediction = np.argmax(Y_prediction,axis=1)


# In[27]:



test_dataset['label'] = Y_prediction
test_dataset.to_csv('submission.csv',index=False)

