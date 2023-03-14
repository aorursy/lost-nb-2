#!/usr/bin/env python
# coding: utf-8

# In[111]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,BatchNormalization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[112]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()


# In[113]:


test.head()


# In[114]:


train.info()


# In[115]:


test.info()


# In[116]:


col = ("spacegroup","number_of_total_atoms","percent_atom_al","percent_atom_ga","percent_atom_in","lattice_vector_1_ang","lattice_vector_2_ang",
       "lattice_vector_3_ang","lattice_angle_alpha_degree","lattice_angle_beta_degree","lattice_angle_gamma_degree")
train_x = train.loc[:,col]
train_y = train.loc[:,("formation_energy_ev_natom","bandgap_energy_ev")]
test_x = test.loc[:,col]

train_x = train_x.as_matrix()
train_y = train_y.as_matrix()
test_x = test_x.as_matrix()
cv_x = train_x[:400,:]
cv_y = train_y[:400,:]
train_x = train_x[400:,:]
train_y = train_y[400:,:]
print(train_x.shape,train_y.shape,test_x.shape,cv_x.shape,cv_y.shape)


# In[117]:


print(train_x[0:2])


# In[118]:


from keras import backend as K
def my_loss(y_true,y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_true+1)-K.log(y_pred+1))))


# In[ ]:


for i in range(1):
    model = Sequential()
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,input_shape=(11,)))
    model.add(Dense(32, kernel_initializer='random_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(16, kernel_initializer='random_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(8, kernel_initializer='random_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(4, kernel_initializer='random_uniform'))
    model.add(Dense(2))
    
    model.compile(loss=my_loss, optimizer="sgd")
    model.fit(train_x, train_y,epochs=200, batch_size=128,verbose = True)

    score = model.evaluate(cv_x, cv_y, batch_size=128,verbose = False)
    print("在cv集合上的结果：",score)


# In[120]:


pred = model.predict(test_x)
print(pred.shape)
for i in range(pred.shape[0]):
    print(pred[i,0],pred[i,1])


# In[ ]:





# In[121]:


answer[]

