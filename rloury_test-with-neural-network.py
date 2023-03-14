#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc;gc.collect
import datetime

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[2]:


#holidays_events = pd.read_csv('../input/holidays_events.csv')
#items = pd.read_csv('../input/items.csv')
#oil = pd.read_csv('../input/oil.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
#stores = pd.read_csv('../input/stores.csv')
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
#transactions = pd.read_csv('../input/transactions.csv')


# In[3]:


train["date"] =  pd.to_datetime(train["date"])
train_2016 = train[train["date"].dt.year == 2016]
date_list=np.unique(train_2016["date"])
del train; gc.collect()


# In[4]:


#train_2016["Year"]=train_2016["date"].dt.year
train_2016["Month"]=train_2016["date"].dt.month
#train_2016["Day"]=train_2016["date"].dt.day


# In[5]:


train_2016.head()
date_list=pd.DataFrame(date_list)
date_list["day"]=range(1,np.shape(date_list)[0]+1)
date_list["day"]=date_list["day"]%7
date_list.columns=["date","day of week"]


# In[6]:


train_2016=pd.merge(train_2016,date_list,on=["date"],how="left")


# In[7]:


train_2016.drop("date",axis=1,inplace=True)
train_2016.drop("id",axis=1,inplace=True)
train_2016.reset_index(inplace=True)
train_2016.drop("index",axis=1,inplace=True)


# In[8]:


train_2016.head()


# In[9]:


#store_list=pd.DataFrame(np.unique(train_2016["store_nbr"]))
#item_list=pd.DataFrame(np.unique(train_2016["item_nbr"]))


# In[10]:


#date_list["Month"]=date_list[0].dt.month
#date_list
del date_list


# In[11]:


memo = train_2016.memory_usage(index=True).sum()
print(memo/ 1024**2," MB")


# In[12]:


print(train_2016.dtypes)


# In[13]:


train_2016['store_nbr'] = train_2016['store_nbr'].astype(np.int8)
train_2016['item_nbr'] = train_2016['item_nbr'].astype(np.int32)
train_2016['unit_sales'] = train_2016['unit_sales'].astype(np.int32)
train_2016['Month'] = train_2016['Month'].astype(np.int8)
train_2016['day of week'] = train_2016['day of week'].astype(np.int8)


# In[14]:


memo = train_2016.memory_usage(index=True).sum()
print(memo/ 1024**2," MB")


# In[15]:


train_2016.head()


# In[16]:


onpromotion={True : 1,False: 0}
train_2016["onpromotion"]=train_2016["onpromotion"].map(onpromotion)
train_2016.head()


# In[17]:


memo = train_2016.memory_usage(index=True).sum()
print(memo/ 1024**2," MB")


# In[18]:


train_2016['onpromotion'] = train_2016['onpromotion'].astype(np.bool)


# In[19]:


memo = train_2016.memory_usage(index=True).sum()
print(memo/ 1024**2," MB")


# In[20]:


train_2016.head()


# In[21]:


print(np.shape(np.unique(train_2016["item_nbr"]))[0]*np.shape(np.unique(train_2016["store_nbr"]))[0]*365)
np.shape(train_2016)[0]


# In[22]:


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[23]:


train_2016 = train_2016[train_2016["store_nbr"] <= 25]
train_2016 = train_2016[train_2016["item_nbr"] == 105574]
train, test =train_2016[train_2016["Month"] <= 11], train_2016[train_2016["Month"] > 11]
TrainY=train["unit_sales"].values
TrainX=train.drop("unit_sales",axis=1).values

TestY=test["unit_sales"].values
TestX=test.drop("unit_sales",axis=1).values


# In[24]:


model = Sequential()
model.add(Dense(50, input_dim=5, activation='relu'))
model.add(Dense(1,))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')


# In[25]:


model.fit(TrainX, TrainY,
          batch_size=10,epochs=20,verbose=1,
          validation_data=(TestX, TestY))


# In[26]:


testPredict = model.predict(TestX)
result=pd.DataFrame(testPredict)
result["TestY"]=TestY
print(result)


# In[27]:





# In[27]:




