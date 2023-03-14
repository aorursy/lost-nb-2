#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet, ElasticNetCV # Lasso is L1, Ridge is L2, ElasticNet is both
from sklearn.model_selection import ShuffleSplit # For cross validation
from sklearn.cluster import KMeans
import lightgbm as lgb # LightGBM is an alternative to XGBoost. I find it to be faster, more accurate and easier to install.

train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')
meta_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')


# In[2]:


train_df.head()


# In[3]:


weather_df.head()
# There are some missing values. We should also eventually ensure that all of the values fall within a reasonable range. 


# In[4]:


meta_df.head()
# Missing values as well. 


# In[5]:


train_df['meter'].value_counts()


# In[6]:


train_df['timestamp'][0] 


# In[7]:


train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

train_df['month'] = train_df['timestamp'].dt.month
train_df['weekday'] = train_df['timestamp'].dt.dayofweek
train_df['monthday'] = train_df['timestamp'].dt.day
train_df['hour'] = train_df['timestamp'].dt.hour
train_df['minute'] = train_df['timestamp'].dt.minute


# In[8]:


train_df['minute'].unique() # Looks like the data doesn't go down to minute resolution. Lets drop it. 


# In[9]:


train_df = train_df.drop(['minute'], axis = 1)


# In[10]:


plt.plot(train_df[train_df['building_id'] == 0]['meter_reading'], alpha = 0.8)
plt.plot(train_df[train_df['building_id'] == 1]['meter_reading'], alpha = 0.8)
plt.plot(train_df[train_df['building_id'] == 2]['meter_reading'], alpha = 0.8)
plt.plot(train_df[train_df['building_id'] == 500]['meter_reading'], alpha = 0.8)


# In[11]:


pd.plotting.lag_plot(train_df[train_df['building_id'] == 0]['meter_reading'])
plt.plot([0,400],[0,400])
# Look at the 3 clusters. 


# In[12]:


pd.plotting.lag_plot(train_df[train_df['building_id'] == 500]['meter_reading'])
plt.plot([0,400],[0,400])


# In[13]:


pd.plotting.autocorrelation_plot(train_df[train_df['building_id'] == 500]['meter_reading'])
plt.show()
pd.plotting.autocorrelation_plot(train_df[train_df['building_id'] == 500]['meter_reading'][:300])
plt.show()


# In[14]:


train_df[train_df['meter'] == 2].head()


# In[15]:


print(train_df[train_df['building_id'] == 745].meter.unique())
print(train_df[train_df['building_id'] == 1414].meter.unique())


# In[16]:


train_df[train_df['building_id'] == 745].head()


# In[17]:


plt.hist(weather_df['air_temperature'])
plt.show()


# In[18]:


all_df = pd.merge(train_df, meta_df, on = 'building_id', how = 'left')
all_df.head()


# In[19]:


weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp']) # Convert weather to the correct format before merging
all_df = pd.merge(all_df, weather_df, on = ['site_id', 'timestamp'], how = 'left')
all_df.head()


# In[20]:


data = all_df.groupby(['month', 'monthday'])[['meter']].sum()         .join(all_df.groupby(['month', 'monthday'])['air_temperature'].mean()).reset_index()
data.head()

