#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1. # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[ ]:


train_df = pd.read_csv('../input/train.csv', nrows = 1000000)


# In[ ]:


train_df = train_df.dropna(how = 'any', axis = 'rows')


# In[ ]:


a = train_df[train_df['passenger_count']>=5]


# In[ ]:


b = train_df[train_df['passenger_count']<5]


# In[ ]:


a['fare_amount'].mean()


# In[ ]:


b['fare_amount'].mean()


# In[ ]:


train_df.groupby('passenger_count')['fare_amount'].median()


# In[ ]:


train_df.groupby('passenger_count')['fare_amount'].mean()


# In[ ]:


train_df = train_df[(train_df['passenger_count']<=6) & (train_df['passenger_count']!=0)]


# In[ ]:


def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df)


# In[ ]:


train_df.plot.scatter(x='abs_diff_longitude', y = 'abs_diff_latitude')


# In[ ]:


train_df = train_df[(train_df['abs_diff_longitude']<5) & (train_df['abs_diff_latitude']<5)]


# In[ ]:


train_df = train_df[(train_df['fare_amount']<250) & (train_df['fare_amount']>=2.5)]


# In[ ]:


train_df = train_df[(train_df['pickup_longitude']!=0.0) & (train_df['pickup_latitude']!=0.0)]


# In[ ]:


train_df = train_df[(train_df['dropoff_longitude']!=0.0) & (train_df['dropoff_latitude']!=0.0)]


# In[ ]:


train_df['pickup_datetime'] = train_df['pickup_datetime'].map(lambda x: x[11:13])


# In[ ]:


train_df['pickup_datetime'] = train_df['pickup_datetime'].map(lambda x: (int(x)+19)%24)


# In[ ]:


train_df.head(20)

