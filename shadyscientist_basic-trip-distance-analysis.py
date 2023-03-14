#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


df_train = df = pd.read_csv('../input/train.csv',parse_dates=["pickup_datetime","dropoff_datetime"])

df_test = df = pd.read_csv('../input/test.csv',parse_dates=["pickup_datetime"])


# In[2]:


df_train.head()


# In[3]:


df_test.head()


# In[4]:


train_ids = set(df_train['id'].tolist())
test_ids = set(df_test['id'].tolist())


# In[5]:


len(train_ids.difference(test_ids))


# In[6]:


combine = [df_train,df_test]

for dataset in combine : 
    dataset.drop("id",axis=1,inplace=True)


# In[7]:


df_train.nsmallest(10,'trip_duration')['trip_duration']


# In[8]:


short_trips_mask = "trip_duration<=1"
len(df_train.query(short_trips_mask)['trip_duration'])


# In[9]:


df_train.nlargest(10,'trip_duration')['trip_duration']


# In[10]:


(df_train.nlargest(10,'trip_duration')['trip_duration']/60)/1440


# In[11]:


from math import sin, cos, sqrt, atan2, radians


# In[12]:


def ride_distance(location_points) : 
    
    from math import sin, cos, sqrt, atan2, radians
    
    R = 6373.0 #(Approximate.Radius of the earth in KM)

    lat1 = radians(location_points[0])
    lon1 = radians(location_points[1])
    lat2 = radians(location_points[2])
    lon2 = radians(location_points[3])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return format(distance,".2f") #In KM


# In[13]:


df_train.nlargest(4,'trip_duration')[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(ride_distance,axis=1)


# In[14]:


df_train.nlargest(4,'trip_duration')


# In[15]:


same_latlong_mask = "(pickup_latitude == dropoff_latitude) & (dropoff_longitude==pickup_longitude)"

len(df_train.query(same_latlong_mask))


# In[16]:


df_train.query(same_latlong_mask).nlargest(10,'trip_duration')['trip_duration']


# In[17]:


df_train['pickup_day']  = df_train['pickup_datetime'].dt.dayofweek
df_train['dropoff_day']  = df_train['dropoff_datetime'].dt.dayofweek


# In[18]:


df_train.head()


# In[19]:


df_train['trip_distance'] = df_train[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(ride_distance,axis=1)


# In[20]:


df_train.head()


# In[21]:


df_train['store_and_fwd_flag'] = df_train['store_and_fwd_flag'].map({"N":0,"Y":1})


# In[22]:


df_nearby_trips = df_train.query(same_latlong_mask)


# In[23]:


df_train['trip_distance'] = df_train['trip_distance'].astype("float")
df_train.head()


# In[24]:


vendor1_mask = "vendor_id == 1 "
vendor2_mask = "vendor_id == 2 "


# In[25]:


df_train.groupby("vendor_id").mean()


# In[26]:





# In[26]:




