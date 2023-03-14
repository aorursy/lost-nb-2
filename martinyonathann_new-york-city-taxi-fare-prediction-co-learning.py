#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


df_train=pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=20000)


# In[4]:


df_train.info()


# In[5]:


df_train.head()


# In[6]:


df_train.tail()


# In[7]:


#Basic Stats of the data set
df_train.describe()


# In[8]:


#drop the negative value
print("old size: %d" % len(df_train))
df_train = df_train[df_train.fare_amount >=0]
print("New size: %d" % len(df_train))


# In[9]:


#check missing value
df_train.isnull().sum()/len(df_train)


# In[10]:


#see the distribution of fae amount
df_train.fare_amount.hist(bins=100,figsize=(16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")


# In[11]:


#lest see the distribution of fae amount less then 60
df_train[df_train.fare_amount<60].fare_amount.hist(bins=50,figsize=(16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")


# In[12]:


df_train[df_train.fare_amount > 60].shape


# In[13]:


#lest see the distribution of fare amount more than 60
df_train[df_train.fare_amount>60].fare_amount.hist(bins=100,figsize=(16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")


# In[14]:


df_train[df_train.passenger_count<6].passenger_count.hist(bins=20,figsize=(16,8))
plt.xlabel("Passanger Count")
plt.ylabel("Frequency")


# In[15]:


df_train[df_train.passenger_count==0].shape


# In[16]:


plt.figure(figsize= (16,8))
sns.boxplot(x = df_train[df_train.passenger_count< 6].passenger_count, y = df_train.fare_amount)


# In[17]:


df_train[df_train.passenger_count < 6][['fare_amount','passenger_count']].corr()


# In[18]:


df_test=pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
df_test.head()


# In[19]:


df_test.shape


# In[20]:


#check for missing value
df_test.isnull().sum()


# In[21]:


df_test.describe()


# In[22]:


min(df_test.pickup_longitude.min(),df_test.dropoff_longitude.min()), max(df_test.pickup_longitude.max(),df_test.dropoff_longitude.max())


# In[23]:


min(df_test.pickup_latitude.min(),df_test.dropoff_latitude.min()), max(df_test.pickup_latitude.max(),df_test.dropoff_latitude.max())


# In[24]:


#this function will also be used with the test set below
def select_within_test_boundary(df,BB):
    return (df.pickup_longitude>=BB[0])&(df.pickup_longitude<=BB[1])&(df.pickup_latitude>=BB[2])&(df.pickup_latitude<=BB[3])&(df.dropoff_longitude>=BB[0])&(df.dropoff_longitude<=BB[1])&(df.dropoff_latitude>=BB[2])&(df.dropoff_latitude<=BB[3])


# In[25]:


BB=(-74.5, -72.8, 40.5, 41.8)
print('Old size: %d' %len(df_train))
df_train=df_train[select_within_test_boundary(df_train,BB)]
print('New size: %d'%len(df_train))


# In[26]:


def prepare_time_features(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    df['hour_of_day'] = df.pickup_datetime.dt.hour
#     df['week'] = df.pickup_datetime.dt.week
    df['month'] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
#     df['day_of_year'] = df.pickup_datetime.dt.dayofyear
#     df['week_of_year'] = df.pickup_datetime.dt.weekofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
#     df["quarter"] = df.pickup_datetime.dt.quarter
#     df["day_of_month"] = df.pickup_datetime.dt.day
    
    return df


# In[ ]:




