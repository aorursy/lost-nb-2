#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("/kaggle/input"))

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import math


# In[2]:


train = pd.read_csv('../input/train.csv', nrows=1000000)
test = pd.read_csv('../input/test.csv')
train.head()


# In[3]:


print(train.isnull().sum())
train = train.dropna(how='any',axis=0)


# In[4]:


train['abs_diff_longitude'] = np.abs(train['dropoff_longitude'] - train['pickup_longitude'])
train['abs_diff_latitude'] = np.abs(train['dropoff_latitude'] - train['pickup_latitude'])
test['abs_diff_longitude'] = np.abs(test['dropoff_longitude'] - test['pickup_longitude'])
test['abs_diff_latitude'] = np.abs(test['dropoff_latitude'] - test['pickup_latitude'])


# In[5]:


# sns.relplot(x='passenger_count',y='fare_amount',data=train.iloc[:1000,:],kind='scatter')
sns.countplot(x='passenger_count',data=train.iloc[:1000,:])
sns.relplot(x='abs_diff_longitude', y="fare_amount",data=train.loc[:1000,:])
sns.relplot(x="abs_diff_latitude", y='fare_amount',data=train.loc[:1000,:])


# In[6]:


train = train.loc[train['fare_amount']>0,:]
train = train.loc[(train["passenger_count"]<=6) & (train["passenger_count"]>0),:]
train = train.loc[(train["abs_diff_latitude"]<2) & (train["abs_diff_longitude"]<2),:]
train = train.loc[(train["abs_diff_latitude"]>0) & (train["abs_diff_longitude"]>0),:]


# In[7]:


# sns.relplot(x='passenger_count',y='fare_amount',data=train,kind='scatter')
sns.countplot(x='passenger_count',data=train.iloc[:1000000,:])
# f,ax = plt.subplots(1,2,figsize=(18,8))
sns.relplot(x='abs_diff_longitude', y="fare_amount",data=train.loc[:1000,:])
sns.relplot(x="abs_diff_latitude", y='fare_amount',data=train.loc[:1000,:])


# In[8]:


train.loc[:,'timestamp_with_key'] = train.loc[:,'key'] 
test.loc[:,'timestamp_with_key'] = test.loc[:,'key']
train.key = pd.DataFrame({'key':train['key'].str.split('.').str[1].astype('int')})
test.key = pd.DataFrame({'key':test['key'].str.split('.').str[1].astype('int')})


# In[9]:


# #using datetime try to analyse is there cor b/w time and price (you could get time windows
# #(divide a day into 8 parts) and add that feature and see cor b/w time and price)
# from math import floor
# def chooseSlot(x):
#     hr = x.hour
#     return int(hr/3 + 1)

# train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], infer_datetime_format=True).dt.tz_localize('UTC')
# test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], infer_datetime_format=True).dt.tz_localize('UTC')
# train['time_slot'] = pd.DataFrame(list(map(lambda x : chooseSlot(x), train['pickup_datetime'][:])), index=train.index)
# test['time_slot'] = pd.DataFrame(list(map(lambda x : chooseSlot(x), test['pickup_datetime'][:])), index=test.index)


# In[10]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], infer_datetime_format=True).dt.tz_localize('UTC')
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], infer_datetime_format=True).dt.tz_localize('UTC')
train.loc[:,'hour_no'] = train['pickup_datetime'][:].dt.strftime('%-H').astype('int')
test.loc[:,'hour_no'] = test['pickup_datetime'][:].dt.strftime('%-H').astype('int')
train.loc[:,'weekday_no'] = train['pickup_datetime'][:].dt.strftime('%w').astype('int')
test.loc[:,'weekday_no'] = test['pickup_datetime'][:].dt.strftime('%w').astype('int')
train.loc[:,'day_no'] =  train['pickup_datetime'][:].dt.strftime('%-d').astype('int')
test.loc[:,'day_no'] = test['pickup_datetime'][:].dt.strftime('%-d').astype('int')
train.loc[:,'year_no'] = train['pickup_datetime'][:].dt.strftime('%-y').astype('int')
test.loc[:,'year_no'] = test['pickup_datetime'][:].dt.strftime('%-y').astype('int')


# In[11]:


sns.countplot(x='hour_no',data=train.loc[:1000,:])
# sns.countplot(x='weekday_no',data=train.loc[:1000,:])
# sns.countplot(x='year_no',data=train.loc[:1000,:])


# In[12]:


# def eucledian(x):
#     dist = np.sqrt( np.power(x[0]-x[2],2) + np.power(x[1]-x[3],2))
#     return dist

# train['dist_eucledian'] = pd.DataFrame(list(
#     map(lambda x: eucledian(x), train[["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]].values)),
#                                       index=train.index)


# In[13]:


def dist_haversine(x):
    R = 6371 #for metres 6371e3
    picklat = math.radians(x[1])
    droplat = math.radians(x[3])
    latdiff = abs(droplat-picklat)
    picklon = math.radians(x[0])
    droplon = math.radians(x[2])
    londiff = abs(droplon-picklon)

    a = math.sin(latdiff/2) * math.sin(latdiff/2) +            math.cos(picklat) * math.cos(droplat) *            math.sin(londiff/2) * math.sin(londiff/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return (R * c)

train['dist_haversine_km'] = pd.DataFrame(list(
    map(lambda x: dist_haversine(x), train[["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]].values)),
                                      index=train.index)

test['dist_haversine_km'] = pd.DataFrame(list(
    map(lambda x: dist_haversine(x), test[["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]].values)),
                                      index=test.index)


# In[14]:


# train['dist_haversine_km'] = 1
# for i in range(9):
#     train['dist_haversine_km'][(i*1000000):((i+1) * 1000000)] = pd.DataFrame(list(
#         map(lambda x: dist_haversine(x),
#             train[["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]][(i*1000000):((i+1) * 1000000)].values)))
                                                                             
# # ),index=train.index[(i*1000000):((i+1) * 1000000)].reshape(len(train.index[(i*1000000):((i+1) * 1000000)]),1)
                                                                             
# train['dist_haversine_km'][(i*1000000):] = pd.DataFrame(list(
#     map(lambda x: dist_haversine(x),
#         train[["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]][(i*1000000):].values)))
# # ),index=train.index[(i*1000000):].reshape(len(train.index[(i*1000000):]),1)


# In[15]:


train['fare_per_km'] = train['fare_amount']/(train["dist_haversine_km"])
train['fare_per_km_passenger'] = train['fare_amount']/(train['dist_haversine_km']*train['passenger_count'])


# In[16]:


train.groupby('key').agg({'fare_per_km_passenger':'mean','key':'count','passenger_count':'mean','fare_amount':'mean'})


# In[17]:


train.loc[train['fare_per_km_passenger']>20,['fare_per_km_passenger','fare_amount','dist_haversine_km']]
# train[['fare_per_km_passenger','fare_amount','dist_haversine_km','passenger_count']][30:60]


# In[18]:


grouped_df = train.groupby('key')
count = 0
for key, item in grouped_df:
    count += 1
    if count == 2: ## to view key = 2
        filtered = grouped_df.get_group(key)["dist_haversine_km"]>1 #ignoring drives within 1km
        df = pd.DataFrame(grouped_df.get_group(key).loc[filtered,:].sort_values(by='pickup_datetime'))
#         print(grouped_df.get_group(key)[['pickup_datetime','fare_per_km_passenger']][grouped_df.get_group(key)["dist_haversine_km"]>1].describe())
        break

indexes = ['key',df['pickup_datetime'].dt.strftime('%a'),'hour_no']
grouped = df[:][:].groupby(indexes).agg({'fare_per_km_passenger':'mean','fare_per_km':'mean','fare_amount':'count'}) 
grouped.rename(columns={'fare_amount':'count'},inplace=True)
grouped


# In[19]:


reindexed = grouped.reset_index().drop('key',axis=1)
get_max_count = reindexed.groupby(['pickup_datetime']).agg({'count':'max'})
get_max_count = get_max_count.reindex(reindexed['pickup_datetime'], method='ffill')
reindexed = reindexed.set_index('pickup_datetime')
reindexed.loc[get_max_count['count'] == reindexed['count'],:]


# In[20]:


#train.loc[ ((train['fare_per_km']<0.2) & (train['dist_haversine_km']>1))]
train = train.loc[ ~ (((train['dist_haversine_km']>1) & train['fare_per_km']<0.2))]
train = train.loc[~ ((train['dist_haversine_km']<0.01) & (train['fare_per_km']>50))]


# In[21]:


train = train.drop(['fare_per_km_passenger','fare_per_km'],axis=1)


# In[22]:


# train['pickup_latitude'].shape[0] - train['pickup_latitude'].between(40,41).sum()
print(train.shape[0] - train.loc[train['pickup_latitude'].between(40.5,41) & train['dropoff_latitude'].between(40.5,41) &
          train['pickup_longitude'].between(-74,-73.9) &  train['dropoff_longitude'].between(-74,-73.9)].shape[0])
old_train = train.copy()
train = train.loc[train['pickup_latitude'].between(40.5,41) & train['dropoff_latitude'].between(40.5,41) &
                  train['pickup_longitude'].between(-74,-73.9) &  train['dropoff_longitude'].between(-74,-73.9)]
# dont alter test data


# In[23]:


train.loc[:,'pickuplat_no'], pick_lat_bin = pd.cut(train['pickup_latitude'],100, labels=False, retbins=True)
train.loc[:,'pickuplong_no'], pick_long_bin = pd.cut(train['pickup_longitude'],100, labels=False, retbins=True)
train.loc[:,'dropofflat_no'], drop_lat_bin = pd.cut(train['dropoff_latitude'],100, labels=False, retbins=True)
train.loc[:,'dropofflong_no'], drop_long_bin = pd.cut(train['dropoff_longitude'],100, labels=False, retbins=True)
test['pickuplat_no'] = pd.cut(test['pickup_latitude'], pick_lat_bin, labels=False).fillna(int(train.loc[:,'pickuplat_no'].mean()))
test['pickuplong_no'] = pd.cut(test['pickup_longitude'], pick_long_bin, labels=False).fillna(int(train.loc[:,'pickuplong_no'].mean()))
test['dropofflat_no'] = pd.cut(test['dropoff_latitude'], drop_lat_bin, labels=False).fillna(int(train.loc[:,'dropofflat_no'].mean()))
test['dropofflong_no'] = pd.cut(test['dropoff_longitude'], drop_long_bin, labels=False).fillna(int(train.loc[:,'dropofflong_no'].mean()))


# In[24]:


train.loc[:,'pickdrop_lat_diff'] = abs(train['pickuplat_no'].astype(int) - train['dropofflat_no'].astype(int))#.astype('category')
train.loc[:,'pickdrop_long_diff'] = abs(train['pickuplong_no'].astype(int) - train['dropofflong_no'].astype(int))#.astype('category')
train.loc[:,'final_dist_factor'] = (train['pickdrop_lat_diff'].astype(int) + train['pickdrop_long_diff'].astype(int))#.astype('category')
test.loc[:,'pickdrop_lat_diff'] = abs(test['pickuplat_no'].astype(int) - test['dropofflat_no'].astype(int))#.astype('category')
test.loc[:,'pickdrop_long_diff'] = abs(test['pickuplong_no'].astype(int) - test['dropofflong_no'].astype(int))#.astype('category')
test.loc[:,'final_dist_factor'] = (test['pickdrop_lat_diff'].astype(int) + test['pickdrop_long_diff'].astype(int))#.astype('category')


# In[25]:


#  train.groupby('pickuplat_no')['key'].count().sort_values(ascending=False).reset_index()


# In[26]:


def calc_cwd_factor(df, col):
    new_df = df.groupby(col)['key'].count().sort_values(ascending=False).reset_index()
    new_df['cwd_factor'] = 1
    count = 1
    for i in range(1,new_df.shape[0]):
        count += 1
        if new_df.loc[i-1,'key'] == new_df.loc[i,'key']:
            count -= 1
        new_df.loc[i,'cwd_factor'] = count
    new_df.index = new_df[col]
    return new_df

fact_df = calc_cwd_factor(train, 'pickuplat_no')
train.loc[:,'pickuplat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], train['pickuplat_no']))
fact_df = calc_cwd_factor(train, 'pickuplong_no')
train.loc[:,'pickuplong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], train['pickuplong_no']))
fact_df = calc_cwd_factor(train, 'dropofflat_no')
train.loc[:,'dropofflat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], train['dropofflat_no']))
fact_df = calc_cwd_factor(train, 'dropofflong_no')
train.loc[:,'dropofflong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], train['dropofflong_no']))
#do the same for test set:
fact_df = calc_cwd_factor(test, 'pickuplat_no')
test.loc[:,'pickuplat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['pickuplat_no']))
fact_df = calc_cwd_factor(test, 'pickuplong_no')
test.loc[:,'pickuplong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['pickuplong_no']))
fact_df = calc_cwd_factor(test, 'dropofflat_no')
test.loc[:,'dropofflat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['dropofflat_no']))
fact_df = calc_cwd_factor(test, 'dropofflong_no')
test.loc[:,'dropofflong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['dropofflong_no']))


# In[27]:


print(train.shape,test.shape)

