#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#loading of the datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


test.head()


# In[6]:


test.info()


# In[7]:


#A first comparison of the two datasets's heads reveal that they don't have the same number of columns
diff_col = list(set(train.columns).difference(set(test.columns)))
diff_col


# In[8]:


#Checking of null values
train.isna().sum()


# In[9]:


#Checking of duplicates
train.duplicated().sum()


# In[10]:


#Representation of outliers globally
plt.subplots(figsize=(18,7))
plt.title("Repartition of the outliers")
train.boxplot()


# In[11]:


fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(10,10))
plt.ylim(40.63, 40.85)
plt.xlim(-74.03, -73.77)
ax.scatter(train['pickup_longitude'],train['pickup_latitude'], s=0.0002, color='black', alpha=1)
ax.set_title("Pickup outliers representation as coordinates");


# In[12]:


ax = train['passenger_count'].value_counts(normalize=True).plot.bar();
ax.set_title("Proportion of passenger count");
ax.set_ylabel("Percentage")
ax.set_xlabel("Passenger count")


# In[13]:


train = train[train['passenger_count']>0]
train = train[train['passenger_count']<6]


# In[14]:


train.plot.scatter(x='pickup_longitude',y='pickup_latitude')


# In[15]:


train = train.loc[train['pickup_longitude']> -85]
train = train.loc[train['pickup_latitude']< 46]


# In[16]:


train.plot.scatter(x='dropoff_longitude',y='dropoff_latitude')


# In[17]:


train = train.loc[train['dropoff_longitude']> -80]
train = train.loc[train['dropoff_latitude']> 36]


# In[18]:


train.loc[train.trip_duration<5000,"trip_duration"].hist(bins=120)


# In[19]:


train = train[(train['trip_duration'] > 60) & (train['trip_duration'] < 3600)]
train[train['trip_duration']<=120].shape
train[train['trip_duration']>=3600].shape
train['trip_duration'] = np.log(train['trip_duration'].values)

train['hour'] = train['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))


test['hour'] = test['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))


# In[20]:


import math

def haversine(lat1, lon1, lat2, lon2):
   R = 6372800  # Earth radius in meters
   phi1, phi2 = math.radians(lat1), math.radians(lat2)
   dphi       = math.radians(lat2 - lat1)
   dlambda    = math.radians(lon2 - lon1)

   a = math.sin(dphi/2)**2 +        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

   return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

train['dist_long'] = train['pickup_longitude'] - train['dropoff_longitude']
test['dist_long'] = test['pickup_longitude'] - test['dropoff_longitude']

train['dist_lat'] = train['pickup_latitude'] - train['dropoff_latitude']
test['dist_lat'] = test['pickup_latitude'] - test['dropoff_latitude']

train['dist'] = np.sqrt(np.square(train['dist_long']) + np.square(train['dist_lat']))
test['dist'] = np.sqrt(np.square(test['dist_long']) + np.square(test['dist_lat']))


# In[21]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])


# In[22]:


ax = train['pickup_datetime'].dt.month.value_counts(normalize=True, ascending=True,).plot.bar()
ax.set_title("Pickup frequency by months ");
ax.set_xlabel("month");
ax.set_ylabel("frequency")


# In[23]:


train['minute'] = train.pickup_datetime.dt.minute
train['hour'] = train.pickup_datetime.dt.hour
train['day'] = train.pickup_datetime.dt.dayofweek
train['month'] = train.pickup_datetime.dt.month
test['minute'] = train.pickup_datetime.dt.minute
test['hour'] = test.pickup_datetime.dt.hour
test['day'] = test.pickup_datetime.dt.dayofweek
test['month'] = test.pickup_datetime.dt.month


# In[24]:


y = train["trip_duration"] # <-- target
X_train = train[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","dist","month","hour","day"]] # <-- features

X_test = test[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","dist","month","hour","day"]]


# In[25]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[26]:


xtrain, xvalid, ytrain, yvalid = train_test_split(X_train,y, test_size=0.2, random_state=42)
xtrain.shape, xvalid.shape, xtrain.shape, yvalid.shape


# In[27]:


# from sklearn.ensemble import RandomForestRegressor


# In[28]:


# m1 = RandomForestRegressor(n_estimators=20, random_state=42)
# m1.fit(X, y)


# In[29]:


# from sklearn.model_selection import ShuffleSplit

# shuff = ShuffleSplit(n_splits=4, test_size=0.8, random_state=42)


# In[30]:


# m1_scores = cross_val_score(m1, X, y, cv=shuff, scoring ="neg_mean_squared_log_error")


# In[31]:


# #using RMSE for scoring
# for i in range(len(m1_scores)):
#     m1_scores[i] = np.sqrt(abs(m1_scores[i])) #abs -> only the positive values thanks to 'abs'
# np.mean(m1_scores)

# TOO LONG AND NOT ENOUGH EFFICIENT !


# In[32]:


import lightgbm as lgb


# In[33]:


dtrain = lgb.Dataset(X_train,y)


# In[34]:


lgb_params = {
    'learning_rate': 0.1,
    'max_depth': 25,
    'num_leaves': 1000, 
    'objective': 'regression', #For our case of regression
    #'metric': {'rmse'},
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    #'bagging_freq': 5,
    'max_bin': 1000}       # 1000


# In[35]:


model_lgb = lgb.train(lgb_params, 
                      dtrain,
                      num_boost_round=1200)

#STABLE, QUICK AND SUFFICIENTLY EFFICIENT


# In[36]:


cv_results = lgb.cv(
        lgb_params,
        dtrain,
        num_boost_round=100,
        nfold=3,
        metrics='mae',
        early_stopping_rounds=10,
        stratified=False
        )


# In[37]:


print('Current parameters:\n', lgb_params)
print('\nBest num_boost_round:', len(cv_results['l1-mean']))
print('Best CV score:', cv_results['l1-mean'][-1])


# In[38]:


test.head()


# In[39]:


#storing the predicitions

pred_test = np.exp(model_lgb.predict(X_test))
pred_test


# In[40]:


submit = pd.read_csv('../input/sample_submission.csv')


# In[41]:


submit.head()


# In[42]:


submit['trip_duration'] = pred_test
submit.head()


# In[43]:


submit_file = pd.DataFrame({"id": test.id, "trip_duration": pred_test})


# In[44]:


submit_file.to_csv('submission.csv', index=False)

