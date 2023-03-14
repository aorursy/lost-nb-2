#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb

import os
from pathlib import Path

# importer la lib pour cross valider le model
from sklearn.model_selection import cross_val_score

# importer la lib pour la regression de Random Forest
from sklearn.ensemble import RandomForestRegressor

# importer la lib pour la regression de Random Forest
from sklearn.linear_model import SGDRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import ShuffleSplit


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#train = pd.read_csv('training/train.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')


# In[3]:


train.head()


# In[4]:


train.dtypes


# In[5]:


train.info()


# In[6]:


train.isna().sum()


# In[7]:


train.trip_duration.min()


# In[8]:


train.trip_duration.max()


# In[9]:


fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
plt.ylim(40.6, 40.9)
plt.xlim(-74.1,-73.7)
ax.scatter(train['pickup_longitude'],train['pickup_latitude'], s=0.0002, alpha=1)


# In[10]:


plt.subplots(figsize=(18,7))
plt.title("Répartition des outliers")
train.boxplot()


# In[11]:


#train.loc[train.trip_duration<4000,"trip_duration"].hist(bins=120)
train['trip_duration'] = np.log(train['trip_duration'].values)


# In[12]:


train['passenger_count'].value_counts()


# In[13]:


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


# In[14]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

train['hour'] = train.pickup_datetime.dt.hour
train['day'] = train.pickup_datetime.dt.dayofweek
train['month'] = train.pickup_datetime.dt.month
test['hour'] = test.pickup_datetime.dt.hour
test['day'] = test.pickup_datetime.dt.dayofweek
test['month'] = test.pickup_datetime.dt.month


# In[15]:





# In[15]:


train.isnull().sum()


# In[16]:


col_diff = list(set(train.columns).difference(set(test.columns)))


# In[17]:


train.head()


# In[18]:


y_train = train["trip_duration"] # <-- target
X_train = train[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month","hour","day","dist"]] # <-- features

X_datatest = test[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month","hour","day","dist"]]


# In[19]:


# declarer le model et l'entrainer

#sgd = SGDRegressor()
#sgd.fit(X_train, y_train)


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)


# In[21]:


rfr = RandomForestRegressor(n_estimators=100,min_samples_leaf=10, min_samples_split=15, max_depth=80,verbose=0,max_features="auto",bootstrap=True,n_jobs=-1)
rfr.fit(X_train, y_train)


# In[22]:


# Trop long
# calculer les scores de cross validation du model selon une decoupe du dataset de train
cv_scores = cross_val_score(rfr, X_train, y_train, cv=5, scoring= 'neg_mean_squared_log_error')


# In[23]:


cv_scores


# In[24]:


for i in range(len(cv_scores)):
    cv_scores[i] = np.sqrt(abs(cv_scores[i]))
cv_scores


# In[25]:


train_pred = rfr.predict(X_datatest)
train_pred[:5]


# In[26]:


train_pred


# In[27]:


my_submission = pd.DataFrame({"id": test.id, "trip_duration": np.exp(train_pred)})
print(my_submission)


# In[28]:


my_submission.to_csv('submission.csv', index=False)


# In[29]:


my_submission.head()


# In[30]:




