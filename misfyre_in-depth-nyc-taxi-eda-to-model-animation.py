#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import operator
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from dateutil import parser
from matplotlib import animation
import io
import base64
from IPython.display import HTML

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[2]:


# Reading the Train Data and looking at the Given Features
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.columns)
print(train.info())
print(test.info())


# In[3]:


# First Look at the Data
print('We have {} training rows and {} test rows.'.format(train.shape[0], test.shape[0]))
print('We have {} training columns and {} test columns.'.format(train.shape[1], test.shape[1]))
train.head(2)


# In[4]:


vendor_popularity = (train['vendor_id'].value_counts())
popularity_dict = dict(vendor_popularity)

print('Most Popular Vendor:', max(vendor_popularity.iteritems(), key=operator.itemgetter(1))[0])
print('Difference in Popularity:', popularity_dict[2] - popularity_dict[1])

f = plt.figure(figsize=(10,5))
sns.barplot(vendor_popularity.index, vendor_popularity.values, alpha=0.8)
plt.xlabel('Vendor', fontsize=14)
plt.ylabel('Trips', fontsize=14)
plt.show()


# In[5]:


vendor1_change = []
vendor2_change = []

for i, row in train.iterrows():    
    
    if row['vendor_id'] == 1:
        if vendor1_change:
            list.append(vendor1_change, vendor1_change[-1] + 1)
        else:
            list.append(vendor1_change, 1)
        if vendor2_change:
            list.append(vendor2_change, vendor2_change[-1])
        else:
            list.append(vendor2_change, 0)
            
    if row['vendor_id'] == 2:
        if vendor2_change:
            list.append(vendor2_change, vendor2_change[-1] + 1)
        else:
            list.append(vendor2_change, 1)
        if vendor1_change:
            list.append(vendor1_change, vendor1_change[-1])
        else:
            list.append(vendor1_change, 0)

plt.figure(figsize=(10,5))
plt.scatter(range(train.shape[0]), vendor1_change)
plt.scatter(range(train.shape[0]), vendor2_change)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Trips Requested', fontsize=12)
plt.show()


# In[6]:


# Feature Engineering
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

train['pickup_date'] = train['pickup_datetime'].dt.date
train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
train['pickup_day'] = train['pickup_datetime'].dt.day
train['pickup_month'] = train['pickup_datetime'].dt.month
train['pickup_hour'] = train['pickup_datetime'].dt.hour
train['pickup_minute'] = train['pickup_datetime'].dt.minute
train['pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).map(
    lambda x: x.total_seconds())

test['pickup_date'] = test['pickup_datetime'].dt.date
test['pickup_weekday'] = test['pickup_datetime'].dt.weekday
test['pickup_day'] = test['pickup_datetime'].dt.day
test['pickup_month'] = test['pickup_datetime'].dt.month
test['pickup_hour'] = test['pickup_datetime'].dt.hour
test['pickup_minute'] = test['pickup_datetime'].dt.minute
test['pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).map(
    lambda x: x.total_seconds())


# In[7]:


day, count = np.unique(train['pickup_weekday'], return_counts = True)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = day, y = count)
ax.set(xlabel = "Day of Week", ylabel = "Count of Taxi Rides")
plt.show();


# In[8]:


day, count = np.unique(train['pickup_day'], return_counts = True)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = day, y = count)
ax.set(xlabel = "Day of Month", ylabel = "Count of Taxi Rides")
plt.show();


# In[9]:


day, count = np.unique(train['pickup_month'], return_counts = True)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = day, y = count)
ax.set(xlabel = "Month in Year", ylabel = "Count of Taxi Rides")
plt.show();


# In[10]:


day, count = np.unique(train['pickup_hour'], return_counts = True)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = day, y = count)
ax.set(xlabel = "Hour in Day", ylabel = "Count of Taxi Rides")
plt.show();


# In[11]:


passengers, count = np.unique(train['passenger_count'], return_counts = True)
passenger_count = train['passenger_count'].value_counts()
print(passenger_count)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = passengers, y = count)
ax.set(xlabel = "Number of Passengers", ylabel = "Count of Taxi Rides")
plt.show();


# In[12]:


# Pickup Latitude/Longitude
sns.lmplot(x="pickup_longitude", y="pickup_latitude", fit_reg=False, 
           size=9, scatter_kws={'alpha':0.3,'s':5}, data=train[(
                 train.pickup_longitude>train.pickup_longitude.quantile(0.005))
               &(train.pickup_longitude<train.pickup_longitude.quantile(0.995))
               &(train.pickup_latitude>train.pickup_latitude.quantile(0.005))                           
               &(train.pickup_latitude<train.pickup_latitude.quantile(0.995))])

plt.xlabel('Pickup Longitude');
plt.ylabel('Pickup Latitude');
plt.show()


# In[13]:


# Dropoff Latitude/Longitude
sns.lmplot(x="dropoff_longitude", y="dropoff_latitude", fit_reg=False, 
           size=9, scatter_kws={'alpha':0.3,'s':5}, data=train[(
                 train.dropoff_longitude>train.dropoff_longitude.quantile(0.005))
               &(train.dropoff_longitude<train.dropoff_longitude.quantile(0.995))
               &(train.dropoff_latitude>train.dropoff_latitude.quantile(0.005))                           
               &(train.dropoff_latitude<train.dropoff_latitude.quantile(0.995))])

plt.xlabel('Dropoff Longitude');
plt.ylabel('Dropoff Latitude');
plt.show()


# In[14]:


fig = plt.figure(figsize = (10,10))
ax = plt.axes()


# In[15]:


# Weekday Pickup Latitude/Longitude
# Credit to DrGuillermo for the Animation idea

def pickup_weekday(day):
    ax.clear()
    ax.set_title('Pickup Locations - Day ' + str(int(day)))    
    plt.figure(figsize = (8,10))
    temp = train[train['pickup_weekday'] == day]
    temp = temp[(
        train.pickup_longitude>train.pickup_longitude.quantile(0.005))
      &(train.pickup_longitude<train.pickup_longitude.quantile(0.995))
      &(train.pickup_latitude>train.pickup_latitude.quantile(0.005))                           
      &(train.pickup_latitude<train.pickup_latitude.quantile(0.995))]
    ax.plot(temp['pickup_longitude'], temp['pickup_latitude'],'.', 
            alpha = 1, markersize = 2, color = 'gray')

ani = animation.FuncAnimation(fig,pickup_weekday,sorted(train.pickup_weekday.unique()), interval = 1000)
ani.save('animation.gif', writer='imagemagick', fps=2)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# In[16]:


# Weekday Pickup Latitude/Longitude
# Credit to DrGuillermo for the Animation idea

def pickup_hour(hour):
    ax.clear()
    ax.set_title('Pickup Locations - Hour ' + str(int(hour)))    
    plt.figure(figsize = (8,10))
    temp = train[train['pickup_hour'] == hour]
    temp = temp[(
        train.pickup_longitude>train.pickup_longitude.quantile(0.005))
      &(train.pickup_longitude<train.pickup_longitude.quantile(0.995))
      &(train.pickup_latitude>train.pickup_latitude.quantile(0.005))                           
      &(train.pickup_latitude<train.pickup_latitude.quantile(0.995))]
    ax.plot(temp['pickup_longitude'], temp['pickup_latitude'],'.', 
            alpha = 1, markersize = 2, color = 'gray')

ani = animation.FuncAnimation(fig,pickup_hour,sorted(train.pickup_hour.unique()), interval = 1000)
ani.save('animation.gif', writer='imagemagick', fps=2)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# In[17]:


# Feature Engineering (Credit to Beluga)
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train['distance_haversine'] = haversine_array(
    train['pickup_latitude'].values, train['pickup_longitude'].values,
    train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train['distance_dummy_manhattan'] = dummy_manhattan_distance(
    train['pickup_latitude'].values, train['pickup_longitude'].values,
    train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test['distance_haversine'] = haversine_array(
    test['pickup_latitude'].values, test['pickup_longitude'].values,
    test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test['distance_dummy_manhattan'] = dummy_manhattan_distance(
    test['pickup_latitude'].values, test['pickup_longitude'].values,
    test['dropoff_latitude'].values, test['dropoff_longitude'].values)

train['avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train['avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

train['center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train['center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
test['center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
test['center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

train['pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)
train['pickup_long_bin'] = np.round(train['pickup_longitude'], 2)
train['center_lat_bin'] = np.round(train['center_latitude'], 2)
train['center_long_bin'] = np.round(train['center_longitude'], 2)
train['pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))
test['pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)
test['pickup_long_bin'] = np.round(test['pickup_longitude'], 2)
test['center_lat_bin'] = np.round(test['center_latitude'], 2)
test['center_long_bin'] = np.round(test['center_longitude'], 2)
test['pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, 
                                          train['pickup_longitude'].values, 
                                          train['dropoff_latitude'].values, 
                                          train['dropoff_longitude'].values)

test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, 
                                         test['pickup_longitude'].values, 
                                         test['dropoff_latitude'].values, 
                                         test['dropoff_longitude'].values)


# In[18]:


# Feature Engineering (Credit to Beluga)
full = pd.concat([train, test])
coords = np.vstack((full[['pickup_latitude', 'pickup_longitude']], 
                   full[['dropoff_latitude', 'dropoff_longitude']]))

pca = PCA().fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

train['pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) +                          np.abs(train['dropoff_pca0'] - train['pickup_pca0'])

test['pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) +                         np.abs(test['dropoff_pca0'] - test['pickup_pca0'])


# In[19]:


corr = train.corr()
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()


# In[20]:




