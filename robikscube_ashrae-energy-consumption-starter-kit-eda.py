#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns

train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
weather_te = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
weather_tr = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')
bmd = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

# Set timestamps
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])
weather_tr['timestamp'] = pd.to_datetime(weather_tr['timestamp'])
weather_te['timestamp'] = pd.to_datetime(weather_te['timestamp'])

sns.set(style="whitegrid")
sns.set_color_codes("pastel")


# In[2]:


meter_mapping = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
train['meter_type'] = train['meter'].map(meter_mapping)
test['meter_type'] = test['meter'].map(meter_mapping)


# In[3]:


train.groupby(['timestamp','meter_type'])['meter_reading']     .median()     .reset_index().set_index('timestamp')     .groupby('meter_type')['meter_reading']     .plot(figsize=(15, 5), title='Median Meter Reading by Meter Type (Test Set)')
plt.legend()
plt.show()


# In[4]:


train['train'] = 1
test['train'] = 0
tt = pd.concat([train, test], axis=0, sort=True)

tt.groupby(['timestamp','meter_type'])['meter_reading']     .median()     .reset_index().set_index('timestamp')     .groupby('meter_type')['meter_reading']     .plot(figsize=(15, 5), title='Median Meter Reading by Meter Type (train and test timeframe)')
plt.legend()
plt.show()


# In[5]:


pd.DataFrame(train.groupby('meter_type')['meter_reading']                  .describe()                  .astype(int))                  .sort_values('count',
                              ascending=False)


# In[6]:


train['meter_reading'].plot(kind='hist',
                        bins=50,
                        figsize=(15, 2),
                       title='Distribution of Target Variable (meter_reading)')
plt.show()


# In[7]:


train.query('meter_reading < 5000')['meter_reading']     .plot(kind='hist',
          figsize=(15, 3),
          title='Distribution of meter_reading, excluding values greater than 5000',
          bins=200)
plt.show()
train.query('meter_reading < 500')['meter_reading']     .plot(kind='hist',
          figsize=(15, 3),
          title='Distribution of meter_reading, excluding values greater than 500',
         bins=200)
plt.show()
train.query('meter_reading < 100')['meter_reading']     .plot(kind='hist',
          figsize=(15, 3),
          title='Distribution of meter_reading, excluding values greater than 100',
         bins=100)
plt.show()


# In[8]:


train.query('building_id == 0 and meter == 0')     .set_index('timestamp')['meter_reading'].plot(figsize=(15, 3),
                                                 title='Building 0 - Meter 0')

plt.show()
train.query('building_id == 753').set_index('timestamp').groupby('meter')['meter_reading'].plot(figsize=(15, 3),
                                                 title='Building 753 - Meters 0-3')
plt.show()
train.query('building_id == 1322').set_index('timestamp').groupby('meter')['meter_reading'].plot(figsize=(15, 3),
                                                 title='Building 1322 - Meters 0-3')
plt.show()


# In[9]:


# First take a look at the building metadata
bmd.describe()


# In[10]:


bmd.groupby('year_built')['site_id']     .count()     .plot(figsize=(15, 5),
          style='.-',
          title='Building Meta Data - Count by Year Built')
plt.show()
print('{} Buildings have no year data.'.format(np.sum(bmd['year_built'].isna())))


# In[11]:


bmd.groupby('primary_use')     .count()['site_id']     .sort_values()     .plot(kind='barh',
          figsize=(15, 5),
          title='Count of Buildings by Primary Use')
plt.show()


# In[12]:


# Aggregate some meter reading stats
meter_reading_stats = train.groupby('building_id')['meter_reading'].agg(['mean','max','min']).reset_index()
bmd_with_stats = pd.merge(bmd, meter_reading_stats, on=['building_id']).rename(columns={'mean':'mean_meter_reading',
                                                                       'max':'max_meter_reading',
                                                                       'min':'min_meter_reading'})


# In[13]:


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
sns.pairplot(bmd_with_stats.dropna(),
             vars=['mean_meter_reading','min_meter_reading',
                   'max_meter_reading','square_feet','year_built'],
             hue='primary_use')
plt.show()


# In[14]:


train['Weekday'] = train['timestamp'].dt.weekday
train['Weekday_Name'] = train['timestamp'].dt.weekday_name
train['Month'] = train['timestamp'].dt.month
train['DayofYear'] = train['timestamp'].dt.dayofyear
train['Hour'] = train['timestamp'].dt.hour


# In[15]:


train['normalized_meter_reading_type'] =     train.groupby('meter_type')['meter_reading']         .transform(lambda x: (x - x.mean()) / x.std())


# In[16]:


fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data=train.groupby(['Weekday_Name','meter_type']).mean().reset_index(),
            x='Weekday_Name',
            y='normalized_meter_reading_type',
            hue='meter_type',
            ax=ax)
plt.title('Day of Week vs. Normalized Meter Reading')
plt.show()


# In[17]:


fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data=train.groupby(['Month','meter_type']).mean().reset_index(),
            x='Month',
            y='normalized_meter_reading_type',
            hue='meter_type',
            ax=ax)
plt.title('Month vs. Normalized Meter Reading')
plt.show()


# In[18]:


fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data=train.groupby(['Hour','meter_type']).mean().reset_index(),
            x='Hour',
            y='normalized_meter_reading_type',
            hue='meter_type',
            ax=ax)
plt.title('Hour within Day vs. Normalized Meter Reading')
plt.show()


# In[19]:


fig, ax = plt.subplots(figsize=(15, 5))
sns.lineplot(data=train.groupby(['DayofYear','meter_type']).mean().reset_index(),
            x='DayofYear',
            y='normalized_meter_reading_type',
            hue='meter_type',
            ax=ax)
# plt.title('Day of Year vs. Normalized Meter Reading')
plt.show()

