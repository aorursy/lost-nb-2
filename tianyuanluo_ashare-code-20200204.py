#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, datetime, gc

import numpy as np, pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
train.head(10)


# In[3]:


train.info()


# In[4]:


test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
test.head()


# In[5]:


test.info()


# In[6]:


building_meta = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
building_meta.head()


# In[7]:


building_meta.info()


# In[8]:


weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")
weather_train.head()


# In[9]:


weather_train.info()


# In[10]:


weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")
weather_test.head()


# In[11]:


weather_test.info()


# In[12]:


# Building meta data contains null values.

print('Missing Value Percentage:\n',
     building_meta.isnull().sum() / building_meta.shape[0]) 
sns.heatmap(building_meta.isnull(), cbar = False, cmap = 'bwr')


# In[13]:


# shrink the size of DataFrame

def df_col_size(df):
    print('Total Size (Mb): {:.3f}'.format(df.memory_usage().sum() * 1e-6 ))
    print(df.memory_usage() * 1e-6)
    for col in df.columns:
        print(f"{col}: Type: {df[col].dtypes},        Min: {df[col].min()},        Max: {df[col].max()}")


# In[14]:


df_col_size(building_meta)


# In[15]:


from sklearn.preprocessing import LabelEncoder

use_le = LabelEncoder()

use_le.fit(building_meta.primary_use)
building_meta['primary_use'] = use_le.transform(building_meta.primary_use)

use_le.classes_


# In[16]:


use_le.classes_


# In[17]:


building_meta['site_id'] = building_meta['site_id'].astype(np.int8)
building_meta['building_id'] = building_meta['building_id'].astype(np.int16)
building_meta['square_feet'] = building_meta['square_feet'].astype(np.int32)
building_meta['primary_use'] = building_meta['primary_use'].astype(np.int8)
building_meta['year_built'] = building_meta['year_built'].astype(np.float16)
building_meta['floor_count'] = building_meta['floor_count'].astype(np.float16)


# In[18]:


df_col_size(building_meta)


# In[19]:


# How many unique site ID?

building_meta['site_id'].value_counts().sort_index().plot.bar(figsize = (8,6))


# In[20]:


# What's the missing rate per each site ID
building_meta.groupby('site_id').apply(lambda x: x.isnull().sum() / (x.count() + x.isnull().sum()))


# In[21]:


# Cannot impute missing values only based on site_id. Join the train data and impute all together.


# In[22]:


# weather is split into train and test data. We combine them together and then analyze.

print('size of train:', weather_train.shape,'\n',
     'size of test', weather_test.shape)

weather_all = pd.concat([weather_train, weather_test], axis = 0, keys = ('train', 'test'))

print('size of all', weather_all.shape)

del weather_train
del weather_test


# In[23]:


#weather data contains null values.

print('Missing Value Percentage:\n',
     weather_all.isnull().sum() / weather_all.shape[0])
plt.figure(figsize = (10,10))
sns.heatmap(weather_all.isnull(), cbar = False, cmap = 'bwr')


# In[24]:


# input data before shrinking the size
weather_all['timestamp'] = pd.to_datetime(weather_all.timestamp)

weather_all['year'] = weather_all['timestamp'].dt.year
weather_all['month'] = weather_all['timestamp'].dt.month
weather_all['day'] = weather_all['timestamp'].dt.day
weather_all['week'] = weather_all['timestamp'].dt.week
weather_all['weekday'] = weather_all['timestamp'].dt.weekday
weather_all['hour'] = weather_all['timestamp'].dt.hour
weather_all['season'] = np.where(weather_all['timestamp'].dt.month <=2, 4, 
         np.where(weather_all['timestamp'].dt.month <= 5, 1,
                 np.where(weather_all['timestamp'].dt.month <= 8, 2,
                         np.where(weather_all['timestamp'].dt.month <= 11, 3, 4))))


# In[25]:


# air_temp

print('NaN before impute: {}'.format(weather_all.air_temperature.isnull().sum()))

air_temp_filler = weather_all.groupby(['site_id', 'year','month','hour'])['air_temperature'].median()

for site_id, year, month, hour in air_temp_filler.index:
    weather_all.loc[(weather_all['site_id'] == site_id) & 
                    (weather_all['year'] == year) & 
                    (weather_all['month'] == month) & 
                    (weather_all['hour'] == hour) &
                    (weather_all.air_temperature.isnull()), 'air_temperature']\
    = air_temp_filler[(site_id, year, month, hour)]

print('NaN after impute: {}'.format(weather_all.air_temperature.isnull().sum()))


# In[26]:


weather_all.groupby(['site_id', 'year','month'])['air_temperature'].describe().head(12)


# In[27]:


# cloud_average

print(weather_all.cloud_coverage.isnull().sum(), 
      'Pecentage: {:.2f}'.format(weather_all.cloud_coverage.isnull().sum() / weather_all.shape[0]))


# In[28]:


# study the cloud coverage (take a small portion of data and study its pattern)

fig, ax = plt.subplots(len(set(weather_all['site_id'].values)),1, figsize = (20, 60))
idx = 0
for site_id in set(weather_all['site_id'].values):
    ax[idx].plot(weather_all[(weather_all['site_id'] == site_id) &
                             (weather_all['year'] == 2016)][['timestamp', 'cloud_coverage']]\
    .set_index('timestamp'))
    ax[idx].set_title('Site ID: {}'.format(site_id), fontsize = 10)
    idx += 1
    
    


# In[29]:


# site 7 and 11 did not have cloud_coverage data at all. Hence inpute the rest NaN's with -1.
weather_all.groupby(['site_id','year','day'])['cloud_coverage'].mean()[weather_all.groupby(['site_id','year','day'])['cloud_coverage'].mean().isnull()].index.tolist()


# In[30]:


# from the analysis, the cloud coverage should be similar for the same day of each month.

cloud_coverage_fillter = weather_all.groupby(['site_id','year','day'])['cloud_coverage'].median()

print("NaN before impute: {}".format(weather_all.cloud_coverage.isnull().sum()))

for site_id, year, day in cloud_coverage_fillter.index:
    weather_all.loc[(weather_all['site_id'] == site_id) & 
               (weather_all['year'] == year) & 
               (weather_all['day'] == day) & 
                    (weather_all.cloud_coverage.isnull()), 'cloud_coverage'] = cloud_coverage_fillter[(site_id, year, day)]

weather_all['cloud_coverage'].fillna(-999, inplace = True)
print("NaN after impute: {}".format(weather_all.cloud_coverage.isnull().sum()))


# In[31]:


weather_all.groupby(['site_id', 'year','day'])['cloud_coverage'].describe().head(12)


# In[32]:


# Temperaure should be the same for the same hour of each day within a specific month. 
# Therefore we impute the missing values this way.
    
dew_temperature_filler = weather_all.groupby(['site_id', 'year', 'month', 'hour'])['dew_temperature'].median()

print('NaN before impute: {}'.format(weather_all.dew_temperature.isnull().sum()))

for site_id, year, month, hour in dew_temperature_filler.index:
    weather_all.loc[(weather_all['site_id'] == site_id) & 
                    (weather_all['year'] == year) & 
                    (weather_all['month'] == month) & 
                    (weather_all['hour'] == hour) &
                    (weather_all.dew_temperature.isnull()), 'dew_temperature']\
    = dew_temperature_filler[(site_id, year, month, hour)]

print('NaN after impute: {}'.format(weather_all.dew_temperature.isnull().sum()))


# In[33]:


weather_all.groupby(['site_id', 'year','month'])['dew_temperature'].describe().head(12)


# In[34]:


# precip_depth_1_hr

print('Nan count:', weather_all.precip_depth_1_hr.isnull().sum(), 'Percentage: {:.2f}'.format(weather_all.precip_depth_1_hr.isnull().sum() / weather_all.shape[0]))


# In[35]:


# site_id 5, 12, 1 did not have any percipitation records. Therefore impute -999.
weather_all.groupby(['site_id','year','month']).precip_depth_1_hr.apply(lambda x: x.isnull().sum() / (x.count() + x.isnull().sum())).sort_values(ascending = False)[:108].index.tolist()


# In[36]:


print('NaN before impute: {}'.format(weather_all.precip_depth_1_hr.isnull().sum()))

precipt_depth_filler = weather_all.groupby(['site_id','year','month']).precip_depth_1_hr.median()

for site_id, year, month in precipt_depth_filler.index:
    weather_all.loc[(weather_all['site_id'] == site_id) & 
                    (weather_all['year'] == year) & 
                    (weather_all['month'] == month) & 
                    (weather_all.precip_depth_1_hr.isnull()), 'precip_depth_1_hr']\
    = precipt_depth_filler[(site_id, year, month)]

weather_all.precip_depth_1_hr.fillna(-999, inplace = True)

print('NaN after impute: {}'.format(weather_all.precip_depth_1_hr.isnull().sum()))


# In[37]:


# sea_level_pressure

print('Nan count:', weather_all.sea_level_pressure.isnull().sum(), 
      'Percentage: {:.2f}'.format(weather_all.sea_level_pressure.isnull().sum() / weather_all.shape[0]))


# In[38]:


# site 5 did not have sea level pressure data. therefore impute with -1
weather_all.groupby(['site_id', 'year','month']).sea_level_pressure.median()[weather_all.groupby(['site_id', 'year','month']).sea_level_pressure.median().isnull()]


# In[39]:


print('NaN before impute: {}'.format(weather_all.sea_level_pressure.isnull().sum()))

sea_level_filler = weather_all.groupby(['site_id','year', 'month']).sea_level_pressure.median()

for site_id, year, month in sea_level_filler.index:
    weather_all.loc[(weather_all['site_id'] == site_id) & 
                    (weather_all['year'] == year) & 
                    (weather_all['month'] == month) & 
                    (weather_all.sea_level_pressure.isnull()), 'sea_level_pressure']\
    = sea_level_filler[(site_id, year, month)]

weather_all.sea_level_pressure.fillna(-999, inplace = True)

print('NaN after impute: {}'.format(weather_all.sea_level_pressure.isnull().sum()))


# In[40]:


# wind_direction

print('Nan count:', weather_all.wind_direction.isnull().sum(), 
      'Percentage: {:.2f}'.format(weather_all.wind_direction.isnull().sum() / weather_all.shape[0]))


# In[41]:


print('NaN before impute: {}'.format(weather_all.wind_direction.isnull().sum()))

wind_direction_filler = weather_all.groupby(['site_id','year','month','hour']).wind_direction.median()

for site_id, year, month, hour in wind_direction_filler.index:
    weather_all.loc[(weather_all['site_id'] == site_id) & 
                    (weather_all['year'] == year) & 
                    (weather_all['month'] == month) &
                    (weather_all['hour'] == hour) &
                    (weather_all.wind_direction.isnull()), 'wind_direction']\
    = wind_direction_filler[(site_id, year, month, hour)]

print('NaN after impute: {}'.format(weather_all.wind_direction.isnull().sum()))


# In[42]:


# wind_speed

print('NaN before impute: {}'.format(weather_all.wind_speed.isnull().sum()))

wind_speed_filler = weather_all.groupby(['site_id','year','month','hour']).wind_speed.median()

for site_id, year, month, hour in wind_speed_filler.index:
    weather_all.loc[(weather_all['site_id'] == site_id) & 
                    (weather_all['year'] == year) & 
                    (weather_all['month'] == month) &
                    (weather_all['hour'] == hour) &
                    (weather_all.wind_speed.isnull()), 'wind_speed']\
    = wind_speed_filler[(site_id, year, month, hour)]

print('NaN after impute: {}'.format(weather_all.wind_speed.isnull().sum()))


# In[43]:


weather_all.describe()


# In[44]:


df_col_size(weather_all)


# In[45]:


weather_all['site_id'] = weather_all['site_id'].astype(np.int8)
weather_all['air_temperature'] = weather_all['air_temperature'].astype(np.float16)
weather_all['cloud_coverage'] = weather_all['cloud_coverage'].astype(np.float16)
weather_all['dew_temperature'] = weather_all['dew_temperature'].astype(np.float16)
weather_all['precip_depth_1_hr'] = weather_all['precip_depth_1_hr'].astype(np.float16)
weather_all['sea_level_pressure'] = weather_all['sea_level_pressure'].astype(np.float16)
weather_all['wind_direction'] = weather_all['wind_direction'].astype(np.float16)
weather_all['wind_speed'] = weather_all['wind_speed'].astype(np.float16)
weather_all['year'] = weather_all['year'].astype(np.float16)
weather_all['month'] = weather_all['month'].astype(np.float16)
weather_all['day'] = weather_all['day'].astype(np.float16)
weather_all['week'] = weather_all['week'].astype(np.float16)
weather_all['weekday'] = weather_all['weekday'].astype(np.float16)
weather_all['hour'] = weather_all['hour'].astype(np.float16)
weather_all['season'] = weather_all['season'].astype(np.float16)

df_col_size(weather_all)


# In[46]:


df_col_size(train)
train.head()


# In[47]:


train['building_id'] = train['building_id'].astype(np.int16)
train['meter'] = train['meter'].astype(np.int8)
train['timestamp'] = pd.to_datetime(train['timestamp'])

df_col_size(train)


# In[48]:


df_col_size(test)
test.head()


# In[49]:


test['building_id'] = test['building_id'].astype(np.int16)
test['meter'] = test['meter'].astype(np.int8)
test['timestamp'] = pd.to_datetime(test['timestamp'])

df_col_size(test)


# In[50]:


# Merge dataframes to prepare training dataset

print(train.building_id.nunique(), building_meta.building_id.nunique())


# In[51]:


train_df = train.merge(building_meta, how = 'inner', on = 'building_id' )

df_col_size(train_df)
train_df.head()


# In[52]:


train_df = train_df.merge(weather_all, how = 'left', on = ['site_id', 'timestamp'])

df_col_size(train_df)
train_df.head()

del train


# In[53]:


train_df.to_csv('../working/train_df.csv', index = False)


# In[54]:


gc.collect()


# In[55]:


train_df.columns


# In[56]:


id_col = ['site_id','building_id','meter']
train_col = ['primary_use','square_feet','year_built','floor_count',
            'air_temperature','cloud_coverage','dew_temperature',
            'precip_depth_1_hr','sea_level_pressure','wind_direction',
            'wind_speed','month','day','week','weekday','hour','season']
target_col = ['meter_reading']


# In[57]:


from sklearn.ensemble import GradiantBoostingRegressor

train_set = train_df[train_df.timestamp]


# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


from catboost import Pool, CatBoostRegressor
import math

class RMSLE():
    
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)
    
    def is_max_optimal(self):
        return False
    
    def evaluate(self, approxes, target, weight):
        '''
            approxes is list of indexed containers
            (containers with only __len__ and __getitem__ defined), one container
            per approx dimension. Each container contains floats.
            weight is one dimensional indexed container.
            target is float.
            weight parameter can be None.
            Returns pair (error, weights sum)
            
        '''
        
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        
        approx = approxes[0]
        
        error_sum = 0.0
        weight_sum = 0.0
        
        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += ((math.log(approx[i] + 1) - math.log(target[i] + 1))**2) * w

        return math.sqrt(error_sum / weight_sum), weight_sum        


# In[59]:


demo = train_df[(train_df.site_id == 1) &
        (train_df.building_id == 119) &
        (train_df.meter == 0)]

train_pool = Pool(data = demo.loc[demo.timestamp.dt.month <= 9, train_col],
                 label = demo.loc[demo.timestamp.dt.month <= 9, target_col],
                 cat_features = ['primary_use'])

validation_pool = Pool(data = demo.loc[demo.timestamp.dt.month > 9, train_col],
                 label = demo.loc[demo.timestamp.dt.month > 9, target_col],
                 cat_features = ['primary_use'])


# In[60]:


model = CatBoostRegressor(iterations = 1000, learning_rate = 0.1, depth = 6,
                         loss_function = 'RMSE',
                         eval_metric = RMSLE(),
                         random_seed = 1234,
                         use_best_model = True)

model.fit(train_pool, eval_set = validation_pool, silent = True, early_stopping_rounds = 50)


# In[61]:


DataFrame({'names': model.feature_names_,
           'importance': model.feature_importances_
          }).sort_values(by = 'importance', ascending = False)
           
       


# In[62]:


model.drop_unused_features()
model.feature_names_


# In[ ]:





# In[ ]:





# In[ ]:





# In[63]:


import pandas as pd
building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
sample_submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")
weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

