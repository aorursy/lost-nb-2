#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/ashrae-energy-prediction'):
    for filename in filenames:
        #df_name = os.path.splitext(filename)[0]
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from time import time
import datetime
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',1500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from collections import Counter 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score,                             f1_score, roc_curve, confusion_matrix


# In[3]:


# importing datas of weather and simultaneously determine the datatypes:

weather_dtype = {"site_id":"uint8",'air_temperature':"float16",'cloud_coverage':"float16",'dew_temperature':"float16",'precip_depth_1_hr':"float16",
                 'sea_level_pressure':"float32",'wind_direction':"float16",'wind_speed':"float16"}

df_weather_train=pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv', parse_dates=['timestamp'],dtype=weather_dtype)
df_weather_test=pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv', parse_dates=['timestamp'],dtype=weather_dtype)


# importing datas of building characteristics. 

metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'int','year_built':'float32','floor_count':"float16"}
df_buildings=pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv', dtype=metadata_dtype)



# importing train-data

train_dtype = {'meter':"uint8",'building_id':'uint16','meter_reading':"float32"}
df_train=pd.read_csv('../input/ashrae-energy-prediction/train.csv',parse_dates=['timestamp'],dtype=train_dtype)


# importing test-data

df_test=pd.read_csv('../input/ashrae-energy-prediction/test.csv',parse_dates=['timestamp'],dtype=train_dtype)


print("data loaded")


# In[4]:


# find missing values in dataframes:
def missing_values(df):
    return pd.DataFrame(df.isna().sum()/len(df),columns=["% NANs"])


# In[5]:


df_weather_train.isnull().sum(axis = 0)


# In[6]:


df_weather_train.describe()


# In[7]:


display(df_buildings.describe())
display(missing_values(df_buildings))
df_buildings.info(memory_usage='deep')


# In[8]:


display(df_train.head())
display(missing_values(df_train))
display(df_train.describe())
df_buildings[df_buildings['building_id']==1099]
df_buildings[df_buildings['building_id']==1099]


# In[9]:


display(df_train[(df_train['building_id'] == 1099) & (df_train['meter'] == 2)]['meter_reading'].describe())
df_buildings[df_buildings['building_id']==1099]


# In[10]:


# dataset without rows with building1099
tt = df_train[df_train.building_id != 1099]
q75 = tt[tt['meter'] == 2]['meter_reading'].quantile(0.75)
q25 = tt[tt['meter'] == 2]['meter_reading'].quantile(0.25)

IQR = q75-q25
lowerIQR = q25 - 1.5*(IQR)
upperIQR = q75 + 1.5*(IQR)
print(lowerIQR, upperIQR)
print(int(21904700/2483.25))


# In[11]:


ajut=df_train[(df_train['building_id'] == 1099) & (df_train['meter'] == 2)]
ajut_e=df_train[(df_train['building_id'] == 1099) & (df_train['meter'] == 0)]

plt.plot(ajut_e['timestamp'].dt.date, ajut_e['meter_reading'])
plt.plot(ajut['timestamp'].dt.date, ajut['meter_reading']/8820)
plt.show()


# In[12]:


# look closer at site 0 energuy consumtion

df_train['meter_reading'] = np.log1p(df_train['meter_reading'])


site0_bds = list(df_buildings[df_buildings['site_id']==0]['building_id'])
plt.figure(figsize=(10,6))
for i in site0_bds:
    temp_df = df_train[df_train['building_id'] == i]
    plt.scatter(temp_df['timestamp'].dt.date, temp_df['meter_reading'], marker='.')

plt.show()


# In[13]:


sns.distplot(df_train[df_train['meter'] == 0]['meter_reading'],kde=False, label="Electricity")
sns.distplot(df_train[df_train['meter'] == 1]['meter_reading'],kde=False, label="ChilledWater")
sns.distplot(df_train[df_train['meter'] == 2]['meter_reading'],kde=False, label="Steam")
sns.distplot(df_train[df_train['meter'] == 3]['meter_reading'],kde=False, label="HotWater")
plt.title("Distribution of Log of Meter Reading Variable")
plt.legend()
plt.show()


# In[14]:


btypes = Counter(df_buildings['primary_use'])
building_meters={}
for b in btypes.keys():
    building_meters[b] = df_buildings[df_buildings['primary_use']==b]['building_id'].unique().tolist()

df_train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

for btype, b_list in building_meters.items():
    #print(Counter([df_train[df_train['building_id']==b]['meter'] for b in b_list])
    temp = Counter([df_train[df_train['building_id']==int(b)]['meter'].unique()[0] for b in b_list])
    print(btype, dict(temp))


# In[15]:


#import train data again:

df_train=pd.read_csv('../input/ashrae-energy-prediction/train.csv',parse_dates=['timestamp'],dtype=train_dtype)


# In[16]:


get_ipython().run_cell_magic('time', '', '\n# remove weird date data from site 0:\nto_del = df_train[(df_train[\'building_id\'] <= 104) & (df_train[\'timestamp\'] <= "2016-05-20")].index\ndf_train=df_train.drop(to_del, axis=0)\n\n\n# Fix format error for "Energy" in site 0:\n#   Site 0: Multiply by 0.2931 to get to model inputs into kWh like the other sites, and 3.4118 to get back to kBTU for scoring.\ndf_train.loc[(\n    df_train[\'building_id\'] <=104) & (df_train[\'meter\'] == 0), \'meter_reading\'] *= 0.2931\n\n# reduce abnormaly high "steam" values for building 1099 so that their max_val is out of outlier border \n#  (calculated from all "steam" values excluding this one)\ndf_train.loc[(\n    df_train[\'building_id\'] == 1099) & (df_train[\'meter\'] == 2), \'meter_reading\'] /= 8744\n\n\n# now convert meter values to log1p: # need to convert back later?\ndf_train[\'meter_reading\'] = np.log1p(df_train[\'meter_reading\'])\n\n\n# Weather: remove \'precip_depth_1_hr\'\ndf_weather_train.drop(\'precip_depth_1_hr\',axis=1,inplace=True)\ndf_weather_test.drop(\'precip_depth_1_hr\',axis=1,inplace=True)\n\n\n# BUILDINGS: remove \'floor_count\' and \'year_built\'\ndf_buildings.drop(\'floor_count\',axis=1,inplace=True)\ndf_buildings.drop(\'year_built\',axis=1,inplace=True)\n\n# group least common building types under "Other"\ndf_buildings[\'primary_use\'].replace({\'Healthcare\':"Other",\n                                     \'Parking\':"Other",\n                                     \'Warehouse/storage\':"Other",\n                                     \'Manufacturing/industrial\':"Other", \n                                     \'Retail\':"Other",\n                                     \'Services\':"Other",\n                                     \'Technology/science\':"Other", \n                                     \'Food sales and service\':"Other",\n                                     \'Utility\':"Other", \n                                     \'Religious worship\':"Other"},inplace=True)\n\n\n#impute missing variables for weather (within the site-ids!):\ndef impute_cols(df):\n    \n    cols = df.columns\n    sites=list(Counter(df.site_id).values())\n    sites[0]=sites[0]-1\n    counter = 0\n    for i in sites:\n        df.loc[counter:counter+i, cols] = df.loc[counter:counter+i, cols].interpolate(axis=0)\n        counter+=i\n        \nimpute_cols(df_weather_train)\nimpute_cols(df_weather_test)\n\n\n\n#weather_test_df = weather_test_df.groupby(\'site_id\').apply(lambda group: group.interpolate(limit_direction=\'both\'))\n#df_weather_test.groupby(\'site_id\').apply(lambda group: group.isna().sum())\n#Counter(df_weather_train.isnull().any(axis=1))\n\ngc.collect()')


# In[17]:


get_ipython().run_cell_magic('time', '', '\n#Merge all datasets\n\n# for train:\ndf_train = pd.merge(df_train, df_buildings, on=\'building_id\', how=\'left\', copy=False)\ndf_train = pd.merge(df_train, df_weather_train, on=[\'site_id\', \'timestamp\'], how=\'left\', copy=False)\n#del(df_train["timestamp"])\nprint("trainig data shape:", df_train.shape)\n\n# for test:\ndf_test = pd.merge(df_test, df_buildings, on=\'building_id\', how=\'left\', copy=False)\ndf_test = pd.merge(df_test, df_weather_test, on=[\'site_id\', \'timestamp\'], how=\'left\', copy=False)\n#del(df_test["timestamp"])\n\n\n\n# site 8 has more data on more dates, so we need to trim off data from other sites at these dates:\ndf_train=df_train.dropna()   \ndf_test=df_test.dropna()  \nprint("NA values in train dataset:", dict(Counter(df_train.isnull().any(axis=1))))\nprint("NA values in test dataset:", dict(Counter(df_test.isnull().any(axis=1))))\n#df_train[df_train.isnull().any(axis=1)]\n\n\n# Generate time data from timestamp and delete the latter:\ndef preprocess(df):\n    df["hour"] = df["timestamp"].dt.hour  # test deleting\n    df["day"] = df["timestamp"].dt.day\n    df["month"] = df["timestamp"].dt.month\n    df["dayofweek"] = df["timestamp"].dt.dayofweek\n    df["weekend"] = df["dayofweek"] >= 5\n    del(df["timestamp"])\n    \n\npreprocess(df_train) \npreprocess(df_test)\n\ngc.collect()')


# In[18]:


print(df_test.shape)
print(df_train.shape)


# In[19]:


# extra preprocess:

def prepare_meter_data(metertype):

    # get indexes of rows with selected metertype 
    tr_rowids = df_train[df_train['meter'] == metertype].index
    ts_rowids = df_test[df_test['meter'] == metertype].index

    # slice out selected rows for train and test dataset separately
    df_train_mod = df_train.loc[tr_rowids].drop(['meter_reading'], axis=1)
    df_val_mod = df_train.loc[tr_rowids]['meter_reading']

    df_test_mod = df_test.loc[ts_rowids]

    # delete unnecesarry cols
    todrop = (["hour", "day", "weekend", "meter"])
    df_train_mod = df_train_mod.drop(todrop, axis = 1) 
    df_test_mod = df_test_mod.drop(todrop, axis = 1) 

    #one-hot encoding for cateorical variables
    df_train_mod = pd.get_dummies(df_train_mod, columns = ["month", "dayofweek", "primary_use"])
    df_test_mod = pd.get_dummies(df_test_mod, columns = ["month", "dayofweek", "primary_use"])
    
    return (df_train_mod, df_val_mod, df_test_mod)


# In[20]:


# As example, create train, val, test datsets for "steam"
train, val, test = prepare_meter_data(2)


# In[21]:


print(train.shape, val.shape, test.shape)


# In[22]:


'''

# convert all meter readings back freom log1p scale: 
df_test['meter_reading'] = np.expm1(df_test['meter_reading'])

# de-fix format error for "Energy" in site 0 ((Multiply model inputs 3.4118 to get back to kBTU for scoring)):
df_test.loc[(
    df_test['building_id'] <=104) & (df_test['meter'] == 0), 'meter_reading'] *= 3.4118

# de-fix abnormaly high "steam" values for building 1099 ((multiply values with 8744):
df_test.loc[(
    df_test['building_id'] == 1099) & (df_test['meter'] == 2), 'meter_reading'] *= 8744

'''

