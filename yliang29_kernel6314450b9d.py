#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gc
import sys
import math

from pandas.io.json import json_normalize
from datetime import datetime

import os
print(os.listdir("../input"))
['sample_submission_v2.csv', 'test_v2.csv', 'train_v2.csv']
gc.enable()

features = ['channelGrouping', 'date', 'fullVisitorId', 'visitId',       'visitNumber', 'visitStartTime', 'device.browser',       'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',       'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',       'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',       'geoNetwork.subContinent', 'totals.bounces', 'totals.hits',       'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue',       'trafficSource.adContent', 'trafficSource.campaign',       'trafficSource.isTrueDirect', 'trafficSource.keyword',       'trafficSource.medium', 'trafficSource.referralPath',       'trafficSource.source', 'customDimensions']

def load_df(csv_path):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    ans = pd.DataFrame()
    dfs = pd.read_csv(csv_path, sep=',',
            converters={column: json.loads for column in JSON_COLUMNS}, 
            dtype={'fullVisitorId': 'str'}, # Important!!
            chunksize=100000,nrows=50000)
    for df in dfs:
        df.reset_index(drop=True, inplace=True)
        for column in JSON_COLUMNS:
            column_as_df = json_normalize(df[column])
            column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

        #print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
        use_df = df[features]
        del df
        gc.collect()
        ans = pd.concat([ans, use_df], axis=0).reset_index(drop=True)
        #print(ans.shape)
    return ans

train = load_df('../input/train_v2.csv')
test = load_df('../input/test_v2.csv')
train.head()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


    


# In[ ]:


for c in train.columns.values:
    if c not in test.columns.values: print(c)

        
train['totals.transactionRevenue'].fillna(0, inplace=True)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'].astype(float))
print(train['totals.transactionRevenue'].describe()) 

test['totals.transactionRevenue'] = np.nan
all_data = train.append(test, sort=False).reset_index(drop=True)
print(all_data.info())
all_data.head(20)


#null_cnt = train.isnull().sum().sort_values()
#print(null_cnt[null_cnt > 0])
   
  
        


# In[ ]:


null_cnt = train.isnull().sum().sort_values()
# print(null_cnt[null_cnt > 0])
# fillna object feature
for col in ['trafficSource.keyword',
            'trafficSource.referralPath',
            'trafficSource.adContent']:
    all_data[col].fillna('unknown', inplace=True)

# fillna numeric feature
all_data['totals.pageviews'].fillna(1, inplace=True)
all_data['totals.newVisits'].fillna(0, inplace=True)
all_data['totals.bounces'].fillna(0, inplace=True)
all_data['totals.pageviews'] = all_data['totals.pageviews'].astype(int)
all_data['totals.newVisits'] = all_data['totals.newVisits'].astype(int)
all_data['totals.bounces'] = all_data['totals.bounces'].astype(int)

# fillna boolean feature
all_data['trafficSource.isTrueDirect'].fillna(False, inplace=True)
# drop constant column
constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]
#for c in constant_column:
#    print(c + ':', train[c].unique())

print('drop columns:', constant_column)
all_data.drop(constant_column, axis=1, inplace=True)

# pickup any visitor
all_data[all_data['fullVisitorId'] == '7813149961404844386'].sort_values(by='visitNumber')[
    ['date','visitId','visitNumber','totals.hits','totals.pageviews']]
train_rev = train[train['totals.transactionRevenue'] > 0].copy()
print(len(train_rev))
# train_rev.head()


vals = train_rev['customDimensions'].value_counts()
vals.head()










# In[ ]:


subA=all_data.loc[all_ data['customDimensions'].isin(vals.index.values),'customDimensions']
subA.head(10)

