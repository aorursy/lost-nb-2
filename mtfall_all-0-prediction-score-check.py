#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize

import os
import gc

from sklearn.metrics import mean_squared_error


# In[2]:


def load_df(csv_path, chunksize=100000):
    features = ['date', 'fullVisitorId', 'totals_transactionRevenue']
    JSON_COLS = ['totals']
    print('Load {}'.format(csv_path))
    df_reader = pd.read_csv(csv_path,
                            converters={ column: json.loads for column in JSON_COLS },
                            dtype={ 'date': str, 'fullVisitorId': str},
                            usecols=['date', 'fullVisitorId', 'totals'], 
                            chunksize=chunksize)
    res = pd.DataFrame()
    for cidx, df in enumerate(df_reader):
        df.reset_index(drop=True, inplace=True)
        for col in JSON_COLS:
            col_as_df = json_normalize(df[col])
            col_as_df.columns = ['{}_{}'.format(col, subcol) for subcol in col_as_df.columns]
            df = df.drop(col, axis=1).merge(col_as_df, right_index=True, left_index=True)
        res = pd.concat([res, df[features]], axis=0).reset_index(drop=True)
        del df
        gc.collect()
        print('{}: {}'.format(cidx + 1, res.shape))
    return res


# In[3]:


train = load_df('../input/train_v2.csv')
test = load_df('../input/test_v2.csv')


# In[4]:


full_df = train.append(test).reset_index(drop=True)
full_df['date'] = pd.to_datetime(full_df['date'])
full_df.loc[:, full_df.columns.str.startswith('totals_')] = full_df.loc[:, full_df.columns.str.startswith('totals_')].astype(float).fillna(0)


# In[5]:


full_df.head()


# In[6]:


def get_target_fullvisitorid(full_df, target_datestart):
    target_fullvisitorid = pd.DataFrame(
        full_df.loc[
            (full_df['date'] >= pd.to_datetime(target_datestart)-pd.DateOffset(214)) &  
            (full_df['date'] < pd.to_datetime(target_datestart)-pd.DateOffset(45)), 
            'fullVisitorId'
        ].unique(), 
        columns=['fullVisitorId']).reset_index(drop=True)
    return target_fullvisitorid


# In[7]:


test_ids = get_target_fullvisitorid(full_df, '2018-12-01')
ss = pd.read_csv('../input/sample_submission_v2.csv', dtype={'fullVisitorId': str},)


# In[8]:


print((~test_ids['fullVisitorId'].isin(ss['fullVisitorId'])).sum())


# In[9]:


print((~ss['fullVisitorId'].isin(test_ids['fullVisitorId'])).sum())


# In[10]:


def make_groundtruth(target_datestart):
    target_fullvisitorid = get_target_fullvisitorid(full_df, target_datestart)
    
    date_range_for_groundtruth = [pd.to_datetime(target_datestart), pd.to_datetime(target_datestart)+pd.DateOffset(61)]
    
    exist_users_groundtruth = full_df.loc[
        (full_df['date'] >= date_range_for_groundtruth[0]) &  
        (full_df['date'] < date_range_for_groundtruth[1]) &
        (full_df['fullVisitorId'].isin(target_fullvisitorid['fullVisitorId'])), 
        ['fullVisitorId', 'totals_transactionRevenue']
    ].groupby('fullVisitorId')['totals_transactionRevenue'].sum()\
    .to_frame(name='LogSumRevenue').apply(np.log1p)
    full_groundtruth = target_fullvisitorid.merge(exist_users_groundtruth.reset_index(), on='fullVisitorId', how='left').set_index('fullVisitorId').fillna(0)
        
    return date_range_for_groundtruth, full_groundtruth


# In[11]:


for target_datestart in ['2017-12-01', '2018-01-01', '2018-02-01', 
                         '2018-03-01', '2018-04-01', '2018-05-01', 
                         '2018-06-01', '2018-07-01', '2018-08-01']:
    date_range_for_groundtruth, groundtruth = make_groundtruth(target_datestart)
    all0_pred = groundtruth.assign(pred=0)[['pred']]
    score = np.sqrt(mean_squared_error(groundtruth['LogSumRevenue'], all0_pred['pred']))
    print('validatin date range:{0} to {1} num of fullVisitorIds:{2} score:{3:.5f}'              .format(str(date_range_for_groundtruth[0].date()), str(date_range_for_groundtruth[1].date()), groundtruth.shape[0], score))

