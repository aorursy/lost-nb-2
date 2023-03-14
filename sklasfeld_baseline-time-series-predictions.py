#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import gc # garbage collector

# stats models
import statsmodels.api as sm


# deal with date in x-axis of plots
from pandas.plotting import register_matplotlib_converters

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
test_df = pd.read_csv('../input/ashrae-energy-prediction/test.csv')


# In[3]:


train_df.columns


# In[4]:


# Saving some memory
d_types = {'building_id': np.int16,
          'meter': np.int8}

for feature in d_types:
    train_df[feature] = train_df[feature].astype(d_types[feature])
    
    
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], infer_datetime_format=True)


# In[5]:


meter_mapping = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
train_df['meter_type'] = train_df['meter'].map(meter_mapping)
test_df['meter_type'] = test_df['meter'].map(meter_mapping)


# In[6]:


train_df["log_meter_reading"]=np.log(train_df["meter_reading"]+.00001)


# In[7]:


def rmsle(pred_series,true_series):
    sum_series = (np.log(pred_series+1) -         np.log(true_series+1))**2
    return np.sqrt(np.sum(sum_series))


# In[8]:


train = train_df.loc[train_df["timestamp"]<'2016-10-01',:]
valid = train_df.loc[train_df["timestamp"]>='2016-10-01',:]


# In[9]:


# create new data frame for this model
valid_1hrPrior_df = valid.copy()
# rename timestamp to signify the current meter reading time
valid_1hrPrior_df = valid_1hrPrior_df.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})

# get previous hour before validation set
prev_hour = pd.to_datetime("2016-10-01 00:00:00") -         pd.Timedelta(hours=1)

# This model splits the data based on 
# building ID and model type
for b_id in list(valid["building_id"].unique()):
    for meter_t in list(
        valid_1hrPrior_df.loc[valid_1hrPrior_df["building_id"]==b_id,"meter"].unique()):
        if(not ((b_id in train["building_id"]) and
           (meter_t in train.loc[train["building_id"]==b_id,"meter"].values) and
           ((train.loc[(
               (train["building_id"]==b_id) & 
               (train["meter"]==meter_t)),"timestamp"] == prev_hour).any()))):
            print("missing!")
            print(b_id)
            print(meter_t)
            # if there is no meter reading for a specific
            # building ID in the previous hour
            # then I'll just set the reading
            # to 0
            valid_1hrPrior_df.loc[((valid_1hrPrior_df["building_id"]==b_id) &
                valid["meter"]==meter_t),"pred_meter_reading"] = 0.0
        else:
            valid_1hrPrior_df.loc[((valid_1hrPrior_df["building_id"]==b_id) &
                valid_1hrPrior_df["meter"]==meter_t),"pred_meter_reading"] = \
                train.loc[(
                (train["building_id"]==b_id) &
                (train["meter"]==meter_t) &
                (train["timestamp"]==prev_hour)),"meter_reading"].values[0]

        


# In[10]:


print("Naive Approach - RMSLE value:")
print(rmsle(valid_1hrPrior_df["pred_meter_reading"],
           valid_1hrPrior_df["cur_meter_reading"]))


# In[11]:


# create new data frame for this model
valid_avgVal_df = valid.copy()
# rename timestamp to signify the current meter reading time
valid_avgVal_df = valid_avgVal_df.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})

# This model splits the data based on 
# building ID and model type
for b_id in list(valid["building_id"].unique()):
    for meter_t in list(
        valid_avgVal_df.loc[valid_avgVal_df["building_id"]==b_id,"meter"].unique()):
        if(not ((b_id in train["building_id"]) and
           (meter_t in train.loc[train["building_id"]==b_id,"meter"].values))):
            print("missing!")
            print(b_id)
            print(meter_t)
            # if there is no meter reading for a specific
            # building ID then I'll just set the reading
            # to the average value of that meter given
            # all of the building IDs.
            valid.loc[((valid_avgVal_df["building_id"]==b_id) &
                valid_avgVal_df["meter"]==meter_t),"pred_meter_reading"] = \
                train.loc[train["meter"]==meter_t,"meter_reading"].mean()
        else:
            # calculate the average meter_reading values
            # for each meter given the building id
            valid_avgVal_df.loc[((valid_avgVal_df["building_id"]==b_id) &
                valid_avgVal_df["meter"]==meter_t),"pred_meter_reading"] = \
                train.loc[(
                (train["building_id"]==b_id) &
                (train["meter"]==meter_t)),"meter_reading"].mean()


# In[12]:


print("Naive Approach - RMSLE value:")
print(rmsle(valid_avgVal_df["pred_meter_reading"],
           valid_avgVal_df["cur_meter_reading"]))


# In[13]:


avgVal_rmsle_list=[]
for meter_t in list(valid_avgVal_df["meter"].unique()):
        sub_valid_avgVal_df = valid_avgVal_df.loc[(
            valid_avgVal_df["meter"]==meter_t),:].copy()
        sub_rmsle = rmsle(sub_valid_avgVal_df["pred_meter_reading"],
           sub_valid_avgVal_df["cur_meter_reading"])
        sub_rmsle_df = pd.DataFrame({"meter":[meter_t],
                                   "rmsle":[sub_rmsle]})
        avgVal_rmsle_list.append(sub_rmsle_df)
avgVal_rmsle_df = pd.concat(avgVal_rmsle_list)
avgVal_rmsle_df  

