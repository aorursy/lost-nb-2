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
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing, Holt
from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
)
import gc # garbage collector

# stats models
import statsmodels.api as sm
from fbprophet import Prophet

# time libraries
import datetime

# warning libraries for debugging
import warnings

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


import time, sys
from IPython.display import clear_output

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


# In[3]:


get_ipython().run_line_magic('time', "train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')")
get_ipython().run_line_magic('time', "test_df = pd.read_csv('../input/ashrae-energy-prediction/test.csv')")


# In[4]:


# Saving some memory
d_types = {'building_id': np.int16,
          'meter': np.int8}

for feature in d_types:
    train_df[feature] = train_df[feature].astype(d_types[feature])
    
    
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], infer_datetime_format=True)


# In[5]:


train_df["log_meter_reading"]=np.log(train_df["meter_reading"]+.00001)


# In[6]:


def rmsle(pred_series,true_series):
    sum_series = (np.log(pred_series+1) -         np.log(true_series+1))**2
    return np.sqrt(np.sum(sum_series))


# In[7]:


start_validation='2016-12-15'
train = train_df.loc[train_df["timestamp"]<start_validation,:]
valid = train_df.loc[train_df["timestamp"]>=start_validation,:]


# In[8]:


# for the training data I want to reformat
# the dataframe so that the timestamp is the 
# index
print("reformat training data frame...")
def trainDF2timeDF(training_df):
    timeValue_df =  training_df.copy()
    timeValue_df = timeValue_df.set_index("timestamp")
    warnings.simplefilter("ignore")
    timeValue_df.index = pd.to_datetime(timeValue_df.index.values)
    return(timeValue_df)

timeIndexed_train = trainDF2timeDF(train)


# In[9]:


# create new data frame for this model
valid_avgVal_df = valid.copy()
# rename timestamp to signify the current meter reading time
valid_avgVal_df = valid_avgVal_df.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})

# This model splits the data based on 
# building ID and model type
nbuildings=len(valid["building_id"].unique())
print("number of buildings: "+ str(nbuildings))
x=0
for b_id in list(valid["building_id"].unique()):
    update_progress(x / nbuildings)
    x+=1
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
update_progress(1)


# In[10]:


b_i=1
m_t=0
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_avgVal_df.loc[(
    (valid_avgVal_df["building_id"]==b_i) &
    (valid_avgVal_df["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Simple Exponential Smoothing')
plt.legend(loc = 'best')


# In[11]:


print("Naive Approach - RMSLE value:")
print(rmsle(valid_avgVal_df["pred_meter_reading"],
           valid_avgVal_df["cur_meter_reading"]))


# In[12]:


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


# In[13]:


valid_expSmooth = valid.copy()
# rename timestamp to signify the current meter reading time
valid_expSmooth = valid_expSmooth.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})

# for the training data I want to reformat
# the dataframe so that the timestamp is the 
# index
print("reformat training data frame...")
def trainDF2timeDF(training_df):
    timeValue_df =  train.copy()
    timeValue_df = timeValue_df.set_index("timestamp")
    warnings.simplefilter("ignore")
    timeValue_df.index = pd.to_datetime(timeValue_df.index.values)
    return(timeValue_df)

timeIndexed_train = trainDF2timeDF(train)

# This model splits the data based on 
# building ID and model type
nbuildings=len(valid["building_id"].unique())
print("number of buildings: "+ str(nbuildings))
x=0
for b_id in list(valid["building_id"].unique()):
    update_progress(x / nbuildings)
    x+=1
    for meter_t in list(
        valid_expSmooth.loc[valid_expSmooth["building_id"]==b_id,"meter"].unique()):
        if(not ((b_id in train["building_id"]) and
           (meter_t in train.loc[train["building_id"]==b_id,"meter"].values))):
            print("missing!")
            print(b_id)
            print(meter_t)
            # if there is no meter reading for a specific
            # building ID then I'll just train
            # independent of the building ID
            sub_timeTrain_df = timeIndexed_train.loc[(
                timeIndexed_train["meter"]==meter_t),"meter_reading"].copy()
            numValid = len(valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),:])
            fit_simExpSmooth = SimpleExpSmoothing(sub_timeTrain_df).fit()
            # forecast the meter_readings
            valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),"pred_meter_reading"] = \
                fit_simExpSmooth.forecast(numValid).values
            # collect the alpha level used
            valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),"alpha"] = \
                fit_simExpSmooth.model.params['smoothing_level']
        else:
            # fit the model to the meter values of
            # this building type
            sub_timeTrain_df = timeIndexed_train.loc[(
                (timeIndexed_train["building_id"]==b_id) &
                (timeIndexed_train["meter"]==meter_t)),"meter_reading"].copy()
            numValid = len(valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),:])
            fit_simExpSmooth = SimpleExpSmoothing(sub_timeTrain_df).fit()
            # forecast the meter_readings
            valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),"pred_meter_reading"] = \
                fit_simExpSmooth.forecast(numValid).values
            # collect the alpha level used
            valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),"alpha"] = \
                fit_simExpSmooth.model.params['smoothing_level']
update_progress(1)


# In[14]:


b_i=0
m_t=0
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_expSmooth.loc[(
    (valid_expSmooth["building_id"]==b_i) &
    (valid_expSmooth["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Simple Exponential Smoothing')
plt.legend(loc = 'best')


# In[15]:


print("Simple Exponential Smoothing - RMSLE value:")
print(rmsle(valid_expSmooth["pred_meter_reading"],
           valid_expSmooth["cur_meter_reading"]))


# In[16]:


expSmooth_rmsle_list=[]
for meter_t in list(valid_expSmooth["meter"].unique()):
        sub_valid_expSmooth_df = valid_expSmooth.loc[(
            valid_expSmooth["meter"]==meter_t),:].copy()
        sub_rmsle = rmsle(sub_valid_expSmooth_df["pred_meter_reading"],
           sub_valid_expSmooth_df["cur_meter_reading"])
        sub_rmsle_df = pd.DataFrame({"meter":[meter_t],
                                   "rmsle":[sub_rmsle]})
        expSmooth_rmsle_list.append(sub_rmsle_df)
expSmooth_rmsle_df = pd.concat(expSmooth_rmsle_list)
expSmooth_rmsle_df  


# In[17]:


# create new data frame for this model
valid_holt = valid.copy()
# rename timestamp to signify the current meter reading time
valid_holt = valid_holt.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})


# This model splits the data based on 
# building ID and model type
nbuildings=len(valid["building_id"].unique())
print("number of buildings: "+ str(nbuildings))
x=0
for b_id in list(valid["building_id"].unique()):
    update_progress(x / nbuildings)
    x+=1
    for meter_t in list(
        valid_holt.loc[valid_holt["building_id"]==b_id,"meter"].unique()):
        if(not ((b_id in train["building_id"]) and
           (meter_t in train.loc[train["building_id"]==b_id,"meter"].values))):
            print("missing!")
            print(b_id)
            print(meter_t)
            # if there is no meter reading for a specific
            # building ID then I'll just train
            # independent of the building ID
            sub_timeTrain_df = timeIndexed_train.loc[(
                timeIndexed_train["meter"]==meter_t),"meter_reading"].copy()
            numValid = len(valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),:])
            fit_holt = Holt(
                sub_timeTrain_df).fit(optimized=True)
            # forecast the meter_readings
            valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),"pred_meter_reading"] = \
                fit_holt.forecast(numValid).values
            # collect the alpha level used
            valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),"alpha"] = \
                fit_holt.model.params['smoothing_level']
        else:
            # fit the model to the meter values of
            # this building type
            sub_timeTrain_df = timeIndexed_train.loc[(
                (timeIndexed_train["building_id"]==b_id) &
                (timeIndexed_train["meter"]==meter_t)),"meter_reading"].copy()
            numValid = len(valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),:])
            fit_holt = Holt(
                sub_timeTrain_df).fit(optimized=True)
            # forecast the meter_readings
            valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),"pred_meter_reading"] = \
                fit_holt.forecast(numValid).values
            # collect the alpha level used
            valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),"alpha"] = \
                fit_holt.model.params['smoothing_level']
update_progress(1)


# In[18]:


b_i=0
m_t=0
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_holt.loc[(
    (valid_holt["building_id"]==b_i) &
    (valid_holt["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Holt Model')
plt.legend(loc = 'best')


# In[19]:


b_i=161
m_t=1
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_holt.loc[(
    (valid_holt["building_id"]==b_i) &
    (valid_holt["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Holt Model')
plt.legend(loc = 'best')


# In[20]:


b_i=745
m_t=2
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_holt.loc[(
    (valid_holt["building_id"]==b_i) &
    (valid_holt["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Holt Model')
plt.legend(loc = 'best')


# In[21]:


b_i=106
m_t=3
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_holt.loc[(
    (valid_holt["building_id"]==b_i) &
    (valid_holt["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Holt Model')
plt.legend(loc = 'best')


# In[22]:


print("Holt - RMSLE value:")
print(rmsle(valid_holt["pred_meter_reading"],
           valid_holt["cur_meter_reading"]))


# In[23]:


holt_rmsle_list=[]
for meter_t in list(valid_holt["meter"].unique()):
        sub_valid_holt_df = valid_holt.loc[(
            valid_holt["meter"]==meter_t),:].copy()
        sub_rmsle = rmsle(sub_valid_holt_df["pred_meter_reading"],
           sub_valid_holt_df["cur_meter_reading"])
        sub_rmsle_df = pd.DataFrame({"meter":[meter_t],
                                   "rmsle":[sub_rmsle]})
        holt_rmsle_list.append(sub_rmsle_df)
holt_rmsle_list = pd.concat(holt_rmsle_list)
holt_rmsle_list  


# In[24]:


# Winter-Holt's prediction model
# parameters
# - train_dataframe: dataframe containing training data
# - timeIdx_train: data frame with time series in the index
# - valid_winterHolt: copy of the validation data frame
# - seasonility: a list of all the seasonal period that are being tested
# - b_id: building ID
# - meter_t: meter number
# - pred_col: list of the predicion column names (must be unique)
# - plot: set to True to print a plot of the predictions
def winHolt(train_dataframe, timeIdx_train, valid_winterHolt, 
            seasonality, b_id, meter_t, plot=False, 
            pred_col=[], known_true=False):
    if len(seasonality) > len(pred_col):
        if len(seasonality) ==1:
            pred_col=["winterHolt"]
        else:
            pred_col=[]
            for i in range(0,len(seasonality)):
                pred_col.append("winterHolt_sp_"+str(seasonality[i]))
    ignored_pred_cols=[]
    if(not ((b_id in train_dataframe["building_id"]) and
           (meter_t in train_dataframe.loc[train_dataframe["building_id"]==b_id,"meter"].values))):
        print("missing!")
        print(b_id)
        print(meter_t)
        # if there is no meter reading for a specific
        # building ID then I'll just train
        # independent of the building ID
        sub_timeTrain_df = timeIdx_train.loc[(
            timeIdx_train["meter"]==meter_t),"meter_reading"].copy()
        numValid = len(valid_winterHolt.loc[(
            (valid_winterHolt["building_id"]==b_id) &
            (valid_winterHolt["meter"]==meter_t)),:])
        for i in range(0,len(seasonality)):
            if (len(sub_timeTrain_df.index.unique())/
                 seasonality[i]) >= 2:
                fit_wintHolt = ExponentialSmoothing(
                    sub_timeTrain_df,
                    seasonal_periods=seasonality[i],
                    trend='add',
                    seasonal='add').fit()
                # forecast the meter_readings
                valid_winterHolt.loc[(
                    (valid_winterHolt["building_id"]==b_id) &
                    (valid_winterHolt["meter"]==meter_t)),pred_col[i]] = \
                    fit_wintHolt.forecast(numValid).values
            else:
                ignored_pred_cols.append(pred_col[i])
    else:
        # fit the model to the meter values of
        # this building type
        sub_timeTrain_df = timeIdx_train.loc[(
                        (timeIdx_train["building_id"]==b_id) &
                        (timeIdx_train["meter"]==meter_t)),"meter_reading"].copy()
        numValid = len(valid_winterHolt.loc[(
                        (valid_winterHolt["building_id"]==b_id) &
                        (valid_winterHolt["meter"]==meter_t)),:])
        for i in range(0,len(seasonality)):
            if (len(sub_timeTrain_df.index.unique())/
                 seasonality[i]) >= 2:
                fit_wintHolt = ExponentialSmoothing(
                    sub_timeTrain_df,
                    seasonal_periods=seasonality[i],
                    trend='add',
                    seasonal='add').fit()
                # forecast the meter_readings
                valid_winterHolt.loc[(
                    (valid_winterHolt["building_id"]==b_id) &
                    (valid_winterHolt["meter"]==meter_t)),pred_col[i]] = \
                    fit_wintHolt.forecast(numValid).values
            else:
                ignored_pred_cols.append(pred_col[i])
    if plot:
        b_i=b_id
        m_t=meter_t
        train_bidX_meterY = train_dataframe.loc[(
            (train_dataframe["building_id"]==b_i) &
            (train_dataframe["meter"]==m_t)),:].copy()
        
        valid_bidX_meterY = valid_winterHolt.loc[(
            (valid_winterHolt["building_id"]==b_i) &
            (valid_winterHolt["meter"]==m_t)),:].copy()
        plt.figure(figsize =(15,8))
        plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
        if known_true:
            plt.plot(valid_bidX_meterY['cur_meter_reading'], label = 'Validation')
        for i in range(0,len(seasonality)):
            if pred_col[i] not in ignored_pred_cols:
                plt.plot(valid_bidX_meterY[pred_col[i]], label = pred_col[i])
        plt.legend(loc = 'best')
    return(valid_winterHolt)    


# In[25]:


# create new data frame for this model
valid_winterHolt = valid.copy()
# rename timestamp to signify the current meter reading time
valid_winterHolt = valid_winterHolt.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})


# In[26]:


# set seasonality
sp=[365-90,4*9,9,3]


# This model splits the data based on 
# building ID and model type
nbuildings=len(valid["building_id"].unique())
print("number of buildings: "+ str(nbuildings))
x=0
for b_id in list(valid["building_id"].unique()):
    
    update_progress(x / nbuildings)
    x+=1
    for meter_t in list(
        valid_winterHolt.loc[valid_winterHolt["building_id"]==b_id,"meter"].unique()):
        print(b_id)
        print(meter_t)
        valid_winterHolt = winHolt(train,
            timeIndexed_train, valid_winterHolt,
            sp, b_id,meter_t)
update_progress(1)


# In[27]:


sp=[365-90,4*9,9,3]
b_id=0
meter_t=0
valid_winterHolt = winHolt(train, timeIndexed_train, valid_winterHolt, sp, b_id,meter_t, True, known_true=True)


# In[28]:


sp=[365-90,4*9,9,3]
b_id=161
meter_t=1
valid_winterHolt = winHolt(train, timeIndexed_train, valid_winterHolt, sp, b_id,meter_t, True, known_true=True)


# In[29]:


sp=[365-90,4*9,9,3]
b_id=745
meter_t=2
valid_winterHolt = winHolt(train, timeIndexed_train, valid_winterHolt, sp, b_id,meter_t, True, known_true=True)


# In[30]:


sp=[365-90,4*9,9,3]
b_id=106
meter_t=3
valid_winterHolt = winHolt(train, timeIndexed_train, valid_winterHolt, sp, b_id,meter_t, True, known_true=True)


# In[31]:


pred_col=[]
for i in sp:
    pred_colName = "winterHolt_sp_"+str(i)
    print("winterHolt (sp ="+str(i)+") - RMSLE value:")
    print(rmsle(valid_winterHolt[pred_colName],
           valid_winterHolt["cur_meter_reading"]))
    pred_col.append(pred_colName)


# In[32]:


winterHolt_rmsle_list=[]
for i in range(0,len(sp)):
    for meter_t in list(valid_winterHolt["meter"].unique()):
        sub_valid_winterHolt_df = valid_winterHolt.loc[(
            valid_winterHolt["meter"]==meter_t),:].copy()
        sub_rmsle = rmsle(sub_valid_winterHolt_df[pred_col[i]],
           sub_valid_winterHolt_df["cur_meter_reading"])
        sub_rmsle_df = pd.DataFrame({"seasonality":[sp[i]],
                                     "meter":[meter_t],
                                   "rmsle":[sub_rmsle]})
        winterHolt_rmsle_list.append(sub_rmsle_df)
winterHolt_rmsle_list = pd.concat(winterHolt_rmsle_list)
print(winterHolt_rmsle_list)  

