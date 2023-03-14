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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


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


# In[3]:


from pykalman import KalmanFilter


# In[4]:


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


# In[5]:


train_df = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
test_df = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

train_df["Date"] = pd.to_datetime(train_df["Date"], infer_datetime_format=True)
test_df["Date"] = pd.to_datetime(test_df["Date"], infer_datetime_format=True)


# In[6]:


# look at the top of the training data frame
train_df.head()


# In[7]:


target_values=["ConfirmedCases","Fatalities"]


# In[8]:


# breathe of the target values
train_df.describe()


# In[9]:


# number of nulls in the training set
# compared to the number of rows
print("number of rows in training set:")
print(len(train_df))
print("null values in each column:")
print(train_df.isnull().sum())


# In[10]:


print("number of unique countries in training data: %i" %
    train_df['Country_Region'].nunique())


# In[11]:


print("Of the countries with provinces, how many provinces do they have?")
print(train_df.loc[train_df['Province_State'].notnull(),:].     groupby('Country_Region')['Province_State'].nunique())


# In[12]:


print("Of the countries with provinces, do they have any rows with no province listed?")
print("How many rows do they have?")
countriesWithProvinces = list(train_df.loc[train_df['Province_State'].notnull(),"Country_Region"])
train_df.loc[((train_df["Country_Region"].isin(countriesWithProvinces))
              & (train_df["Province_State"].isnull())),"Country_Region"].value_counts()


# In[13]:


def location(country, province):
    if province == province:
        loc = ("%s, %s" % 
               (province, 
               country))
        return(loc)
    else:
        return(country)
    
train_df['location'] = train_df.apply(
    lambda x: location(x["Country_Region"],
                      x["Province_State"]), axis=1)
test_df['location'] = test_df.apply(
    lambda x: location(x["Country_Region"],
                      x["Province_State"]), axis=1)


# In[14]:


print ("Start Date:")
print (train_df['Date'].min())
print ("End Date:")
print (train_df['Date'].max())


# In[15]:


# number of dates for each country/provice
print(train_df.     groupby(['Country_Region','Province_State'])['Date'].       nunique().      reset_index()['Date'].unique())


# In[16]:


uniq_location=list(train_df["location"].unique())
for target in target_values:
    plt.figure(figsize =(15,8))
    plt.title(target)
    for l_id in uniq_location:
        train_locationX = train_df.loc[(
                train_df["location"]==l_id),:].copy()
        plt.plot(train_locationX["Date"],
                 train_locationX[target], 
                 label = l_id)    
    #plt.legend(loc = 'best')


# In[17]:


# look at the top of the training data frame
test_df.head()


# In[18]:


print ("Start Date:")
print (test_df['Date'].min())
print ("End Date:")
print (test_df['Date'].max())


# In[19]:


def rmsle(pred_series,true_series):
    sum_series = (np.log(pred_series+1) -         np.log(true_series+1))**2
    return np.sqrt(np.sum(sum_series))


# In[20]:


start_validation='2020-03-19'
train = train_df.loc[train_df["Date"]<start_validation,:]
valid = train_df.loc[train_df["Date"]>=start_validation,:]


# In[21]:


# for the training data I want to reformat
# the dataframe so that the timestamp is the 
# index
print("reformat training data frame...")
def trainDF2timeDF(training_df):
    timeValue_df =  train.copy()
    timeValue_df = timeValue_df.set_index("Date")
    warnings.simplefilter("ignore")
    timeValue_df.index = pd.to_datetime(timeValue_df.index.values)
    return(timeValue_df)

timeIndexed_train = trainDF2timeDF(train)
timeIndexed_train_df = trainDF2timeDF(train_df)


# In[22]:


valid_holt = valid.copy().rename(
    columns={"timestamp": "now", 
             "ConfirmedCases": "true_ConfirmedCases",
            "Fatalities":"true_Fatalities"})


# In[23]:


holt_params={}
holt_params["damped_False"]=[False]
holt_params["damped_True"]=[True]


# In[24]:


# This model splits the data based on 
# location
uniq_location=list(valid["location"].unique())
nlocations=len(uniq_location)
print("number of locations: "+ str(nlocations))
x=0
for l_id in uniq_location:
    update_progress(x / nlocations)
    x+=1
    # fit the model to the target_values of this location
    for target in target_values:
        sub_timeTrain_df = timeIndexed_train.loc[(
            timeIndexed_train["location"]==l_id),target].copy()
        numValid = len(valid_holt.loc[(
            valid_holt["location"]==l_id),:])
        for param in holt_params.keys():
            fit_holt = Holt(
                sub_timeTrain_df,
                damped=holt_params[param][0]).fit(optimized=True)
            # forecast the targets
            target_col = ("%s_%s" %
                         (param,target))
            valid_holt.loc[(
                valid_holt["location"]==l_id),target_col] = \
                fit_holt.forecast(numValid).values
            alpha_col = (("%s_alpha") % param)
            valid_holt.loc[(
                valid_holt["location"]==l_id),alpha_col] = \
                    fit_holt.model.params['smoothing_level']
update_progress(1)


# In[25]:


# Ignore this code. 
# I just use it when I am too lazy to wait for the plots below
l_id="France"
if 1==0:
    for target in target_values:
        train_bidX_meterY = train.loc[(
                train["location"]==l_id),:].copy()
        valid_bidX_meterY = valid.loc[(
                valid["location"]==l_id),:].copy()
        pred_bidX_meterY = valid_holt.loc[(
                valid_holt["location"]==l_id),:].copy()
        plt.figure(figsize =(15,8))
        plt.title(l_id+" "+target)
        plt.plot(train_bidX_meterY["Date"],
                 train_bidX_meterY[target], 
                 label = 'Train')
        plt.plot(valid_bidX_meterY["Date"],
                 valid_bidX_meterY[target],
                 label = 'Validation')
        plt.plot(pred_bidX_meterY["Date"],
                pred_bidX_meterY["damped_False_"+target],
                 label = 'Holt Model (damped=False)')
        plt.plot(pred_bidX_meterY["Date"],
                pred_bidX_meterY["damped_True_"+target],
                 label = 'Holt Model (damped=True)')
        plt.legend(loc = 'best')


# In[26]:


for target in target_values:
    for l_id in uniq_location:
        train_bidX_meterY = train.loc[(
                train["location"]==l_id),:].copy()
        valid_bidX_meterY = valid.loc[(
                valid["location"]==l_id),:].copy()
        pred_bidX_meterY = valid_holt.loc[(
                valid_holt["location"]==l_id),:].copy()
        plt.figure(figsize =(15,8))
        plt.title(l_id+" "+target)
        plt.plot(train_bidX_meterY["Date"],
                 train_bidX_meterY[target], 
                 label = 'Train')
        plt.plot(valid_bidX_meterY["Date"],
                 valid_bidX_meterY[target],
                 label = 'Validation')
        plt.plot(pred_bidX_meterY["Date"],
                pred_bidX_meterY["damped_False_"+target],
                 label = 'Holt Model (damped=False)')
        plt.plot(pred_bidX_meterY["Date"],
                pred_bidX_meterY["damped_True_"+target],
                 label = 'Holt Model (damped=True)')
        plt.legend(loc = 'best')


# In[27]:


for target in target_values:
    print("Holt (damped=False) RMSLE value for %s:" % target)
    print(rmsle(valid_holt["damped_False_"+target],
               valid_holt["true_"+target]))
    print("Holt (damped=True) RMSLE value for %s:" % target)
    print(rmsle(valid_holt["damped_True_"+target],
               valid_holt["true_"+target]))


# In[28]:


# This model splits the data based on 
# location
nlocations=len(uniq_location)
print("number of locations: "+ str(nlocations))
x=0
for l_id in uniq_location:
    update_progress(x / nlocations)
    x+=1
    # fit the model to the target_values of this location
    for target in target_values:
        sub_timeTrain_df = timeIndexed_train_df.loc[(
            timeIndexed_train_df["location"]==l_id),target].copy()
        numValid = len(test_df.loc[(
            test_df["location"]==l_id),:])
        fit_holt = Holt(
            sub_timeTrain_df,
            damped=False).fit(optimized=True)
        # forecast the targets
        test_df.loc[(
            test_df["location"]==l_id),target] = \
            fit_holt.forecast(numValid).values
update_progress(1)


# In[29]:


submission = test_df.loc[:,["ForecastId","ConfirmedCases","Fatalities"]]
submission.to_csv("submission_holt_dampedFalse.csv",sep=",",index=False)

