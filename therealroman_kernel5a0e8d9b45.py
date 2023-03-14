#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta 
from tqdm import tqdm_notebook as tqdm
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


path = '/kaggle/input/covid19-global-forecasting-week-2'

train = pd.read_csv(os.path.join(path, 'train.csv'))
test = pd.read_csv(os.path.join(path, 'test.csv'))
subm = pd.read_csv(os.path.join(path, 'submission.csv'))


# In[3]:


valid_date = pd.to_datetime('2020-04-01')


# In[4]:


train['area'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)
test['area'] = test['Country_Region'].astype(str) + '_' + test['Province_State'].astype(str)
train['Date'] = pd.to_datetime(train['Date'])


# In[5]:


def get_train_piece(area, valid_date):
    data = train[(train.area == area) & (train.Date < valid_date)].reset_index()
    data = data[data['ConfirmedCases'] > 0].reset_index(drop = True)
    return data


# In[6]:


from sklearn.metrics import mean_squared_log_error


# In[7]:


train['ratio'] = train['Fatalities'] / train['ConfirmedCases']
gg = train.drop_duplicates('area', keep = 'last')
mean_fat = gg[gg['ConfirmedCases'] > 1000].ratio.mean()


# In[8]:


version = 1


# In[9]:


dict_kakaha = {'US_Puerto Rico':(30000, 80000), 'US_Idaho':(30000, 80000)}


# In[10]:


pred_df = pd.DataFrame()

for pred_area in tqdm(test.area.unique()):
    train_df = get_train_piece(pred_area, valid_date)
    len_train = train_df.shape[0]
    
    test_df = test[test.area == pred_area].reset_index(drop = True)
    len_test = test_df.shape[0]
    
    ans = pd.DataFrame()
    ans['ForecastId'] = test_df['ForecastId'].values
    
    if pred_area in dict_kakaha:
        def log_curve(x, x0, k):
            return dict_kakaha[pred_area][version] / (1 + np.exp(-k*(x-x0)))
        popt, pcov = curve_fit(log_curve, list(train_df.index), train_df['ConfirmedCases'].values, 
                               bounds=([0,0],np.inf), 
                               p0=[10,0.3], maxfev=1000000)
        pred = []
        pred_fat = []

        cur_fat = train_df['ratio'].values[-1]
        cur_rat = (cur_fat * train_df['ConfirmedCases'].values[-1] + 10 * mean_fat) / (train_df['ConfirmedCases'].values[-1] + 10)

        for x in range(len_train, len_train + len_test):
            pred += [log_curve(x, popt[0], popt[1])]
            pred_fat += [max(pred[-1] * cur_rat, train_df['Fatalities'].values[-1])]
        ans['ConfirmedCases'] = pred
        ans['Fatalities'] = pred_fat
    else:
        def log_curve(x, x0, k, ymax):
            return ymax / (1 + np.exp(-k*(x-x0)))
        popt, pcov = curve_fit(log_curve, list(train_df.index), train_df['ConfirmedCases'].values, 
                               bounds=([0,0, 0],[np.inf, np.inf, 150000]), 
                               p0=[10,0.3,10000], maxfev=1000000)
        pred = []
        pred_fat = []

        cur_fat = train_df['ratio'].values[-1]
        cur_rat = (cur_fat * train_df['ConfirmedCases'].values[-1] + 10 * mean_fat) / (train_df['ConfirmedCases'].values[-1] + 10)

        for x in range(len_train, len_train + len_test):
            pred += [log_curve(x, popt[0], popt[1], popt[2])]
            pred_fat += [max(pred[-1] * cur_rat, train_df['Fatalities'].values[-1])]
        ans['ConfirmedCases'] = pred
        ans['Fatalities'] = pred_fat
    
    pred_df = pd.concat([pred_df, ans], axis = 0).reset_index(drop = True)


# In[11]:


pred_df.tail(10)


# In[12]:


pred_df.to_csv('submission.csv', index=False)


# In[ ]:




