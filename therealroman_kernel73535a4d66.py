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


path = '/kaggle/input/covid19-global-forecasting-week-3'

train = pd.read_csv(os.path.join(path, 'train.csv'))
test = pd.read_csv(os.path.join(path, 'test.csv'))
subm = pd.read_csv(os.path.join(path, 'submission.csv'))


# In[3]:


valid_date = pd.to_datetime('2020-04-10')


# In[4]:


valid_date.month


# In[5]:


train['area'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)
test['area'] = test['Country_Region'].astype(str) + '_' + test['Province_State'].astype(str)
train['Date'] = pd.to_datetime(train['Date'])


# In[6]:


path = '/kaggle/input/start-index-to-fit'

import pickle
with open(os.path.join(path, 'dict_bst_ind.pickle'), 'rb') as f:
    dict_bst_ind = pickle.load(f)
with open(os.path.join(path, 'dict_bst_ind_Fat.pickle'), 'rb') as f:
    dict_bst_ind_fat = pickle.load(f)
with open(os.path.join(path, 'pop_dict.pickle'), 'rb') as f:
    pop_dict = pickle.load(f)


# In[7]:


def log_curve(x, x0, k, ymax):
    return ymax / (1 + np.exp(-k*(x-x0)))

def fit_predict(vals, len_test, best_ind, population, tp):
    vals = vals[best_ind:]
    if population == -1:
        if tp == 'cases':
            mx = 400000
        else:
            mx = 40000
        popt, pcov = curve_fit(log_curve, list(range(len(vals))), vals, 
                                   bounds=([0,0, vals[-1]],[np.inf, np.inf, mx]), 
                                   p0=[10,0.3,vals[-1]], maxfev=1000000)
    else:
        popt, pcov = curve_fit(log_curve, list(range(len(vals))), vals, 
                                   bounds=([0,0, vals[-1]],[np.inf, np.inf, population * 0.01]), 
                                   p0=[10,0.3,vals[-1]], maxfev=1000000)
    pred = []
    for x in range(len(vals)-13, len(vals)-13 + len_test):
        pred += [log_curve(x, popt[0], popt[1], popt[2])]
    return pred


# In[8]:


def get_train_piece(area, valid_date):
    data = train[(train.area == area) & (train.Date < valid_date)].reset_index()
    data = data[data['ConfirmedCases'] > 0].reset_index(drop = True)
    return data


# In[9]:


pred_df = pd.DataFrame()

for pred_area in tqdm(test.area.unique()):
    train_df = get_train_piece(pred_area, valid_date)
    len_train = train_df.shape[0]
    population = -1
    if pred_area in pop_dict:
        population = pop_dict[pred_area]
    test_df = test[test.area == pred_area].reset_index(drop = True)
    len_test = test_df.shape[0]
    
    ans = pd.DataFrame()
    ans['ForecastId'] = test_df['ForecastId'].values
    
    if pred_area not in dict_bst_ind:
        ans['ConfirmedCases'] = fit_predict(train_df['ConfirmedCases'].values, len_test, 0, population, 'cases')
    else:
        ans['ConfirmedCases'] = fit_predict(train_df['ConfirmedCases'].values, len_test, dict_bst_ind[pred_area], population, 'cases')
    
    if pred_area not in dict_bst_ind_fat:
        ans['Fatalities'] = fit_predict(train_df['Fatalities'].values, len_test, 0, population, 'fat')
    else:
        ans['Fatalities'] = fit_predict(train_df['Fatalities'].values, len_test, dict_bst_ind_fat[pred_area], population, 'fat')

    
    pred_df = pd.concat([pred_df, ans], axis = 0).reset_index(drop = True)


# In[10]:


pred_df.to_csv('submission.csv', index=False)


# In[11]:


pred_df


# In[12]:


pred_df.max()


# In[ ]:




