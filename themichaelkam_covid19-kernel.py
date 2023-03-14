#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels as sm
from datetime import timedelta
from sklearn.metrics import mean_squared_log_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv', header=0)
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv', header=0)


# In[3]:


train.columns


# In[4]:


test.columns


# In[5]:


def covid_curve_fit(train, country, state=None):
    result = {}
    if not state:
        temp = train.loc[train.Country_Region == country].sort_values(by='Date').reset_index()
    else:
        temp = train.loc[np.logical_and(train.Province_State == state, train.Country_Region == country)].sort_values(by='Date').reset_index()
    #with initial guess
    print(country, state)
    popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t), temp.index, temp.ConfirmedCases, p0=[5.7, 0], maxfev=2000)
    result['Country_Region'] = country
    result['Province_State'] = state
    result['a'] = popt[0]
    result['b'] = popt[1]
    result['pcov'] = pcov
    return result


# In[6]:


#TEST
country = 'Australia'
model = []

for state in ['New South Wales', 'Victoria']:
    res = covid_curve_fit(train, country, state)
    model.append(res)


# In[7]:


np.unique(train.loc[train['Country_Region'] == 'Afghanistan']['Province_State'].astype(str))


# In[8]:


countries = np.unique(train.Country_Region.astype(str))
confirmed_model = []

for country in countries:
    result = {}
    states = np.unique(train.loc[train.Country_Region == country].Province_State.astype(str))
    for state in states:
        if state != 'nan':
            result = covid_curve_fit(train, country, state)
        else:
            result = covid_curve_fit(train, country)
        confirmed_model.append(result)
    #if len(states) > 0:
    #    for state in states:
    #        result = covid_curve_fit(train, country, state)
    #        confirmed_model.append(result)
    #else:
    #    result = covid_curve_fit(train, country)
    #    confirmed_model.append(result)


# In[9]:


confirmed_model_pd = pd.DataFrame(confirmed_model)


# In[10]:


confirmed_model_pd.head(10)


# In[11]:


test_states = ['New York', 'New South Wales', 'British Columbia']
for i in test_states:
    predictions = model_pd.loc[model_pd.Province_State == i]['a'].values*np.exp(model_pd.loc[model_pd.Province_State == i]['b'].values*np.arange(0, 81))
    actual = train.loc[train.Province_State == i].reset_index()['ConfirmedCases']
    plt.plot(np.arange(0, 81), 
             predictions,
             'r-', label='Red: Fitted; Blue: Actual')
    plt.plot(actual)
    plt.title(i)
    plt.show()
    print('RMSLE:',np.sqrt(mean_squared_log_error(actual, predictions )))


# In[12]:


def fatality_curve_fit(train, country, state=None):
    result = {}
    if not state:
        temp = train.loc[train.Country_Region == country].sort_values(by='Date').reset_index()
    else:
        temp = train.loc[np.logical_and(train.Province_State == state, train.Country_Region == country)].sort_values(by='Date').reset_index()
    print(country, state)
    temp = temp.loc[temp.ConfirmedCases > 0]
    temp['Date'] = pd.to_datetime(temp.Date)
    temp['Date_20D'] = temp.Date + timedelta(days=20)
    temp_df = temp[['Province_State', 'Country_Region', 'Date', 'Fatalities']]        .merge(
        temp[['Province_State', 'Country_Region', 'Date_20D', 'ConfirmedCases']]\
        .rename(columns={'ConfirmedCases':'Confirmed20D', 'Date_20D':'Date'}), 
        on=['Province_State', 'Country_Region', 'Date'], how='inner')
    
    result['Country_Region'] = country
    result['Province_State'] = state
    
    #if more than 1 point we can use curve fit
    if temp_df.shape[0] > 1:
        popt, pcov = curve_fit(lambda x, a, b: a + b * np.log(x+1), temp_df.Confirmed20D, temp_df.Fatalities, maxfev=2000)    
        result['a'] = popt[0]
        result['b'] = popt[1]
        result['c'] = 0
        result['pcov'] = pcov
    #otherwise set the equation to 0.04 * number of confirmed in the past 20 days 
    else:
        result['a'] = 0
        result['b'] = 0
        result['c'] = 0.04
        result['pcov'] = []
    return result


# In[13]:


countries = np.unique(train.Country_Region.astype(str))

fatalities_model = []

for country in countries:
    result = {}
    states = np.unique(train.loc[train.Country_Region == country].Province_State.astype(str))
    for state in states:
        if state != 'nan':
            result = fatality_curve_fit(train, country, state)
        else:
            result = fatality_curve_fit(train, country)
        fatalities_model.append(result)
#     if len(states) > 0:
#         for state in states:
#             result = fatality_curve_fit(train, country, state)
#             fatalities_model.append(result)
#     else:
#         result = fatality_curve_fit(train, country)
#         fatalities_model.append(result)


# In[14]:


fatalities_model_pd = pd.DataFrame(fatalities_model)
fatalities_model_pd.head(5)


# In[15]:


model =[]
model.append(fatality_curve_fit(train, 'Canada', 'British Columbia'))
model = pd.DataFrame(model)
model


# In[16]:


temp = train.copy()
temp['Date'] = pd.to_datetime(temp.Date)
temp['Date_20D'] = temp.Date + timedelta(days=20)
temp_df = temp[['Province_State', 'Country_Region', 'Date', 'Fatalities']]        .merge(
        temp[['Province_State', 'Country_Region', 'Date_20D', 'ConfirmedCases']]\
        .rename(columns={'ConfirmedCases':'Confirmed20D', 'Date_20D':'Date'}), 
        on=['Province_State', 'Country_Region', 'Date'], how='inner')


# In[17]:


check = temp_df.loc[temp_df.Province_State=='British Columbia']
plt.plot(check['Confirmed20D'], check.Fatalities)
plt.plot(check['Confirmed20D'], np.maximum(model['a'].values + model['b'].values * np.log(check['Confirmed20D'] + 1) + model['c'].values * check['Confirmed20D'], np.zeros(len(check.Fatalities))))


# In[18]:


check_states = ['New York', 'New South Wales', 'British Columbia']
temp = train.loc[train.ConfirmedCases > 0].reset_index()
temp['Date'] = pd.to_datetime(temp.Date)
temp['Date_20D'] = temp.Date + timedelta(days=20)
temp_df = temp[['Province_State', 'Country_Region', 'Date', 'Fatalities']]        .merge(
        temp[['Province_State', 'Country_Region', 'Date_20D', 'ConfirmedCases']]\
        .rename(columns={'ConfirmedCases':'Confirmed20D', 'Date_20D':'Date'}), 
        on=['Province_State', 'Country_Region', 'Date'], how='inner')

for i in check_states:
    state_model = fatalities_model_df.loc[fatalities_model_df.Province_State == i]
    print(state_model['a'].values,state_model['b'].values, state_model['c'].values )
    check = temp_df.loc[temp_df.Province_State==i].reset_index()
    predictions = state_model['a'].values + state_model['b'].values * np.log(check['Confirmed20D'] + 1) + state_model['c'].values * check['Confirmed20D']
    actual = check['Fatalities']
    #print(check['Confirmed20D'])
    #print(predictions)
    plt.plot(check['Confirmed20D'], 
             predictions,
             'r-', label='Red: Fitted; Blue: Actual')
    plt.plot(check['Confirmed20D'], actual)
    plt.title(i)
    plt.show()
#    print('RMSLE:',np.sqrt(mean_squared_log_error(actual, predictions )))


# In[19]:


###Train Evaluate

train_pred = train.copy()
train_pred['Date'] = pd.to_datetime(train_pred.Date)
train_pred['Date20D'] = train_pred.Date + timedelta(days=20)
train_pred = train_pred[['Id','Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']]                .merge(train_pred.drop(columns=['Id','Fatalities', 'Date']).rename(columns={'ConfirmedCases':'Confirmed20D', 'Date20D':'Date'}),
                       on=['Province_State', 'Country_Region', 'Date'],
                       how='left'
                      )
train_pred['Confirmed20D'] = np.where(np.isnan(train_pred['Confirmed20D']), 0, train_pred['Confirmed20D'])
train_pred['DateId'] = (train_pred.Date - pd.to_datetime(np.min(train_pred.Date))).dt.days
train_pred = train_pred.merge(confirmed_model_pd.drop(columns=['pcov']), on = ['Country_Region', 'Province_State'], how='left')                       .merge(fatalities_model_pd.drop(columns=['pcov']), on = ['Country_Region', 'Province_State'], how='left', 
                             suffixes=('_conf', '_fat'))
train_pred['ConfirmedPred'] = np.round(train_pred['a_conf'] * np.exp(train_pred['b_conf'] * train_pred['DateId']))
train_pred['FatalitiesPred'] = np.round(np.maximum(train_pred['a_fat'] + train_pred['b_fat'] * np.log(train_pred['Confirmed20D'] + 1) + train_pred['c'] * train_pred['Confirmed20D'] , np.zeros(train_pred.shape[0])))
train_pred.head(5)


print('RMSLE:',np.sqrt(mean_squared_log_error(train_pred.ConfirmedCases, train_pred.ConfirmedPred)))
print('RMSLE:',np.sqrt(mean_squared_log_error(train_pred.Fatalities, train_pred.FatalitiesPred)))


# In[20]:


test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv', header=0)
test['Date'] = pd.to_datetime(test.Date)
test['Date20D'] = test.Date - timedelta(days=20)
test.head(5)


# In[21]:



#train.drop(columns=['Id','Fatalities']).rename(columns={'ConfirmedCases':'Confirmed20D', 'Date':'Date20D'}).head(5)
test['Date20D'] = test['Date20D'].astype(str)
test.merge(train.drop(columns=['Id','Fatalities']).rename(columns={'ConfirmedCases':'Confirmed20D', 'Date':'Date20D'}),
                       on=['Province_State', 'Country_Region', 'Date20D'],
                       how='left'
                      )


# In[22]:


test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv', header=0)
test['Date20D'] = (pd.to_datetime(test.Date) - timedelta(days=20)).astype(str)
public_sub = test.copy()
public_sub['DateId'] = (pd.to_datetime(public_sub.Date) - pd.to_datetime(np.min(train.Date))).dt.days
public_sub = public_sub.merge(confirmed_model_pd.drop(columns=['pcov']), on = ['Country_Region', 'Province_State'], how='left')                       .merge(fatalities_model_pd.drop(columns=['pcov']), on = ['Country_Region', 'Province_State'], how='left', 
                             suffixes=('_conf', '_fat'))
public_sub['ConfirmedCases'] = np.round(public_sub['a_conf'] * np.exp(public_sub['b_conf'] * public_sub['DateId']))

new_base = public_sub.copy()

public_sub = public_sub                .merge(train.drop(columns=['Id','Fatalities']).rename(columns={'ConfirmedCases':'Confirmed20D', 'Date':'Date20D'}),
                       on=['Province_State', 'Country_Region', 'Date20D'],
                       how='left'
                      )\
                .merge(new_base[['Province_State', 'Country_Region','Date', 'ConfirmedCases']]\
                       .rename(columns={'ConfirmedCases':'Confirmed20D', 'Date':'Date20D'}),
                       on=['Province_State', 'Country_Region', 'Date20D'],
                       how='left')

public_sub['Confirmed20D'] = np.where(np.isnan(public_sub['Confirmed20D_x']), public_sub['Confirmed20D_y'], public_sub['Confirmed20D_x'])
public_sub = public_sub.drop(columns=['Confirmed20D_x', 'Confirmed20D_y'])
public_sub['Confirmed20D'] = np.where(np.isnan(public_sub['Confirmed20D']), 0, public_sub['Confirmed20D'])
public_sub['Fatalities'] = np.round(np.maximum(public_sub['a_fat'] + public_sub['b_fat'] * np.log(public_sub['Confirmed20D'] + 1) + public_sub['c'] * public_sub['Confirmed20D'] , np.zeros(public_sub.shape[0])))
public_sub.head(10)


# In[23]:


public_sub.loc[public_sub.Date > '2020-04-14']


# In[24]:


public_sub.loc[public_sub.Date > '2020-05-01']


# In[25]:


submission_sample = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv', header=0)
print(submission_sample.columns)
print(submission_sample.head(5))


# In[26]:


public_sub[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', header=True, index=False)


# In[27]:


public_sub.shape


# In[ ]:




