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


import csv
import os
import xgboost

import re
import string
from sklearn import ensemble
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
pyo.init_notebook_mode()


from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb


train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
df_1 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')


# In[3]:


print(train['Date'].min())
print(train['Date'].max())

print(test['Date'].min())
print(test['Date'].max())


# In[4]:


train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

train['dayofmonth'] = train['Date'].dt.day
train['dayofweek'] = train['Date'].dt.dayofweek
train['month'] = train['Date'].dt.month
train['weekNumber'] = train['Date'].dt.week
train['dayofyear'] = train['Date'].dt.dayofyear
## added in training set
train['Fatalities_ratio'] = train['Fatalities'] / train['ConfirmedCases']

#train['Change_ConfirmedCases'] = train.groupby('Country_Region').ConfirmedCases.pct_change()
#train['Change_Fatalities'] = train.groupby('Country_Region').Fatalities.pct_change()

## to deal with data wih Province State
train['Change_ConfirmedCases'] = train.groupby(np.where(train['Province_State'].isnull(), train['Country_Region'], train['Province_State'])).ConfirmedCases.pct_change()
train['Change_Fatalities'] = train.groupby(np.where(train['Province_State'].isnull(), train['Country_Region'], train['Province_State'])).Fatalities.pct_change()

## added in Test set
test['dayofmonth'] = test['Date'].dt.day
test['dayofweek'] = test['Date'].dt.dayofweek
test['month'] = test['Date'].dt.month
test['weekNumber'] = test['Date'].dt.week
test['dayofyear'] = test['Date'].dt.dayofyear


# In[5]:


train = train[train.Date<'2020-03-26']


# In[ ]:





# In[6]:


enriched = pd.read_csv("/kaggle/input/data-prep-week3/enriched_covid_19_week_3.csv")
enriched['Date'] = pd.to_datetime(train['Date'])
enriched['Date'] = pd.to_datetime(test['Date'])
enriched["quarantine"] = pd.to_datetime(enriched["quarantine"])
enriched["publicplace"] = pd.to_datetime(enriched["publicplace"])
enriched["gathering"] = pd.to_datetime(enriched["gathering"])
enriched["nonessential"] = pd.to_datetime(enriched["nonessential"])
enriched["schools"] = pd.to_datetime(enriched["schools"])
enriched["firstcase"] = pd.to_datetime(enriched["firstcase"])

dates_info = ["publicplace", "gathering", "nonessential", "quarantine", "schools","firstcase"]

enriched["age_40-59"] = enriched.loc[:,["age_40-44", "age_45-49", "age_50-54", "age_55-59"]].values.sum(1)
enriched["age_60-79"] = enriched.loc[:,["age_60-64", "age_65-69", "age_70-74", "age_75-79"]].values.sum(1)
enriched["age_80+"]  = enriched.loc[:,["age_80-84", "age_85-89", "age_90-94", "age_95-99","age_100+"]].values.sum(1)

enriched.drop([
    "age_0-4", "age_5-9", "age_10-14", "age_15-19", "age_20-24", "age_25-29", "age_30-34", "age_35-39",
    "age_40-44", "age_45-49", "age_50-54", "age_55-59", "age_60-64", "age_65-69", "age_70-74", "age_75-79",
    "age_80-84", "age_85-89", "age_90-94","age_95-99","age_100+", "femalelung", "malelung"], 
    axis = 1, inplace = True)

enriched.head()

enriched = enriched.iloc[:,:]
enriched.info()


# In[7]:


#enriched[enriched.Country_Region=='Ukraine'].info()


# In[8]:


def concat_country_province(country, province):
    if not isinstance(province, str):
        return country
    else:
        return country+"_"+province

# Concatenate region and province for training
train["Country_Region_"] = train[["Country_Region", "Province_State"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)
test["Country_Region_"] = test[["Country_Region", "Province_State"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)

enriched["Country_Region_"] = enriched[["Country_Region", "Province_State"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)
enriched = enriched.drop_duplicates(subset=['Country_Region_'], keep="first", inplace=False)


# In[9]:


train = train.merge(enriched.iloc[:, 6:], on ='Country_Region_', how='left')
test = test.merge(enriched.iloc[:, 6:], on ='Country_Region_', how='left')


# In[10]:


def dates_diff_days(date_curr, date_):
    if date_curr>date_:
        return (date_curr - date_).days
    else :
        return 0

for col in ["publicplace", "gathering", "nonessential", "quarantine", "schools"]:
    #print(merged.shape)
    train["days_to_"+ col] =train[[col, 'firstcase']].apply(lambda x : dates_diff_days(x[0], x[1]), axis=1)  
    test["days_to_"+ col] =test[[col, 'firstcase' ]].apply(lambda x : dates_diff_days(x[0], x[1]), axis=1) 


# train['days_to_quarantine'] =train[["quarantine", 'firstcase']].apply(lambda x : dates_diff_days(x[0], x[1]), axis=1)  
# test['days_to_quarantine'] =test[["quarantine", 'firstcase']].apply(lambda x : dates_diff_days(x[0], x[1]), axis=1)  


# In[11]:


for col in dates_info:
    #print(merged.shape)
    train[col] =train[["Date", col]].apply(lambda x : dates_diff_days(x[0], x[1]), axis=1)  
    test[col] =test[["Date", col]].apply(lambda x : dates_diff_days(x[0], x[1]), axis=1) 

print(test.shape)

#drop_country_cols = [x for x in merged.columns if x.startswith("country")] + dates_info


# In[12]:


# def to_log(x):
#     return np.log(x + 1)


# def to_exp(x):
#     return np.exp(x) - 1


# In[13]:


def process_each_location(df):
    dfs = []
    for loc, df in df.groupby('Country_Region_'):
        df = df.sort_values(by='Date')
#         df['Fatalities'] = df['Fatalities'].cummax()
#         df['ConfirmedCases'] = df['ConfirmedCases'].cummax()
#         df['LogFatalities'] = df['LogFatalities'].cummax()
#         df['LogConfirmed'] = df['LogConfirmed'].cummax()
        df['Confirmed_shift7'] = df['ConfirmedCases'].shift(-7)
        df['Confirmed_shift1'] = df['ConfirmedCases'].shift(-1)
        #df['Date_10Confirmed'] = df.loc[df.ConfirmedCases ==10, 'Date'].min()
        df['Fatalities_shift7'] = df['Fatalities'].shift(-7)
        df['Fatalities_shift1'] = df['Fatalities'].shift(-1)
        #df['Date_10Fatalities'] = df.loc[df.Fatalities ==10, 'Date'].min()
        #
        dfs.append(df.fillna(0))
    return pd.concat(dfs)


# In[14]:


def zero_div(a,b):
    if b==0 | a==0:
        return 0
    else:
        return a/b


# In[15]:


# train = process_each_location(train)
# test = process_each_location(test)


# In[16]:


train[train.Country_Region_=='Ukraine']


# In[17]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from xgboost import XGBRegressor


train['ConfirmedCases_diff'] = train.loc[:, ['ConfirmedCases', 'Country_Region_']].groupby('Country_Region_').diff().fillna('0')
train['Fatalities_diff'] = train.loc[:, ['Fatalities', 'Country_Region_']].groupby('Country_Region_').diff().fillna('0')



# train['ConfirmedCases_'] = to_log(train.ConfirmedCases)
# train['Fatalities_'] = to_log(train.Fatalities)
# train['ConfirmedCases_diff'] = train.loc[:, ['ConfirmedCases_', 'Country_Region_']].groupby('Country_Region_').diff().fillna('0')
# train['Fatalities_diff'] = train.loc[:, ['Fatalities_', 'Country_Region_']].groupby('Country_Region_').diff().fillna('0')
# train  = train.drop(columns=['ConfirmedCases_', 'Fatalities_'])
train = train.astype({'ConfirmedCases_diff': 'float64','Fatalities_diff': 'float64' })
#
# train.loc[train.ConfirmedCases_diff<0, 'ConfirmedCases_diff'] = 0
# train.loc[train.Fatalities_diff<0, 'Fatalities_diff'] = 0
# train['ConfirmedCases_diff'] = to_log(train.ConfirmedCases_diff)
# train['Fatalities_diff'] = to_log(train.Fatalities_diff)

#####

train['Country_Region'] = le.fit_transform(train['Country_Region'])
train['Province_State'] = le.fit_transform(train['Province_State'].fillna('0'))

test['Country_Region'] = le.fit_transform(test['Country_Region'])
test['Province_State'] = le.fit_transform(test['Province_State'].fillna('0'))


# In[18]:


train.info()


# In[19]:


# #with validation
# y1_train = train.loc[train.Date<'2020-03-26','ConfirmedCases_diff']
# y2_train = train.loc[train.Date<'2020-03-26','Fatalities_diff']

# y1_val = train.loc[train.Date>='2020-03-26','ConfirmedCases_diff']
# y2_val = train.loc[train.Date>='2020-03-26','Fatalities_diff']

# X_Id = train['Id']

# # X_train = train.drop(columns=['Id', 'Date','ConfirmedCases', 'Fatalities', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])
# # X_test  = test.drop(columns=['ForecastId', 'Date'])

# X_val = train.loc[train.Date>='2020-03-26',:].drop(columns=['Id', 'Fatalities', 'Date',
#                               'Fatalities_ratio','Change_ConfirmedCases'
#                               ,'Change_Fatalities','Country_Region_','ConfirmedCases','Fatalities_diff','ConfirmedCases_diff'])

# X_train = train.loc[train.Date>='2020-03-26',:].drop(columns=['Id', 'Fatalities', 'Date',
#                               'Fatalities_ratio','Change_ConfirmedCases'
#                               ,'Change_Fatalities','Country_Region_','ConfirmedCases','Fatalities_diff','ConfirmedCases_diff'])
# X_test  = test.drop(columns=['ForecastId','Country_Region_', 'Date'])


# In[20]:



y1_train = train['ConfirmedCases_diff']
y2_train = train['Fatalities_diff']

# y1_val = train['ConfirmedCases_diff']
# y2_val = train['Fatalities_diff']

X_Id = train['Id']

# X_train = train.drop(columns=['Id', 'Date','ConfirmedCases', 'Fatalities', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])
# X_test  = test.drop(columns=['ForecastId', 'Date'])

X_val = train.drop(columns=['Id', 'Fatalities', 'Date',
                             'Fatalities_ratio','Change_ConfirmedCases'
                             ,'Change_Fatalities','Country_Region_','ConfirmedCases','Fatalities_diff','ConfirmedCases_diff'])

X_train = train.drop(columns=['Id', 'Fatalities', 'Date',
                             'Fatalities_ratio','Change_ConfirmedCases'
                             ,'Change_Fatalities','Country_Region_','ConfirmedCases','Fatalities_diff','ConfirmedCases_diff'])
X_test  = test.drop(columns=['ForecastId','Country_Region_', 'Date'])


# In[21]:


#X_train.info()


# In[22]:


# from fbprophet import Prophet


# In[23]:


# X_train = train.drop(columns=['Id', 'Fatalities', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])
# X_test  = test.drop(columns=['ForecastId'])

# model=Prophet()
# model.fit(X_train \
#               .rename(columns={'Date':'ds',
#                                'ConfirmedCases':'y'}))
# forecast_conf=model.predict(df=X_test \
#                                    .rename(columns={'Date':'ds'}))


# In[24]:


# X_train = train.drop(columns=['Id', 'ConfirmedCases', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])
# X_test  = test.drop(columns=['ForecastId'])

# model_1=Prophet()
# model_1.fit(X_train \
#               .rename(columns={'Date':'ds',
#                                'Fatalities':'y'}))
# forecast_Fatilities=model.predict(df=X_test \
#                                    .rename(columns={'Date':'ds'}))


# In[25]:


# df_xgb_d = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases': forecast_conf.yhat, 'Fatalities': forecast_Fatilities.yhat })
# df_xgb_d.to_csv('submission.csv', index=False)


# In[26]:


# X_train = train.drop(columns=['Id', 'Date','ConfirmedCases', 'Fatalities', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])
# X_test  = test.drop(columns=['ForecastId', 'Date'])

model = xgboost.XGBRegressor(colsample_bytree=0.8,
                 gamma=0,                 
                 learning_rate=0.1,
                 max_depth=5,
                 min_child_weight=1.5,
                 n_estimators=2000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.7,
                 seed=42                           ) 


model.fit(X_train, y1_train)
y1_pred = model.predict(X_test)


model.fit(X_train, y2_train)
y2_pred = model.predict(X_test)


df = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})


# In[27]:


df.sample(20)


# In[28]:


import matplotlib.pylab as plt
from matplotlib import pyplot
from xgboost import plot_importance


# In[29]:


from pylab import rcParams
rcParams['figure.figsize'] = 5, 10

plot_importance(model, max_num_features=30) # top 10 most important features
#rcParams['figure.figsize'] = 15, 10
plt.show()


# In[ ]:





# In[30]:


#df.to_csv('submission.csv', index=False)


# In[31]:


test1 = test.copy()
test1['ConfirmedCases'] = y1_pred
test1['Fatalities']=y2_pred


# In[32]:


train.Date.max()


# In[33]:


train_max = train[['Country_Region_', 'ConfirmedCases','Fatalities']].groupby('Country_Region_').max().add_prefix('max_').reset_index()
train_max.head()


# In[34]:


test1 = test1.merge(train_max, on='Country_Region_')


# In[35]:


test1.loc[test1.ConfirmedCases<0, 'ConfirmedCases']=0
test1.loc[test1.Fatalities<0, 'Fatalities']=0
#  test1['ConfirmedCases'] = to_exp(test1.ConfirmedCases)
#  test1['Fatalities'] = to_exp(test1.Fatalities)


# In[36]:


test1['ConfirmedCases'] = test1.groupby('Country_Region_')['ConfirmedCases'].cumsum()
test1['Fatalities'] = test1.groupby('Country_Region_')['Fatalities'].cumsum()
test1['ConfirmedCases'] = test1['ConfirmedCases'] + test1['max_ConfirmedCases']
test1['Fatalities'] = test1['Fatalities'] + test1['max_Fatalities']


# In[37]:


test1.head()


# In[38]:


#test1.loc[test1.Country_Region_=='Ukraine','femalelung':]


# In[39]:


subm_fit_bell = pd.read_csv('../input/covid19-forecast-wk3/submission.csv')
subm_fit_bell.rename(columns={"ConfirmedCases": "ConfirmedCases_bell", 'Fatalities':'Fatalities_bell',
                                 }, inplace=True)
subm_fit_bell.head()


# In[40]:


test1_ = test1.merge(subm_fit_bell, on ='ForecastId')
test1_.head()


# In[41]:


test1_['ConfirmedCases'] = test1_['ConfirmedCases']*0.5 + test1_['ConfirmedCases_bell']*0.5
test1_['Fatalities'] = test1_['Fatalities']*0.5 + test1_['Fatalities_bell']*0.5
#"ConfirmedCases": "ConfirmedCases_bell", 'Fatalities':'Fatalities_bell'


# In[42]:


test1_.loc[test1.Country_Region_=='Ukraine',[ 'Date', 'ConfirmedCases', 'Fatalities']]


# In[43]:


test1_.loc[test1.Country_Region_=='Spain',:]


# In[44]:


test1.loc[test1.Country_Region_=='Italy',[ 'Date', 'ConfirmedCases', 'Fatalities']]


# In[45]:


df =test1_[['ForecastId','ConfirmedCases','Fatalities']]
df.to_csv('submission.csv', index=False)


# In[46]:


df.head()


# In[ ]:





# In[47]:


import numpy as np
import random
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib import dates
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from lmfit import minimize, Parameters, Parameter, report_fit

