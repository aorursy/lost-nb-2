#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
from sklearn import preprocessing as prp

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


tr_data = pd.read_csv('../input/train.csv',parse_dates=[0])
test_raw_data = pd.read_csv('../input/test.csv',parse_dates=[0])

train_len = len(tr_data)

train_targets = tr_data.ix[:,-3:]
tr_data = tr_data.ix[:,:-3]

data = pd.concat([tr_data, test_raw_data],axis=0)


# In[3]:


data.describe()


# In[4]:


column_1 = data.ix[:,0]
date_data = pd.DataFrame({"month": column_1.dt.month,
                           "dayofweek": column_1.dt.dayofweek,
                           "hour": column_1.dt.hour,
                         })

month_oh = pd.get_dummies(date_data['month'],prefix="month")
dayofweek_oh = pd.get_dummies(date_data['dayofweek'],prefix="dayofweek")
hour_oh = pd.get_dummies(date_data['hour'],prefix="hour")
season_oh = pd.get_dummies(data['season'],prefix="season")
holiday_oh = pd.get_dummies(data['holiday'],prefix="holiday")
workingday_oh = pd.get_dummies(data['workingday'],prefix="workingday")
weather_oh =pd.get_dummies(data['weather'],prefix="weather")
temp_num = prp.minmax_scale(data['temp']).reshape((-1,1))
atemp_num = prp.minmax_scale(data['atemp']).reshape((-1,1))
humidity_num = prp.minmax_scale(data['humidity']).reshape((-1,1))
windspeed_num = prp.minmax_scale(data['windspeed']).reshape((-1,1))

features_headers = month_oh.columns | dayofweek_oh.columns | hour_oh.columns | season_oh.columns | holiday_oh.columns | workingday_oh.columns | weather_oh.columns

features_headers = list(features_headers)
features_headers.append('temp_num')
features_headers.append('atemp_num')
features_headers.append('humidity_num')
features_headers.append('windspeed_num')

print (temp_num.shape)


# In[5]:


data = np.hstack((month_oh,
                  dayofweek_oh,
                  hour_oh,
                  season_oh,
                  holiday_oh,
                  workingday_oh,
                  weather_oh,
                  temp_num,
                  atemp_num,
                  humidity_num,
                  windspeed_num))

train_targets = prp.minmax_scale(train_targets['count'])

train_data = data[:train_len, :]
test_data = data[train_len:, :]


# In[6]:


from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

clf = ExtraTreesRegressor (n_estimators=200)
clf = clf.fit(train_data, train_targets)
#features = pd.DataFrame()
#features['feature'] = np.arange(len(train_data))
#features['importance'] = clf.feature_importances_

#print (len(clf.feature_importances_))

#print (sorted(clf.feature_importances_,reverse=True))

model = SelectFromModel(clf, prefit=True)
train_data_new = model.transform(train_data)
test_data_new = model.transform(test_data)
print (train_data_new.shape)
print (test_data_new.shape)


# In[7]:


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# In[8]:


forest = RandomForestRegressor(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5], #5,6],#,7,8],
                 'n_estimators': [200,210],#,240,250],
                 'criterion': ['mse']#,'mae']
                 }

cross_validation = KFold(n_splits=5, shuffle=True)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation,
                           scoring=rmsle_scorer
                          )

grid_search.fit(train_data_new, train_targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# In[9]:


output = grid_search.predict(test_data_new).astype(int)
df_output = pd.DataFrame()
df_output['datetime'] = test_raw_data['datetime']
df_output['count'] = output
df_output[['datetime','count']].to_csv('../input/output.csv',index=False)

