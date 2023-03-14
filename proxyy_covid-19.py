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


df_train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
df_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')


# In[3]:


df_train.count()


# In[4]:


df=pd.concat([df_train,df_test])


# In[5]:


df.iloc[22944]


# In[6]:


df.iloc[22950]


# In[7]:


df_test.iloc[0]


# In[8]:


df.Province_State.fillna('NaN',inplace=True)


# In[9]:


df.head()


# In[10]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


# In[11]:


enc=OneHotEncoder()
oe=OrdinalEncoder()


# In[12]:


df[['Province_State','Country_Region']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region']])


# In[13]:


df_ttrain=df.iloc[0:22950,:]


# In[14]:


df_ttrain.shape


# In[15]:


df_ttest=df.iloc[22950:,:]


# In[16]:


df_ttest.shape


# In[17]:


df_ttrain['Date']=pd.to_datetime(df_ttrain['Date'])


# In[18]:


df_ttest['Date']=pd.to_datetime(df_ttest['Date'])


# In[19]:


import datetime as dt
def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df


# In[20]:


df_ttrain.head()


# In[21]:


df_ttrain=df_ttrain.drop(['ForecastId'],axis=1)


# In[22]:


df_ttest=df_ttest.drop(['Id','Fatalities','ConfirmedCases'],axis=1)


# In[23]:


df_ttrain.shape


# In[24]:


df_ttrain = create_features(df_ttrain)


# In[25]:


df_ttrain.head()


# In[26]:


df_ttest = create_features(df_ttest)


# In[27]:


df_ttest.head()


# In[28]:


from xgboost.sklearn import XGBRegressor


# In[29]:


model1 = XGBRegressor(n_estimators=3000)


# In[30]:


X_train=df_ttrain.loc[:,['Country_Region','Province_State','day','month','dayofweek','dayofyear','quarter','weekofyear']].values


# In[31]:


y_train1=df_ttrain.loc[:,['ConfirmedCases']].values


# In[32]:


y_train2=df_ttrain.loc[:,['Fatalities']].values


# In[33]:


model1.fit(X_train,y_train1)


# In[34]:


model2 = XGBRegressor(n_estimators=3000)


# In[35]:


model2.fit(X_train,y_train2)


# In[36]:


X_test=df_ttest.loc[:,['Country_Region','Province_State','day','month','dayofweek','dayofyear','quarter','weekofyear']].values


# In[37]:


y_pred1=model1.predict(X_test)


# In[38]:


y_pred2=model2.predict(X_test)


# In[39]:


submission1=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')


# In[40]:


submission1['ConfirmedCases']=y_pred1
submission1['Fatalities']=y_pred2


# In[41]:


submission1.head()


# In[42]:


submission1.to_csv('submission.csv',index=False)

