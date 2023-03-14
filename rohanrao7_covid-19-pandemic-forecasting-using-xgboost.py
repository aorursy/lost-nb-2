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


train_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
test_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
submission_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')


# In[3]:


train_set.head()


# In[4]:


test_set.head()


# In[5]:


train_set.info()


# In[6]:


test_set.info()


# In[7]:


train_set.sample(5)


# In[8]:


test_set.sample(5)


# In[9]:


df = train_set.fillna('NA').groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum()                           .groupby(['Country_Region','Province_State']).max().sort_values()                           .groupby(['Country_Region']).sum().sort_values(ascending = False)


# In[10]:


top_15_countries = pd.DataFrame(df).head(15)
top_15_countries


# In[11]:


df1 = train_set.fillna('NA').groupby(['Country_Region','Province_State','Date'])['Fatalities'].sum()                           .groupby(['Country_Region','Province_State']).max().sort_values()                           .groupby(['Country_Region']).sum().sort_values(ascending = False)


# In[12]:


top_15_countries_fatal = pd.DataFrame(df1).head(15)
top_15_countries_fatal


# In[13]:


import plotly.express as px
fig = px.bar(top_15_countries, x=top_15_countries.index, y='ConfirmedCases', labels={'x':'Countries'},  color='ConfirmedCases', barmode='group',
             height=400)
fig1 = px.bar(top_15_countries_fatal, x=top_15_countries_fatal.index, y='Fatalities', labels={'x':'Countries'},  color='Fatalities', barmode='group',
             height=400)
fig.show()
fig1.show()


# In[14]:


train_set_copy = train_set.drop(['Province_State'], axis=1)
train_set_copy.head()


# In[15]:


df2 = train_set_copy.groupby(['Country_Region','Date'])['Fatalities'].sum()                     .groupby(['Country_Region']).max().sort_values(ascending = False)

df2.head()


# In[16]:


df3 = train_set_copy.groupby(['Country_Region','Date'])['ConfirmedCases'].sum()                     .groupby(['Country_Region']).max().sort_values(ascending = False)

df3.head()


# In[17]:


percentage_value = ((df2/df3)*100).sort_values(ascending = False)
percentage_value = pd.DataFrame(percentage_value)
percentage_value.columns = ['Percentage']
#Drop all the percentage value with no ratio
percentage_value = percentage_value.replace(0.0, np.nan)
percentage_value = percentage_value.dropna(how='all', axis=0)
percentage_value.tail()


# In[18]:


fig = px.bar(percentage_value.dropna(), x=percentage_value.index, y='Percentage', labels={'x':'Countries'},  color='Percentage', 
             title='Death VS Confirmed_Cases Ratio',
             barmode='group',
             height=700)
fig.show()


# In[ ]:





# In[19]:


train_set.isnull().sum()


# In[20]:


test_set.isnull().sum()


# In[21]:


train_set_copy.isnull().sum()


# In[22]:


test_set_copy = test_set.drop(['Province_State'], axis=1)
test_set_copy.isnull().sum()


# In[23]:


train_set_copy["Date"] = train_set_copy["Date"].apply(lambda x: x.replace("-",""))
train_set_copy["Date"] = train_set_copy["Date"].astype(int)
train_set_copy.head()


# In[24]:


test_set_copy["Date"] = test_set_copy["Date"].apply(lambda x: x.replace("-",""))
test_set_copy["Date"] = test_set_copy["Date"].astype(int)
test_set_copy.head()


# In[25]:


x_train = train_set_copy[['Date']]
y1_train = train_set_copy['ConfirmedCases']
y2_train = train_set_copy['Fatalities']
x_test = test_set_copy[['Date']]


# In[26]:


from xgboost import XGBRegressor
classifier = XGBRegressor(max_depth=8, n_estimators=1000, random_state=0)
classifier.fit(x_train, y1_train)


# In[27]:


x_pred = classifier.predict(x_test)
prediction1 = pd.DataFrame(x_pred)
prediction1.columns = ["ConfirmedCases_prediction"]


# In[28]:


prediction1


# In[29]:


from xgboost import XGBRegressor
classifier = XGBRegressor(max_depth=8, n_estimators=1000, random_state=0)
classifier.fit(x_train, y2_train)


# In[30]:


x_pred = classifier.predict(x_test)
prediction2 = pd.DataFrame(x_pred)
prediction2.columns = ["Fatalities_prediction"]


# In[31]:


prediction2.head()


# In[32]:


submission_set.head()


# In[33]:


submission_forecast = submission_set['ForecastId']
submission_forecast = pd.DataFrame(submission_forecast)
submission_forecast.head()


# In[34]:


submission = pd.concat([submission_forecast, prediction1, prediction2], axis=1)
submission.head()


# In[35]:


submission.columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']
submission.head()


# In[36]:


submission.describe()


# In[37]:


submission.to_csv('submission.csv', index = False)


# In[ ]:




