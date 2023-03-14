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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[3]:


train = pd.read_csv(r'../input/covid19-global-forecasting-week-3/train.csv')
train.head()


# In[4]:


test = pd.read_csv(r'../input/covid19-global-forecasting-week-3/test.csv')
test.head()


# In[5]:


data = train.fillna('NA').groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum()     .groupby(['Country_Region','Province_State']).max().sort_values()     .groupby(['Country_Region']).sum().sort_values(ascending = False)
data.head()


# In[6]:


top15 = pd.DataFrame(data).head(15)
top15


# In[7]:


fig = px.bar(top15, x=top15.index, y='ConfirmedCases', labels={'x':'Country'}, color='ConfirmedCases', 
             color_continuous_scale=px.colors.sequential.Tealgrn)
fig.update_layout(title_text='Total Confirmed COVID-19 cases by country')
fig.show()


# In[8]:


data_deaths = train.fillna('NA').groupby(['Country_Region','Province_State','Date'])['Fatalities'].sum()     .groupby(['Country_Region','Province_State']).max().sort_values()     .groupby(['Country_Region']).sum().sort_values(ascending = False)
data_deaths.head()


# In[9]:


top15_deaths = pd.DataFrame(data_deaths).head(15)
top15_deaths


# In[10]:


fig = px.bar(top15_deaths, x=top15_deaths.index, y='Fatalities', labels={'x':'Country'}, color='Fatalities',
            color_continuous_scale=px.colors.sequential.Burg)
fig.update_layout(title_text='Total Fatalities Caused from COVID-19 by Country')
fig.show()


# In[11]:


daily_cases = train.groupby(['Date'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()
daily_cases.head(10)


# In[12]:


daily_cases_melt = pd.melt(daily_cases, id_vars=['Date'], value_vars=['ConfirmedCases', 'Fatalities'])
fig = px.line(daily_cases_melt, x="Date", y="value", color='variable',
             title="Daily Confirmed Cases and Fatalities")

fig.show()


# In[13]:


daily_cases['Mortality'] = daily_cases['Fatalities'] / daily_cases['ConfirmedCases']

fig = px.line(daily_cases, x="Date", y="Mortality", 
              title="Mortality Rate Over Time")
fig.show()


# In[14]:


data_daywise = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases','Fatalities'].sum().reset_index())
data_daywise_US = data_daywise.loc[(data_daywise['Country_Region'] == 'US') &(data_daywise.Date >= '2020-03-01')]

data_daywise_USmelt = pd.melt(data_daywise_US, id_vars=['Date'], value_vars=['ConfirmedCases', 'Fatalities'])
fig = px.line(data_daywise_USmelt, x="Date", y="value", color='variable',
             title="Daily Confirmed Cases and Fatalities")

fig.show()


# In[15]:


data_daywise = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases','Fatalities'].sum().reset_index())
data_daywise_Italy = data_daywise.loc[(data_daywise['Country_Region'] == 'Italy') &(data_daywise.Date >= '2020-03-01')]

data_daywise_Italymelt = pd.melt(data_daywise_Italy, id_vars=['Date'], value_vars=['ConfirmedCases', 'Fatalities'])
fig = px.line(data_daywise_Italymelt, x="Date", y="value", color='variable',
             title="Daily Confirmed Cases and Fatalities")

fig.show()


# In[16]:


data_daywise = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases','Fatalities'].sum().reset_index())
data_daywise_Spain = data_daywise.loc[(data_daywise['Country_Region'] == 'Spain') &(data_daywise.Date >= '2020-03-01')]

data_daywise_Spainmelt = pd.melt(data_daywise_Spain, id_vars=['Date'], value_vars=['ConfirmedCases', 'Fatalities'])
fig = px.line(data_daywise_Spainmelt, x="Date", y="value", color='variable',
             title="Daily Confirmed Cases and Fatalities")

fig.show()


# In[17]:


data_daywise = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases','Fatalities'].sum().reset_index())
data_daywise_India = data_daywise.loc[(data_daywise['Country_Region'] == 'India') &(data_daywise.Date >= '2020-03-01')]

data_daywise_Indiamelt = pd.melt(data_daywise_India, id_vars=['Date'], value_vars=['ConfirmedCases', 'Fatalities'])
fig = px.line(data_daywise_Indiamelt, x="Date", y="value", color='variable',
             title="Daily Confirmed Cases and Fatalities")

fig.show()


# In[18]:


sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
sub.head()


# In[19]:


sub.to_csv('submission.csv', index = False)


# In[ ]:




