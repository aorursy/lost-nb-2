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


train_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3//train.csv')
test_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3//test.csv')
submission_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3//submission.csv')


# In[3]:


train_set.head()


# In[4]:


test_set.head()


# In[5]:


train_set.info()


# In[6]:


test_set.info()


# In[7]:


df = train_set.fillna('NA').groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum()                           .groupby(['Country_Region','Province_State']).max().sort_values()                           .groupby(['Country_Region']).sum().sort_values(ascending = False)


# In[8]:


top_15_countries = pd.DataFrame(df).head(15)
top_15_countries


# In[9]:


df1 = train_set.fillna('NA').groupby(['Country_Region','Province_State','Date'])['Fatalities'].sum()                           .groupby(['Country_Region','Province_State']).max().sort_values()                           .groupby(['Country_Region']).sum().sort_values(ascending = False)


# In[10]:


top_15_countries_fatal = pd.DataFrame(df1).head(15)
top_15_countries_fatal


# In[11]:


#train_set_week2 = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')


# In[12]:


# df_week2 = train_set_week2.fillna('NA').groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum() \
#                           .groupby(['Country_Region','Province_State']).max().sort_values() \
#                           .groupby(['Country_Region']).sum().sort_values(ascending = False)


# In[13]:


# top_15_countries_week2 = pd.DataFrame(df_week2).head(15)
# top_15_countries_week2


# In[14]:


# df1_week2 = train_set_week2.fillna('NA').groupby(['Country_Region','Province_State','Date'])['Fatalities'].sum() \
#                           .groupby(['Country_Region','Province_State']).max().sort_values() \
#                           .groupby(['Country_Region']).sum().sort_values(ascending = False)


# In[15]:


# top_15_countries_fatal_week2 = pd.DataFrame(df1_week2).head(15)
# top_15_countries_fatal_week2


# In[16]:


# df_all = pd.concat([top_15_countries, top_15_countries_week2], 
#                    axis='columns', sort=True)
# print("Confirmed cases compared between past 2 weeks:" + 
#       "\n \t\t Week3 \t\t Week2" + "\n" +str(df_all))


# In[17]:


# df_all_fatal = pd.concat([top_15_countries_fatal, top_15_countries_fatal_week2], 
#                    axis='columns', sort=True)
# print("Fatalities compared between past 2 weeks:" + 
#       "\n \t\t Week3 \t\t Week2" + "\n" +str(df_all_fatal))


# In[18]:


# import plotly.express as px
# fig = px.bar(top_15_countries, x=top_15_countries.index, y='ConfirmedCases', labels={'x':'Countries'},  color='ConfirmedCases', barmode='group',
#              height=400, title="Week 3 Confimed Cases")
# fig1 = px.bar(top_15_countries_week2, x=top_15_countries_week2.index, y='ConfirmedCases', labels={'x':'Countries'},  color='ConfirmedCases', barmode='group',
#              height=400, title="Week 2 Confirmed Cases")
# fig.show()
# fig1.show()


# In[19]:


# import plotly.express as px
# fig = px.bar(top_15_countries_fatal, x=top_15_countries_fatal.index, y='Fatalities', labels={'x':'Countries'},  color='Fatalities', barmode='group',
#              height=400, title="Week 3 Fatalities")
# fig1 = px.bar(top_15_countries_fatal_week2, x=top_15_countries_fatal_week2.index, y='Fatalities', labels={'x':'Countries'},  color='Fatalities', barmode='group',
#              height=400, title="Week 2 Fatalities")
# fig.show()
# fig1.show()


# In[20]:


train_set_copy = train_set.drop(['Province_State'], axis=1)
train_set_copy.head()


# In[21]:


df2 = train_set_copy.groupby(['Country_Region','Date'])['Fatalities'].sum()                     .groupby(['Country_Region']).max().sort_values(ascending = False)

df2.head()


# In[22]:


df3 = train_set_copy.groupby(['Country_Region','Date'])['ConfirmedCases'].sum()                     .groupby(['Country_Region']).max().sort_values(ascending = False)

df3.head()


# In[23]:


percentage_value = ((df2/df3)*100).sort_values(ascending = False)
percentage_value = pd.DataFrame(percentage_value)
percentage_value.columns = ['Percentage']
#Drop all the percentage value with no ratio
percentage_value = percentage_value.replace(0.0, np.nan)
percentage_value = percentage_value.dropna(how='all', axis=0)
percentage_value.tail()


# In[24]:


fig = px.bar(percentage_value.dropna(), x=percentage_value.index, y='Percentage', labels={'x':'Countries'},  color='Percentage', 
             title='Death VS Confirmed_Cases Ratio',
             barmode='group',
             height=700)
fig.show()


# In[25]:


fig = px.scatter_geo(train_set,  locations="Country_Region",
                     locationmode='country names',
                     color="Country_Region", 
                     hover_name="ConfirmedCases", 
                     size="ConfirmedCases", 
                     title='Total ConfirmedCases over time',
                      
                     projection="orthographic")
fig.show()


# In[26]:


fig = px.scatter_geo(train_set,  locations="Country_Region",
                     locationmode='country names',
                     color="Country_Region", 
                     hover_name="Fatalities", 
                     size="Fatalities", 
                     title='Total Deaths over time',
                      
                     projection="orthographic")
fig.show()


# In[27]:


train_set.isnull().sum()


# In[28]:


test_set.isnull().sum()


# In[29]:


train_set_copy.isnull().sum()


# In[30]:


test_set_copy = test_set.drop(['Province_State'], axis=1)
test_set_copy.isnull().sum()


# In[31]:


train_set_copy["Date"] = train_set_copy["Date"].apply(lambda x: x.replace("-",""))
train_set_copy["Date"] = train_set_copy["Date"].astype(int)
train_set_copy.head()


# In[32]:


test_set_copy["Date"] = test_set_copy["Date"].apply(lambda x: x.replace("-",""))
test_set_copy["Date"] = test_set_copy["Date"].astype(int)
test_set_copy.head()


# In[33]:


x_train = train_set_copy[['Date']]
y1_train = train_set_copy['ConfirmedCases']
y2_train = train_set_copy['Fatalities']
x_test = test_set_copy[['Date']]


# In[34]:


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}


# In[35]:


xgb_grid = GridSearchCV(xgb1,parameters,cv = 2,n_jobs = 5,verbose=True)


# In[36]:


xgb_grid.fit(x_train, y1_train)


# In[37]:


x_pred = xgb_grid.predict(x_test)
prediction1 = pd.DataFrame(x_pred)
prediction1.columns = ["ConfirmedCases_prediction"]


# In[38]:


prediction1 = prediction1.round()


# In[39]:


prediction1.head()


# In[40]:


xgb_grid.fit(x_train, y2_train)


# In[41]:


x_pred = xgb_grid.predict(x_test)
prediction2 = pd.DataFrame(x_pred)
prediction2.columns = ["Fatalities_prediction"]


# In[42]:


prediction2 = prediction2.round()


# In[43]:


prediction2.head()


# In[44]:


submission_set.head()


# In[45]:


submission_forecast = submission_set['ForecastId']
submission_forecast = pd.DataFrame(submission_forecast)
submission_forecast.head()


# In[46]:


submission = pd.concat([submission_forecast, prediction1, prediction2], axis=1)
submission.head()


# In[47]:


submission.columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']
submission.head()


# In[48]:


submission.describe()


# In[49]:


submission.to_csv('submission.csv', index = False)


# In[ ]:




