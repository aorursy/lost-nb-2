#!/usr/bin/env python
# coding: utf-8

# In[71]:


pip install pycountry_convert


# In[72]:


#Libraries to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import requests
import sys
from itertools import chain
import pycountry
import seaborn as sns
import pycountry_convert as pc
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV


# In[101]:


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')


# In[102]:


display(df_train)
display(df_train.describe())
display(df_train.info())


# In[114]:


df_train.isnull().sum()


# In[103]:


df_train.drop(['Province_State'],axis=1,inplace=True)


# In[104]:


df_train.groupby(['Country_Region']).sum()


# In[77]:


df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')
df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')


# In[78]:


train_date_min = df_train['Date'].min()
train_date_max = df_train['Date'].max()
print('Minimum date from training set: {}'.format(train_date_min))
print('Maximum date from training set: {}'.format(train_date_max))


# In[79]:


test_date_min = df_test['Date'].min()
test_date_max = df_test['Date'].max()
print('Minimum date from test set: {}'.format(test_date_min))
print('Maximum date from test set: {}'.format(test_date_max))


# In[80]:


#Reading the cumulative cases dataset
covid_cases = pd.read_csv('/kaggle/input/covid19cases/times_series_covid19_cases_country.csv')
covid_cases


# In[81]:


covid_cases.drop(['Lat','Long_','Incident_Rate','People_Tested','People_Hospitalized','UID','ISO3'],axis=1,inplace=True)


# In[82]:


covid_cases


# In[83]:


sns.set_style('darkgrid')
plt.figure(figsize=(15,9))
sns.barplot(x=df_train['Date'],y=df_train['Fatalities'],palette = "YlOrRd")
plt.xticks(rotation=90)
plt.show()


# In[84]:


df_train['Death']=(df_train['Fatalities']/df_train['ConfirmedCases'])*100


# In[85]:


plt.figure(figsize=(12,6))
plt.plot(df_train['Death'], label = 'Death')
plt.show()


# In[86]:


covid_country=df_train.groupby(['Country_Region']).sum()


# In[87]:


covid_country.loc['India'].transpose().plot(title='Time Series of confirmend cases of india')

Mortality Rate 
# In[91]:


df_train['Mortality Rate']=round((df_train['Death']/df_train['ConfirmedCases'])*100,2)
df_train


# In[92]:


plt.figure(figsize=(15,8))
plt.plot(df_train['Mortality Rate'],label='Mortality Rate')
plt.show()


# In[93]:


plt.figure(figsize=(15,15))
stats = [df_train.loc[:,['Country_Region','Death']]]
label = ["Death"]
threshold = [10000]
for i, stat in enumerate(stats):
    df_train = stat.groupby(["Country_Region"]).sum()
    df_train = df_countries.sort_values(df_countries.columns[-1],ascending= False)
    others = df_countries[df_countries[df_countries.columns[-1]] < threshold[i] ].sum()[-1]
    df_countries = df_countries[df_countries[df_countries.columns[-1]] > threshold[i]]
    df_countries = df_countries[df_countries.columns[-1]]
    df_countries["others"] = others
    labels = [df_countries.index[i] for i in range(df_countries.shape[0])]
    plt.pie(df_countries, labels=labels, autopct='%1.1f%%')
    plt.title('Pie Chart', fontsize = 18)
    plt.legend()
    plt.show()


# In[115]:


df_train


# In[129]:


df_train['Days']=df_train.index-df_train.index[0]
df_train


# In[120]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression(normalize=True)


# In[130]:


x=np.array(df_train['Days']).reshape(-1,1)
x


# In[131]:


y=np.array(df_train['Fatalities']).reshape(-1,1)
y


# In[132]:


lin_reg.fit(x,y)


# In[133]:


df_test['Days']=df_test.index-df_test.index[0]
df_test


# In[136]:


predicted_value=lin_reg.predict(np.array(df_test['Days']).reshape(-1,1))
predicted_value


# In[137]:


df_test['Fatalities']=predicted_value


# In[138]:


df_test


# In[139]:


l=np.array(df_train['Days']).reshape(-1,1)
l


# In[140]:


m=np.array(df_train['ConfirmedCases']).reshape(-1,1)
m


# In[141]:


lin_reg.fit(l,m)


# In[142]:


predicted_confirm=lin_reg.predict(np.array(df_test['Days']).reshape(-1,1))
predicted_confirm


# In[150]:


submission_dataset = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")


# In[151]:


#Adding results to the dataset
submission_dataset['ConfirmedCases'] = predicted_confirm
submission_dataset['Fatalities'] = predicted_value

submission_dataset.head()


# In[153]:


#Submitting the dataset
submission_dataset.to_csv("submission.csv" , index = False)


# In[ ]:




