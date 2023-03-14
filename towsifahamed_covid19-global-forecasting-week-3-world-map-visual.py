#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Dependencies
get_ipython().run_line_magic('matplotlib', 'inline')

# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd
import csv as csv

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv

#Shuffle the datasets
from sklearn.utils import shuffle

#Learning curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')

#import seaborn as sns
#Output plots in notebook
#%matplotlib inline 

addpoly = True
plot_lc = 0   # 1--display learning curve/ 0 -- don't display


# In[2]:


test=pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
test


# In[3]:


train=pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
train


# In[4]:


submission=pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")
submission


# In[5]:


# Plot graphic of missing values
missingno.matrix(train, figsize = (30,10))
# Let's write a little function to show us how many missing values
# there are
def find_missing_values(df, columns):
    """
    Finds number of rows where certain columns are missing values.
    ::param_df:: = target dataframe
    ::param_columns:: = list of columns
    """
    missing_vals = {}
    print("Number of missing or NaN values for each column:")
    df_length = len(df)
    for column in columns:
        total_column_values = df[column].value_counts().sum()
        missing_vals[column] = df_length-total_column_values
        #missing_vals.append(str(column)+ " column has {} missing or NaN values.".format())
    return missing_vals

missing_values = find_missing_values(train, columns=train.columns)
missing_values


# In[6]:


train['Province_State'].fillna(train['Country_Region'],inplace=True)
train


# In[7]:


df_cases=train
df_cases=df_cases.drop(['Id', 'Province_State','Date','Fatalities'], axis = 1) 
df_cases


# In[8]:


df_ft=train
df_ft=df_ft.drop(['Id', 'Province_State','Date','ConfirmedCases'], axis = 1) 
df_ft


# In[9]:


df_cases=df_cases.groupby(['Country_Region']).sum()
df_cases


# In[10]:


df_ft=df_ft.groupby(['Country_Region']).sum()
df_ft


# In[11]:


df_cases['Country_Region']=df_cases.index
df_cases


# In[12]:


df_ft['Country_Region']=df_ft.index
df_ft


# In[13]:


df_cases.reset_index(drop=True, inplace=True)
df_cases


# In[14]:


df_ft.reset_index(drop=True, inplace=True)
df_ft


# In[15]:


df_cases.rename(columns = {"Country_Region": "COUNTRY"}, inplace = True) 
df_cases


# In[16]:


df_ft.rename(columns = {"Country_Region": "COUNTRY"}, inplace = True) 
df_ft


# In[17]:


import folium


# In[18]:


#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
#df.COUNTRY.value_counts()


# In[19]:


#df_cases.COUNTRY.value_counts()


# In[20]:


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
df = pd.merge(df, df_cases, on='COUNTRY')
df


# In[21]:


df1 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
df1 = pd.merge(df1, df_ft, on='COUNTRY')
df1


# In[22]:


df.drop(['GDP (BILLIONS)'], axis = 1,inplace=True) 
df


# In[23]:


df1.drop(['GDP (BILLIONS)'], axis = 1,inplace=True) 
df1


# In[24]:


import plotly.graph_objects as go

fig = go.Figure(data=go.Choropleth(
    locations = df['CODE'],
    z = df['ConfirmedCases'],
    text = df['COUNTRY'],
    colorscale = 'Blues',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix = '',
    colorbar_title = 'Confirmed Cases',
))

fig.update_layout(
    title_text='Confirmed Cases',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='Source: <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
            CIA World Factbook</a>',
        showarrow = False
    )]
)

fig.show()


# In[25]:


import plotly.graph_objects as go

fig = go.Figure(data=go.Choropleth(
    locations = df1['CODE'],
    z = df1['Fatalities'],
    text = df1['COUNTRY'],
    colorscale = 'Blues',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix = '',
    colorbar_title = 'Death Cases',
))

fig.update_layout(
    title_text='Death Cases',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='Source: <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
            CIA World Factbook</a>',
        showarrow = False
    )]
)

fig.show()

