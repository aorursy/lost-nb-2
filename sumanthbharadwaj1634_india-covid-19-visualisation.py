#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pycountry_convert


# In[2]:


#Libraries to import
import pandas as pd
import numpy as np
import datetime as dt
import requests
import sys
from itertools import chain
import pycountry
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


# In[3]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 
test= pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')


# In[4]:


train_india=train[train["Country_Region"]=="India"]
train_india.info()
train_india.Province_State.fillna('NaN',inplace=True)
plot=train_india.groupby(["Date","Country_Region","Province_State"],as_index=False)['ConfirmedCases','Fatalities'].sum()


# In[5]:


tr_df=plot.query("Country_Region=='India'")


# In[6]:


tr_df.reset_index(inplace=True)


# In[7]:


tr_df


# In[8]:


df_ind_cases = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df_ind_cases.dropna(how='all',inplace=True)
df_ind_cases['DateTime'] = pd.to_datetime(df_ind_cases['Date'], format = '%d/%m/%y')
df_ind_cases.columns
df_ind_cases['State/UnionTerritory']


# In[9]:


def change_state_name(state):
    if state == 'Odisha':
        return 'Orissa'
    elif state == 'Telengana':
        return 'Telangana'
    return state
df_ind_cases['State/UnionTerritory'] = df_ind_cases.apply(lambda x: change_state_name(x['State/UnionTerritory']), axis=1)


# In[10]:


Ts_Covid=df_ind_cases[df_ind_cases['State/UnionTerritory']=='Telangana']


# In[11]:


Ts_Covid


# In[12]:


Ts_new = Ts_Covid.drop(columns=["Sno","Date","Time","State/UnionTerritory","ConfirmedIndianNational","ConfirmedForeignNational"])


# In[13]:


Ts_new.sort_values("Confirmed",inplace=True,ascending=False)
Ts_new.reset_index(drop=True,inplace=True)
Ts_new.style.background_gradient(cmap="viridis")


# In[14]:


r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson')
geojson = r.json()


# In[15]:


fig = px.choropleth(df_ind_cases, geojson=geojson, locations="State/UnionTerritory",color="Confirmed", featureidkey="properties.NAME_1",hover_data=["Cured","Deaths"], color_continuous_scale=px.colors.sequential.Viridis,)
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=600,margin={"r":0,"t":30,"l":0,"b":30})
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




