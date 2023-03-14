#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__) # requires version >= 1.9.0
# For Notebooks
init_notebook_mode(connected=True)
# For offline use


# In[2]:


# Handy code which I prefer to enlarge the plots
plt.rcParams["figure.figsize"] = (18,9)


# In[3]:


train = pd.read_csv('../input/train_2016_v2.csv')
properties = pd.read_csv('../input/properties_2016.csv', low_memory=False)
dictionary = pd.read_excel('../input/zillow_data_dictionary.xlsx')


# In[4]:


train.head()


# In[5]:


properties.head()


# In[6]:


null = pd.DataFrame(data= properties.isnull().sum()/len(properties)*100, 
                    columns=['Percentage of Values Missing'],
                    index=properties.columns
                   ).reset_index()


# In[7]:


null['Percentage of Values Missing'].mean()


# In[8]:


plt.rcParams["figure.figsize"] = (13,10)
sns.barplot(x= 'Percentage of Values Missing', 
            y='index', 
            data= null.sort_values(by='Percentage of Values Missing', ascending=False),
            color = '#ff004f') 


# In[9]:


## Caution - Only 50% percentile missing values are taken. There are 29 MORE!!!
Notorious_null = null[null['Percentage of Values Missing'] > null['Percentage of Values Missing'].mean()]


# In[10]:


Notorious_null.sort_values(by='Percentage of Values Missing', ascending=False).head(10)


# In[11]:


plt.rcParams["figure.figsize"] = (13,10)
sns.barplot(x= 'Percentage of Values Missing', 
            y='index', 
            data= Notorious_null,
            color = '#ff004f') 


# In[12]:


len(null) - len(Notorious_null)


# In[13]:


alldata = pd.merge(train, properties, how='inner', on='parcelid')


# In[14]:


alldata.head()


# In[15]:


# sns.heatmap(alldata.corr(), cmap='viridis', vmax=0.8, vmin=0)


# In[16]:


alldata.head(10)


# In[17]:


null_drop = null[null['Percentage of Values Missing'] > 85]


# In[18]:


col_to_drop = []
for cols in list(null_drop['index'].values):
    col_to_drop.append(cols)


# In[19]:


alldata.drop(col_to_drop, axis=1, inplace=True)


# In[20]:


alldata.head()


# In[21]:


nullv2 = pd.DataFrame(data= alldata.isnull().sum()/len(alldata)*100, 
                    columns=['Percentage of Values Missing'],
                    index=alldata.columns
                   ).reset_index()


# In[22]:


nullv2.sort_values(by='Percentage of Values Missing', ascending=False)


# In[23]:


alldata.fillna(value=0, inplace=True)


# In[24]:


alldata.head(8)


# In[25]:


sns.heatmap(alldata.corr().head(500), cmap='viridis', vmax=0.8, vmin=0)


# In[26]:


alldata.describe()


# In[27]:


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=20)


# In[28]:


X = alldata.drop(['parcelid','logerror', axis=1)


# In[29]:




