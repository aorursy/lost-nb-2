#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:





# In[2]:


import pandas as pd
a = pd.read_csv("../input/train.csv")


# In[3]:


a.head()


# In[4]:


a['genres']


# In[5]:


df=list()
for i in range(10):
  st=a['genres'][i].split()[3]
  st = st[1:-3]
  df.append(st)
print(df)


# In[6]:


a.isnull().sum()


# In[7]:



a.head()


# In[8]:


ax=["belongs_to_collection","homepage","production_countries","tagline","Keywords"]


# In[9]:


a


# In[10]:


a.drop(ax,axis=1,inplace = True) 
  


# In[11]:


a


# In[12]:


a["genres"]


# In[13]:


df=list()
for i in range(471):
  st=a['genres'][i].split()[3]
  st = st[1:-3]
  df.append(st)
print(df)


# In[14]:


a.iloc[470]


# In[15]:


a.iloc[100]


# In[16]:


a = a.fillna(a['spoken_languages'].value_counts(),inplace=True)


# In[17]:


a.iloc[470]


# In[18]:


df=list()
for i in range(471):
  st=a['genres'][i].split()[3]
  st = st[1:-3]
  df.append(st)
print(df)


# In[19]:



imp = SimpleImputer(strategy="most_frequent")
print(imp.fit_transform(a['spoken_languages'].reshape(-1,1)))


# In[20]:


for each

