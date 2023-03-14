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


data = pd.read_csv("/kaggle/input/sberbank-russian-housing-market/train.csv")
data.head()


# In[3]:


cleaned_data = data[((data.build_year <= 2010) & (data.build_year >= 1900)) | (data.build_year.isna())]
cleaned_data = cleaned_data[cleaned_data.life_sq <= cleaned_data.full_sq]
cleaned_data = cleaned_data[cleaned_data.floor <]
cleaned_data.head()


# In[4]:


list(data.columns)


# In[5]:


cleaned_data['floor'].describe()


# In[6]:


column_slices = {};
column_slices[("full_sq", "price_doc")] = cleaned_data[["full_sq", "price_doc"]]
column_slices[("full_sq", "build_year")] = cleaned_data[["full_sq", "build_year"]]


# In[7]:


sum(cleaned_data.build_year.isna())


# In[ ]:





# In[8]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

plt.plot(cleaned_data.build_year, cleaned_data.full_sq, 'ro')
plt.xlabel("build_year")
plt.ylabel("full_sq")

plt.show()


# In[9]:


figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

plt.plot(cleaned_data.build_year, cleaned_data.life_sq/cleaned_data.full_sq, 'ro')
plt.xlabel("build_year")
plt.ylabel("life_sq")

plt.show()


# In[10]:


figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

year_ranges = [1900, 1950, 1980, 2000, 2010]

fig, plots = plt.subplots(4, 1, figsize=(15,45))
for i in range(4):
    year_slice_data = cleaned_data[(cleaned_data.build_year >= year_ranges[i]) & (cleaned_data.build_year >= year_ranges[i+1])]
    plots[i].plot(year_slice_data.build_year, year_slice_data.life_sq/year_slice_data.full_sq, 'ro')
    plots[i].set_xlabel("build_year")
    plots[i].set_ylabel("life_sq/full_sq")
    plots[i].set_title(str(year_ranges[i])+'-'+str(year_ranges[i+1]))
plt.show()


# In[11]:


mean_ratio_sq_per_year = cleaned_data[["build_year", "life_sq", "full_sq"]]
mean_ratio_sq_per_year["rate_sq"] = mean_ratio_sq_per_year.life_sq / mean_ratio_sq_per_year.full_sq


figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
years = sorted(mean_ratio_sq_per_year.build_year.unique())
plt.plot(years[1:], mean_ratio_sq_per_year.groupby(["build_year"]).mean().rate_sq.ewm(span = 10, adjust = True).mean())
plt.xlabel("build_year")
plt.ylabel("rate_sq")

plt.show()


# In[12]:


mean_ratio_sq_per_year.groupby(["build_year"]).mean().rate_sq.ewm(alpha=0.1).mean()


# In[13]:


years

