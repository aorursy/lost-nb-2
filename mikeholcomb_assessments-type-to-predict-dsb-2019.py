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


import os
DATA_DIR = '/kaggle/input/data-science-bowl-2019'
test = pd.read_csv(os.path.join(DATA_DIR,'test.csv'))


# In[3]:


test.set_index(['installation_id','timestamp'])


# In[4]:


# Adapted from https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_sql.html#top-n-rows-per-group
last_entries = test.assign(rn=test.sort_values(['timestamp'], ascending=False)            .groupby(['installation_id'])            .cumcount() + 1)            .query('rn == 1')            .sort_values(['installation_id'])


# In[5]:


last_entries[['installation_id','title']].to_csv('test_final_title.csv',index=False)


# In[6]:


last_entries['title'].value_counts()


# In[ ]:




