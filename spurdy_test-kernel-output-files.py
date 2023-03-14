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


ls ../input


# In[3]:


pwd


# In[4]:


df = pd.read_csv('../input/quickdraw-doodle-recognition/sample_submission.csv')
df.head()


# In[5]:


sub = df.to_csv('/kaggle/working/sub_test.csv', index=False)


# In[6]:


ls


# In[7]:


df_submission = pd.read_csv('sub_test.csv')
df_submission.head()


# In[8]:


ls

