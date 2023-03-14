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


projects = pd.read_csv("../input/train.csv")
projects.head()


# In[3]:


projects.shape


# In[4]:


projects["school_state"].value_counts().plot(kind="bar",figsize=(15,10))


# In[16]:


projects.groupby("project_subject_categories").count().plot(y=["project_is_approved"],kind='bar',figsize=(15,10))


# In[17]:


projects.groupby("project_grade_category").count().plot(y=["project_is_approved"],kind='bar',figsize=(15,10))


# In[29]:


nan = projects["project_essay_3"].isna().sum()
print(projects.shape[0]-nan,projects.shape)


# In[33]:


projects["project_essay_3"]  = projects["project_essay_3"].fillna(value="")
projects["project_essay_4"]  = projects["project_essay_4"].fillna(value="")


# In[34]:


project_essay_3_nan = projects["project_essay_3"].isna().sum()
project_essay_4_nan = projects["project_essay_4"].isna().sum()
print(project_essay_3_nan,projects.shape)
print(project_essay_4_nan,projects.shape)


# In[37]:


projects.isnull().sum()


# In[45]:


projects.groupby("teacher_prefix").count().plot(y=['project_is_approved'],kind="Bar",figsize=(15,10))


# In[48]:


projects.groupby("project_is_approved").count().sum()


# In[59]:


projects[projects.isnull().any(axis=1)]


# In[8]:


projects.corr()


# In[64]:


projects[["project_essay_1","project_essay_2","project_is_approved"]].head(20)

