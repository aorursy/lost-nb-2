#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

train.head(10)


# In[3]:


headers = train.columns.values
print(headers)

def count_occ(cat, headers):
    print("Number of ", cat, ": ", len([h for h in ]))
print("Number of cont:")


# In[4]:




