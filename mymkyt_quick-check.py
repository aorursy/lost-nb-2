#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


sample = pd.read_csv('../input/sample_submission.csv', nrows=10)
sample.head()


# In[3]:


train = pd.read_csv('../input/clicks_train.csv', nrows=10)
train.head()


# In[4]:


test= pd.read_csv('../input/clicks_test.csv', nrows=10)
test.head()


# In[5]:


events_head = pd.read_csv('../input/events.csv', nrows=10)
events_head.head()


# In[6]:


promoted_content = pd.read_csv('../input/promoted_content.csv', nrows=10)
promoted_content.head()


# In[7]:


page_views = pd.read_csv('../input/page_views_sample.csv', nrows=10)
page_views.head()


# In[8]:


doc_category = pd.read_csv('../input/documents_categories.csv', nrows = 10)
doc_category.head()


# In[9]:


doc_meta = pd.read_csv('../input/documents_meta.csv', nrows = 10)
doc_meta.head()


# In[10]:


doc_entities = pd.read_csv('../input/documents_entities.csv',nrows = 10)
doc_entities.head()


# In[11]:


doc_topics = pd.read_csv('../input/documents_topics.csv',nrows = 10)
doc_topics.head()


# In[12]:




