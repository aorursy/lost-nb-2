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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", ".."]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[2]:


train_raw = pd.read_csv("../input/train.csv")


# In[3]:


sum(train_raw["is_duplicate"])/len(train_raw.index)
train_raw.head


len(train_raw.index)


# In[4]:


sum(train_raw["is_duplicate"])/len(train_raw.index)


# In[5]:


train_raw.columns.values


# In[6]:


sum(train_raw["is_duplicate"])


# In[7]:


test_raw = pd.read_csv("../input/test.csv")


# In[8]:


test_raw.head


# In[9]:


len(test_raw.index)


# In[10]:


basepreds = np.repeat(sum(train_raw["is_duplicate"])/len(train_raw.index),len(test_raw.index))


# In[11]:


sub1 = pd.DataFrame({"test_id" : test_raw["test_id"], "is_duplicate" : basepreds})


# In[12]:


sub1.head


# In[13]:


submission1 = sub1.to_csv("sub1.csv", index = False)


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer


# In[15]:


count_vec = CountVectorizer()


# In[16]:


train_q1_counts = count_vec.fit_transform(train_raw["question1"])


# In[17]:


train_q1_counts.shape


# In[18]:


print(train[1]"question1"])


# In[19]:


import re

clean1 = train_raw.loc[1,"question1"].lower()
clean2 = re.sub(r'[?.,\/#!$%\^&\*;:{}=\-_`~()]','',clean1)

jacc1 = set(clean2.split())
jacc1


# In[20]:


clean1 = train_raw.loc[1,"question2"].lower()
clean2 = re.sub(r'[?.,\/#!$%\^&\*;:{}=\-_`~()]','',clean1)

jacc2 = set(clean2.split())
jacc2


# In[21]:


union = jacc1.union(jacc2)
union


# In[22]:


intersect = jacc1 & jacc2
intersect


# In[23]:


jacc = len(intersect)/len(union)
jacc


# In[24]:


len(intersect)


# In[25]:


len(union)
train_raw['question']


# In[26]:


train_qs = pd.Series(train_raw['question1'].tolist() + train_raw['question2'].tolist()).astype(str)
train_qs.dtypes


# In[27]:


words = (" ".join(train_qs)).lower().split()
words


# In[28]:




