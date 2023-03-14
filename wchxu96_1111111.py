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

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print (train.shape)
print (test.shape)
#print (train.head(5))


# In[3]:


#eader_board = pd.read_csv('../input/leaderboard.csv')
#print (leader_board)
'''
we should first detect the nan data and fill the nan data with something
'''
def detectnan(df):
    '''
    detect nan data
    '''
    for column in df.columns:
        


# In[4]:


train.tail()


# In[5]:


train.isnull().any()

