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


# In[2]:


### LGB
m15 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m15.csv')
m17 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m17.csv')
m18 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m18.csv')
m19 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m19.csv')
m20 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m20.csv')
### Catboost
m16 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m16.csv')
### NN
m0 = pd.read_csv('/kaggle/input/ieee-top-models-blend/m0.csv')

submission = pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")
submission.head()


# In[3]:


m18.head()


# In[4]:


submission['isFraud'] = (0.20*m15.isFraud) +                         (0.20*m17.isFraud) +                         (0.20*m18.isFraud) +                         (0.20*m19.isFraud) +                         (0.10)*m20.isFraud +                         (0.10*m16.isFraud) +                         (0.0*m0.isFraud)       
                        
submission.to_csv('my_blend_5.csv', index=False)

