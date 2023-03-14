#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import gc
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')


# In[3]:


train = [train.iloc[500000*i:500000*(i+1), :].reset_index(drop = True) for i in range(10)]
test = [test.iloc[500000*i:500000*(i+1), :].reset_index(drop = True) for i in range(4)]


# In[4]:


dr = pd.concat([train[i][['signal']] for i in range(10)]+[test[i][['signal']] for i in range(4)], axis = 1)

dr.columns = ['train_b1', 'train_b2', 'train_b3', 'train_b4', 'train_b5', 'train_b6', 'train_b7', 'train_b8',
             'train_b9', 'train_b10', 'test_b1', 'test_b2', 'test_b3', 'test_b4']

corr = dr.corr()

corr


# In[5]:


import seaborn as sns
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]


# In[6]:


corr.style.background_gradient(cmap, axis=1).set_properties(**{'max-width': '100px', 'font-size': '10pt'})        .set_caption("Signal Correlation").set_precision(2).set_table_styles(magnify())


# In[7]:


MAX_roll = 6
def movingaverage(df):
    df['cummax'] = df['signal'].cummax()
    df['cummin'] = df['signal'].cummin()
    
    for i in range(2,MAX_roll):
        df['MA_{}'.format(i)] = df['signal'].rolling(window=i).mean()
    df.fillna(-999, inplace = True)
    df.reset_index(drop = True, inplace = True)
    return df


# In[8]:


train = [movingaverage(x) for x in train]
test = [movingaverage(x) for x in test]


# In[9]:


train[2].head()


# In[10]:


dr = pd.concat([train[i][['MA_2']] for i in range(10)]+[test[i][['MA_2']] for i in range(4)], axis = 1)

dr.columns = ['train_b1', 'train_b2', 'train_b3', 'train_b4', 'train_b5', 'train_b6', 'train_b7', 'train_b8',
             'train_b9', 'train_b10', 'test_b1', 'test_b2', 'test_b3', 'test_b4']

corr = dr.corr()
corr.style.background_gradient(cmap, axis=1).set_properties(**{'max-width': '100px', 'font-size': '10pt'})        .set_caption("Signal Moving Average 2 Correlation").set_precision(2).set_table_styles(magnify())


# In[11]:


dr = pd.concat([train[i][['MA_3']] for i in range(10)]+[test[i][['MA_3']] for i in range(4)], axis = 1)

dr.columns = ['train_b1', 'train_b2', 'train_b3', 'train_b4', 'train_b5', 'train_b6', 'train_b7', 'train_b8',
             'train_b9', 'train_b10', 'test_b1', 'test_b2', 'test_b3', 'test_b4']

corr = dr.corr()
corr.style.background_gradient(cmap, axis=1).set_properties(**{'max-width': '100px', 'font-size': '10pt'})        .set_caption("Signal Moving Average 3 Correlation").set_precision(2).set_table_styles(magnify())


# In[12]:


dr = pd.concat([train[i][['MA_4']] for i in range(10)]+[test[i][['MA_4']] for i in range(4)], axis = 1)

dr.columns = ['train_b1', 'train_b2', 'train_b3', 'train_b4', 'train_b5', 'train_b6', 'train_b7', 'train_b8',
             'train_b9', 'train_b10', 'test_b1', 'test_b2', 'test_b3', 'test_b4']

corr = dr.corr()
corr.style.background_gradient(cmap, axis=1).set_properties(**{'max-width': '100px', 'font-size': '10pt'})        .set_caption("Signal Moving Average 4 Correlation").set_precision(2).set_table_styles(magnify())


# In[13]:


dr = pd.concat([train[i][['MA_5']] for i in range(10)]+[test[i][['MA_5']] for i in range(4)], axis = 1)

dr.columns = ['train_b1', 'train_b2', 'train_b3', 'train_b4', 'train_b5', 'train_b6', 'train_b7', 'train_b8',
             'train_b9', 'train_b10', 'test_b1', 'test_b2', 'test_b3', 'test_b4']

corr = dr.corr()
corr.style.background_gradient(cmap, axis=1).set_properties(**{'max-width': '100px', 'font-size': '10pt'})        .set_caption("Signal Moving Average 5 Correlation").set_precision(2).set_table_styles(magnify())

