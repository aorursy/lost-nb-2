#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.10f}'.format)

# read the data
train = pd.read_csv('../input/train.csv')
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')


# In[2]:


# fast way to get last historic transaction / first new transaction
last_hist_transaction = historical_transactions.groupby('card_id').agg({'month_lag' : 'max', 'purchase_date' : 'max'}).reset_index()
last_hist_transaction.columns = ['card_id', 'hist_month_lag', 'hist_purchase_date']
first_new_transaction = new_merchant_transactions.groupby('card_id').agg({'month_lag' : 'min', 'purchase_date' : 'min'}).reset_index()
first_new_transaction.columns = ['card_id', 'new_month_lag', 'new_purchase_date']


# In[3]:


# converting to datetime
last_hist_transaction['hist_purchase_date'] = pd.to_datetime(last_hist_transaction['hist_purchase_date']) 
first_new_transaction['new_purchase_date'] = pd.to_datetime(first_new_transaction['new_purchase_date']) 


# In[4]:


# substracting month_lag for each row
last_hist_transaction['observation_date'] =     last_hist_transaction.apply(lambda x: x['hist_purchase_date']  - pd.DateOffset(months=x['hist_month_lag']), axis=1)

first_new_transaction['observation_date'] =     first_new_transaction.apply(lambda x: x['new_purchase_date']  - pd.DateOffset(months=x['new_month_lag']-1), axis=1)


# In[5]:


last_hist_transaction.head(20)


# In[6]:


first_new_transaction.head(20)


# In[7]:


last_hist_transaction['observation_date'] = last_hist_transaction['observation_date'].dt.to_period('M').dt.to_timestamp() + pd.DateOffset(months=1)
first_new_transaction['observation_date'] = first_new_transaction['observation_date'].dt.to_period('M').dt.to_timestamp()


# In[8]:


last_hist_transaction.head(20)


# In[9]:


first_new_transaction.head(20)


# In[10]:


validate = last_hist_transaction.merge(first_new_transaction, on = 'card_id')
all(validate['observation_date_x'] == validate['observation_date_y'])


# In[11]:


train = train.merge(last_hist_transaction, on = 'card_id')


# In[12]:


train.groupby('observation_date').agg({'target': ['mean','count']})

