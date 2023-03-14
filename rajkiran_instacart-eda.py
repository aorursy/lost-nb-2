#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display

import numpy as np
import pandas as pd
#from pandasql import sqldf


READ_MODE = True  # Turn this on to read the data afresh from csv sources
DEBUG_MODE = False  # Turn this on to run various data consistency checks

pysqldf = lambda q: sqldf(q, globals())  # Run sql queries on pandas DataFrames


# In[2]:


if READ_MODE: 
    order_products__prior = pd.read_csv('./data/order_products__prior.csv')


# In[3]:


display(order_products__prior.head(5),         order_products__prior['order_id'].value_counts()[::len(order_products__prior)//100],         order_products__prior.shape)


# In[4]:


if READ_MODE:
    order_products__train = pd.read_csv('./data/order_products__train.csv')


# In[5]:


display(order_products__train.head(5),         order_products__train['order_id'].value_counts()[::len(order_products__train)//100],         order_products__train.shape)


# In[6]:


if DEBUG_MODE:
    assert not set(order_products__prior['order_id']) & set(order_products__train['order_id'])


# In[7]:


if READ_MODE:
    orders = pd.read_csv('./data/orders.csv')


# In[8]:


display(orders.head(),         orders.shape,        orders['eval_set'].value_counts())


# In[9]:


if DEBUG_MODE:
    assert not set(orders.loc[orders['eval_set'] == 'prior', 'order_id']) - set(order_products__prior['order_id'])
    assert not set(orders.loc[orders['eval_set'] == 'train', 'order_id']) - set(order_products__train['order_id'])


# In[10]:


if DEBUG_MODE:
    assert np.all(orders[orders['eval_set'] == 'test'].groupby(['user_id'])['order_id'].count() == 1)
    assert np.all(orders[orders['eval_set'] == 'train'].groupby(['user_id'])['order_id'].count() == 1)


# In[11]:


# nr_prior_orders = orders.groupby(['user_id', 'order_id'], as_index=False)['order_number'].max()
# nr_prior_orders.rename(columns={'order_number':'nr_prior_orders_perUser_perProduct'}, inplace=True)
# nr_prior_orders.head()


# In[12]:


# orders = orders.merge(nr_prior_orders, how='inner', on=['user_id', 'order_id'])
# del nr_prior_orders
# orders.head()


# In[13]:


train = order_products__train.merge(orders, on='order_id', how='left', copy=False)
if DEBUG_MODE:
    assert np.all(train['eval_set'] == 'train')
# q = """
# SELECT order_products__train.order_id
# FROM order_products__train LEFT JOIN orders ON order_products__train.order_id = orders.order_id
# WHERE eval_set = "train"
# """


# In[14]:


train.rename(columns={'reordered':'x1: reordered'}, inplace=True)
train.head()


# In[15]:


nr_prior_orders = order_products__prior.merge(orders, how='left', on='order_id').groupby(by=['user_id', 'product_id'], as_index=False)['order_id'].count()         


# In[16]:


nr_prior_orders.rename(columns={'order_id':'nr_prior_orders_perUserProduct'}, inplace=True)
nr_prior_orders.head()


# In[17]:


train = train.merge(nr_prior_orders, how='left', on=['user_id', 'product_id'])
train.rename(columns={'nr_prior_orders_perUserProduct':'x2: nr_prior_orders_perUserProduct',                       'days_since_prior_order': 'x3: days_since_prior_order'}, inplace=True)
train.head()


# In[18]:


# Checks
# orders[orders['user_id'] == 789].sort_values(by='order_number')
if DEBUG_MODE:
    np.all(train.groupby(by='user_id')['days_since_prior_order'].nunique() == 1)


# In[19]:


if READ_MODE:
    products = pd.read_csv('./data/products.csv')


# In[20]:


train = train.merge(products, how='left', on='product_id')


# In[21]:


train = pd.get_dummies(train, columns=['order_dow', 'order_hour_of_day', 'aisle_id', 'department_id'], 
              prefix={'order_dow':'x4: dow', 'order_hour_of_day':'x5: hod', 'aisle_id':'x6: aisle', 'department_id':'x7: dept'})


# In[22]:


train.columns.values


# In[23]:


train.head()

