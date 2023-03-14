#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import os
from subprocess import check_output

datadir = os.path.join('..','input')
print(check_output(['ls',datadir]).decode('utf8'))


# In[2]:


## Load order datasets
order_prior = pd.read_csv(os.path.join(datadir,'order_products__prior.csv'))
orders = pd.read_csv(os.path.join(datadir,'orders.csv'))


# In[3]:


## Get user_ids in testing set
test_users = orders[orders['eval_set']=='test']['user_id'].values

print('No. of test orders : {}'.format(test_users.shape[0]))
print('No. of unique user_ids : {}'.format(len(set(test_users))))


# In[4]:


## Get prior orders associated with the user_ids within the testing set
df_prior_orders = orders[(orders['user_id'].isin(test_users)) & (orders['eval_set']=='prior')]
prior_order_ids = df_prior_orders['order_id'].values

print('Number of prior orders: {}'.format(len(prior_order_ids)))
print('Number of unique prior order_ids : {}'.format(len(set(prior_order_ids))))


# In[5]:


## Test order_ids
df_test_orders = orders[(orders['user_id'].isin(test_users)) & (orders['eval_set']=='test')]


# In[6]:


# Get product_ids associated with the prior order_ids
df_prior_products = order_prior[order_prior['order_id'].isin(prior_order_ids)]

# Check that number of unique prior order_ids in df_prior_products
# matches with number of uniques prior order_ids in df_prior_orders
print('Number of unique prior order_ids : {}'.format(len(set(df_prior_products['order_id'].values))))


# In[7]:


# For this baseline model we will not make use of user_ids associated with training set
del orders
del order_prior


# In[8]:


# Merge df_prior_products and df_prior_orders
df_prior_products = df_prior_products.merge(
    df_prior_orders[['order_id','user_id','order_number']],
    left_on='order_id',right_on='order_id',how='left'
)


# In[9]:


## Number of products within each order
df_order_length = df_prior_products.groupby('order_id').size().reset_index()
df_order_length.rename(columns={0:'order_length'},inplace=True)

df_prior_products = df_prior_products.merge(
    df_order_length,left_on='order_id',right_on='order_id',how='left'
)

del df_order_length


# In[10]:


## Number of orders for each user 
df_number_orders = df_prior_products.groupby('user_id')['order_id'].nunique().reset_index()
df_number_orders.rename(columns={'order_id':'num_orders'},inplace=True)

df_prior_products = df_prior_products.merge(
    df_number_orders,left_on='user_id',right_on='user_id',how='left'
)

del df_number_orders


# In[11]:


# order importance = order_number/num_orders
# More recent orders have more weight
df_prior_products['order_importance'] = df_prior_products['order_number']/df_prior_products['num_orders']


# In[12]:


## product importance = (order_length-add_to_cart_order+1)/order_length
## Assign more importance to products that are added earlier to the cart for each order

df_prior_products['product_importance'] = (df_prior_products['order_length']-df_prior_products['add_to_cart_order']+1)/df_prior_products['order_length']


# In[13]:


## importance_score = product_importance * order_importance
## for each product-order pair

df_prior_products['importance_score'] = df_prior_products['product_importance']*df_prior_products['order_importance']
df_prior_products.head(10)


# In[14]:


## sum up the importance_scores for each product_id for each user_id
df_importance = df_prior_products.groupby(['user_id','product_id'])['importance_score'].sum()
df_importance = df_importance.reset_index()
df_importance.head(10)


# In[15]:


## The number of products to include in the testing set orders
## will be the mean order length of the prior orders for each user_id

## Average order length for each user_id
df_avg_order_length = df_prior_products.groupby('user_id')['order_length'].agg({
    'AvgOrderLength':lambda x:int(np.mean(x))
})

df_avg_order_length = df_avg_order_length.reset_index()

df_importance = df_importance.merge(
    df_avg_order_length,
    left_on='user_id',right_on='user_id',how='left'
)

del df_avg_order_length


# In[16]:


df_importance.head(10)


# In[17]:


## For each user_id, we select the top N products based on the importance_score,
## where N = average order length of prior orders for that user

df_selected_products = df_importance    .groupby('user_id')    .apply(lambda dfg : dfg.nlargest(dfg['AvgOrderLength'].values[0],'importance_score'))    .reset_index(drop=True)

# Check output using first user_id
df_selected_products[df_selected_products['user_id']==test_users[0]]


# In[18]:


# Match user_ids to corresponding test set order_id
df_submission = df_selected_products.merge(
    df_test_orders[['user_id','order_id']],
    left_on='user_id',right_on='user_id',how='left'
)


# In[19]:


# Drop user_id
df_submission.drop('user_id',axis=1,inplace=True)


# In[20]:


# Join product_ids into a string for each order_id
df_submission = df_submission    .groupby('order_id')['product_id']    .apply(lambda out : ' '.join([str(i) for i in out]))    .reset_index()

df_submission.columns = ['order_id','products']

# Check output for user_id #3 (order id #2774568)
df_submission[df_submission['order_id']==2774568]


# In[21]:


df_submission.to_csv('out.csv',index=False)

