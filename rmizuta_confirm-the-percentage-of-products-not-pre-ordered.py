#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd


# In[48]:


print('loading prior')
priors = pd.read_csv('../input/order_products__prior.csv')
print('loading train')
train = pd.read_csv('../input/order_products__train.csv')
print('loading orders')
orders = pd.read_csv('../input/orders.csv')
print('loading products')
products = pd.read_csv('../input/products.csv')

print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))
print('products {}: {}'.format(products.shape, ', '.join(products.columns)))


# In[49]:


user_product=pd.merge(orders,priors,on="order_id")
train_user_product=pd.merge(orders,train,on="order_id")


# In[50]:


user_product_list = pd.DataFrame(user_product.groupby('user_id')['product_id'].apply(list))
train_user_product_list = pd.DataFrame(train_user_product.groupby('user_id')['product_id'].apply(list))


# In[51]:


train_user_product_list.reset_index(inplace=True)


# In[52]:


user_product_list.reset_index(inplace=True)


# In[53]:


merged_product_list=pd.merge(train_user_product_list,user_product_list,on="user_id")


# In[54]:


merged_product_list


# In[59]:


def check_newproducts(row):
    before_products=frozenset(row["product_id_y"])
    train_products=frozenset(row["product_id_x"])
    new_products=train_products.difference(before_products)
    rate=len(new_products)/float(len(train_products))
    return rate
    
newproduct_rate=tmp.apply(lambda x:check_newproducts(x),axis=1)
newproduct_rate.mean()


# In[ ]:


about 40% products are not pre_ordered

