#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


orprod_prior = pd.read_csv("../input/order_products__prior.csv", 
                    dtype={'order_id': np.int32, 'product_id': np.int32, 'add_to_cart_order': np.int8, 'reordered': np.int8})
orprod_train = pd.read_csv("../input/order_products__train.csv",
                    dtype={'order_id': np.int32, 'product_id': np.int32, 'add_to_cart_order': np.int8, 'reordered': np.int8})

orders = pd.read_csv("../input/orders.csv",
                    dtype={'order_id': np.int32, 'user_id': np.int32, 'order_number': np.int8, 'order_dow': np.int8,
                          'order_hour_of_day': np.int8})

products = pd.read_csv("../input/products.csv")


# In[3]:


print ("{} unique customers".format(len(orders['user_id'].unique())))


# In[4]:


user_orders = orders.groupby(['user_id']).size()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
plt.suptitle('Customer number of orders', fontsize=16)
sns.distplot(user_orders.values, kde=False, ax=ax1)

sns.boxplot(user_orders.values, ax=ax2)
ax2.set_xlim(0,50)
ax2.set_xticks(range(0,105,5))
plt.show()


# In[5]:


plt.hist(user_orders[user_orders >= 20], bins=100)
plt.title('Customers - 20 or more orders', fontsize=14)
plt.xlabel('Number of Orders')
plt.show()


# In[6]:





# In[6]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,8))
plt.suptitle('Orders', fontsize=16)
ax1.set_xlabel('Day of Week')
sns.countplot(orders['order_dow'].values, ax=ax1)

plt.xlabel('Hour of Day')
sns.countplot(orders['order_hour_of_day'].values)
plt.show()


# In[7]:


orders_daytime = orders.groupby(['order_dow', 'order_hour_of_day'])['order_dow'].agg(['count']).reset_index()

fig, ax = plt.subplots()
labels = []

for i, group in orders_daytime.groupby('order_dow'):
    ax = group.plot(ax=ax, x='order_hour_of_day', y='count')
    labels.append(i)

lines, _ = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.suptitle('Orders by day and time', fontsize=14)
plt.xlabel('Hour of Day')
plt.ylabel('Orders')
plt.xticks(range(0,24))

plt.show()


# In[8]:


orfreq = orders[orders['days_since_prior_order'].notnull()].groupby(['user_id'])['days_since_prior_order'].agg(['median']).reset_index()

plt.title('Median days between orders', fontsize=14)
sns.distplot(orfreq['median'], kde=False)
plt.xticks(range(0,31,1))
plt.xlabel('Days')
plt.show()

sns.boxplot(orfreq['median'])
plt.xticks(range(0,31,1))
plt.xlabel('Days')
plt.show()


# In[9]:


orprod = pd.concat([orprod_prior, orprod_train], ignore_index=True)
num_prods = orprod.groupby(['order_id'])['order_id'].count()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,8))
plt.suptitle('Number of Products in Order', fontsize=16)
sns.distplot(num_prods.values, kde=False, ax=ax1)

sns.boxplot(num_prods.values, ax=ax2)
ax2.set_xlim(0,40)
ax2.set_xticks(range(0,41,1))
plt.show()


# In[10]:


top_prod = orprod.groupby(['product_id'])['product_id'].agg(['count']).sort_values(by='count', ascending=False).reset_index()
top_prod_named = top_prod[:20].merge(products[['product_id', 'product_name', 'department_id']], how='left', on=['product_id']).reset_index()

top_prod_named.sort_values(by='count').plot(kind='barh', x='product_name', y='count')
plt.title('Top 20 Products', fontsize=14)
plt.ylabel('Product')
plt.xlabel('Total quantity')
plt.show()


# In[11]:


reorder_prods = orprod[orprod['reordered'] == 1].drop(['add_to_cart_order','reordered'], axis=1)
reorders = orders[orders['eval_set'] != 'test'].merge(reorder_prods[['order_id', 'product_id']], how='right', on=['order_id'])
reorders = reorders.merge(products[['product_id', 'product_name', 'department_id']], how='left', on=['product_id'])            .drop(['eval_set','order_dow','order_hour_of_day','days_since_prior_order'], axis=1).drop_duplicates(subset=['user_id', 'product_id'])

reorders.groupby(['product_name'])['user_id'].agg(['count']).sort_values(by='count',ascending=False)[:20].sort_values(by='count').plot(kind='barh')
plt.title('Top 20 Reordered Products', fontsize=14)
plt.ylabel('Product')
plt.xlabel('Users That Have Reordered')
plt.show()


# In[12]:


top_orprod = orprod[orprod['product_id'].isin(top_prod[:20].product_id.values)]

top_orprod_dow = top_orprod.merge(orders[['order_id','order_dow']], how='left', on=['order_id'])
top_orprod_dow = top_orprod_dow.groupby(['order_dow','product_id'])['product_id'].size().reset_index(name='count')
top_orprod_dow = top_orprod_dow.merge(products[['product_id', 'product_name']], how='left', on=['product_id']).drop(['product_id'], axis=1)

dow_prod_sales=top_orprod_dow.pivot(index='product_name', columns='order_dow',values='count')
totals = dow_prod_sales.sum(axis=1)

for c in dow_prod_sales.columns.values:
    dow_prod_sales[c] = dow_prod_sales[c]/totals
    
dow_prod_sales.plot.barh(stacked=True)
plt.legend(loc='upper left', bbox_to_anchor=(1.1,1), title='Day of Week')
plt.title('Top 20 Products: Sales by Day of Week', fontsize=14)
plt.xlabel('Proportion of Sales')
plt.ylabel('Product')
plt.show()    


# In[13]:


first_prod = orprod[orprod['add_to_cart_order'] == 1]    .groupby(['product_id'])['product_id'].agg(['count']).reset_index()    .sort_values(by='count', ascending=False)[:20]

first_prod = first_prod.merge(products[['product_id', 'product_name']], how='left', on=['product_id']).rename(columns={'count':'orders'})
first_prod.sort_values(by='orders').plot(kind='barh', x='product_name', y='orders')

plt.title('Items added to cart first', fontsize=14)
plt.ylabel('Product')
plt.xlabel('')
plt.show()


# In[14]:


top_prod = top_prod[:75].drop(['count'], axis=1)

top_n_orprod = orprod[orprod['product_id'].isin(top_prod.product_id.values)].drop(['add_to_cart_order', 'reordered'], axis=1)
top_n_orprod = top_n_orprod.merge(products[['product_id', 'product_name']], how='left', on=['product_id']).drop(['product_id'], axis=1)

top_n_orprod['ordered'] = 1
orders_top_products = top_n_orprod.pivot(index='order_id', columns='product_name', values='ordered').fillna(value=0)

topcor = orders_top_products.corr().abs().nlargest(30, 'Banana').index
corr_matrix = np.corrcoef(orders_top_products[topcor].values.T)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, yticklabels=topcor.values, xticklabels=topcor.values, vmax=0.2)
plt.show()


# In[15]:




