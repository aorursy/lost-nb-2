#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'


# In[2]:


train_df = pd.read_csv("../input/train.csv")
train_df.describe()


# In[3]:


macro_df = pd.read_csv("../input/macro.csv")
macro_df.describe()


# In[4]:


price = train_df['price_doc']
plt.figure(figsize=(8,4))
sns.distplot(price, kde=False)


# In[5]:


ulimit = np.percentile(train_df.price_doc.values, 99)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
price = train_df['price_doc']
plt.figure(figsize=(8,4))
sns.distplot(price, kde=False)


# In[6]:


corrmat = train_df.corr()
n = 15
cols = corrmat.nlargest(n, 'price_doc')['price_doc'].index
cm_df = train_df[cols].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cm_df, square=True, annot=True, fmt='.2f', annot_kws={'size':10}, cbar=True)


# In[7]:


var = 'full_sq'
data = pd.concat([train_df['price_doc'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='price_doc')


# In[8]:


ulimit = np.percentile(train_df['full_sq'].values, 99.9)
trimmed_df = train_df.drop(train_df[train_df['full_sq']>ulimit].index)

data = pd.concat([trimmed_df['price_doc'], trimmed_df[var]], axis=1)
data.plot.scatter(x=var, y='price_doc')
train_df = trimmed_df


# In[9]:


dlimit = np.percentile(train_df['full_sq'].values, 0.1)
trimmed_df = train_df.drop(train_df[train_df['full_sq']<dlimit].index)

data = pd.concat([trimmed_df['price_doc'], trimmed_df[var]], axis=1)
data.plot.scatter(x=var, y='price_doc')
train_df = trimmed_df


# In[10]:


total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Precent'])
missing_data.head(50)


# In[11]:


g_p_type = train_df.groupby('product_type').mean()['price_doc']
plt.figure(figsize=(8,4))
sns.barplot(g_p_type.index, g_p_type.values)
plt.ylabel('price_doc')
plt.show()


# In[12]:


g_p_type = train_df['product_type'].value_counts()
plt.figure(figsize=(8,4))
sns.barplot(g_p_type.index, g_p_type.values)
plt.ylabel('Number of Occurrences')
plt.show()


# In[13]:


Wow! People like to invest in real estate.  
get_ipython().set_next_input('How about the sub_area');get_ipython().run_line_magic('pinfo', 'sub_area')


# In[14]:


sub_area_list = train_df.groupby('sub_area').mean()['price_doc'].sort_values(ascending=False)[:15]
plt.figure(figsize=(8,4))
sns.barplot(sub_area_list.index, sub_area_list.values)
plt.ylabel('price_doc')
plt.xticks(rotation=70)
plt.show()


# In[15]:


sub_area_list = train_df.groupby('sub_area').mean()['price_doc'].sort_values(ascending=True)[:15]
plt.figure(figsize=(8,4))
sns.barplot(sub_area_list.index, sub_area_list.values)
plt.ylabel('price_doc')
plt.xticks(rotation=70)
plt.show()

