#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
import statsmodels.api as sm 
from scipy.interpolate import interp1d
import datetime as dt

pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


# In[2]:


osj = os.path.join
INPUT_DIR = '../input/m5-forecasting-accuracy/'
train = pd.read_csv(osj(INPUT_DIR, 'sales_train_validation.csv'))
price = pd.read_csv(osj(INPUT_DIR, 'sell_prices.csv'))
calender = pd.read_csv(osj(INPUT_DIR, 'calendar.csv'))

sample_submit = pd.read_csv(osj(INPUT_DIR, 'sample_submission.csv'))


# In[3]:


calender['date'] = pd.to_datetime(calender['date'])


# In[4]:


d_cols = [c for c in train.columns if 'd_' in c] # d_で始まる日付列のリスト
length = len(d_cols) # d_で始まる日付列の長さ
zero_rate = (train == 0).sum(axis=1)/length


# In[5]:


# ランダムに抽出（個数、random_stateは任意）
examples = train.sample(6, random_state=5050)
# calenderとマージ
examples = examples.set_index('id')[d_cols].T.merge(calender.set_index('d')['date'],left_index=True, right_index=True, validate='1:1').set_index('date')


# In[6]:


for item in examples.columns:
    examples[item].plot(title=item,
                        figsize=(15, 2),
                        color=next(color_cycle))
    plt.show()


# In[7]:


past_sales = train.set_index('id')[d_cols].T.merge(calender.set_index('d')['date'],
                                                   left_index=True,
                                                   right_index=True,
                                                   validate='1:1').set_index('date')


# In[8]:


past_sales.sum(axis=1).plot(figsize=(15, 5), alpha=0.8, title='Total Sales')
plt.show()


# In[9]:


state_list = train['state_id'].unique()
for i in state_list:
    items_col = [c for c in past_sales.columns if i in c]
    past_sales[items_col]         .sum(axis=1)         .rolling(30).mean()         .plot(figsize=(15, 5),
              alpha=0.8,
              title='Rolling 30 Day Average Total Sales by State')
plt.legend(train['state_id'].unique())
plt.show()


# In[10]:


train['cat_id'].unique()


# In[11]:


for i in train['cat_id'].unique():
    items_col = [c for c in past_sales.columns if i in c]
    past_sales[items_col]         .sum(axis=1)         .rolling(30).mean()         .plot(figsize=(15, 5),
              alpha=0.8,
              title='Rolling 30 Day Average Total Sales by Category')
plt.legend(train['cat_id'].unique())
plt.show()


# In[12]:


store_list = price['store_id'].unique()
store_list_ca = [s for s in store_list if 'CA' in s]
store_list_tx = [s for s in store_list if 'TX' in s]
store_list_wi = [s for s in store_list if 'WI' in s]


# In[13]:


for s in store_list_ca:
    store_items = [c for c in past_sales.columns if s in c]
    past_sales[store_items]         .sum(axis=1)         .rolling(30).mean()         .plot(figsize=(15, 5),
              ylim=[0,8000],
              alpha=0.8,
            color=next(color_cycle),
              title='Rolling 30 Day Average Total Sales (CA)')
plt.legend(store_list_ca)
plt.show()

for s in store_list_tx:
    store_items = [c for c in past_sales.columns if s in c]
    past_sales[store_items]         .sum(axis=1)         .rolling(30).mean()         .plot(figsize=(15, 5),
              ylim=[0,8000],
              color=next(color_cycle),
              alpha=0.8,
              title='Rolling 30 Day Average Total Sales (TX)')
plt.legend(store_list_tx)
plt.show()

for s in store_list_wi:
    store_items = [c for c in past_sales.columns if s in c]
    past_sales[store_items]         .sum(axis=1)         .rolling(30).mean()         .plot(figsize=(15, 5),
              ylim=[0,8000],
              color=next(color_cycle),
              alpha=0.8,
              title='Rolling 30 Day Average Total Sales (WI)')
plt.legend(store_list_wi)
plt.show()


# In[14]:


d_list = train['dept_id'].unique()
d_list_foods     = [d for d in d_list if 'FOODS' in d]
d_list_hobbies   = [d for d in d_list if 'HOBBIES' in d]
d_list_household = [d for d in d_list if 'HOUSEHOLD' in d]
d_list_h_h = d_list_hobbies + d_list_household


# In[15]:


l = d_list_foods
for st in state_list:
    for d in l:
        store_items = [c for c in past_sales.columns if st in c]
        store_d_items = [s for s in store_items if d in s]
        past_sales[store_d_items]             .sum(axis=1)             .rolling(30).mean()             .plot(figsize=(15, 5),
                  ylim=[0,10000],
                  alpha=0.8,
                  title=f'Rolling 30 Day Average FOODS Sales ({st})')
    plt.legend(l)
    plt.show()


# In[16]:


l = d_list_h_h
for st in state_list:
    for d in l:
        store_items = [c for c in past_sales.columns if st in c]
        store_d_items = [s for s in store_items if d in s]
        past_sales[store_d_items]             .sum(axis=1)             .rolling(30).mean()             .plot(figsize=(15, 5),
                  ylim=[0,10000],
                  alpha=0.8,
                  title=f'Rolling 30 Day Average HOBBIES & HOUSEHOLD Sales ({st})')
    plt.legend(l)
    plt.show()


# In[17]:


def make_total_sales_lowess( p ):
    total_sales = pd.DataFrame(p.sum(axis=1), columns=['total sales'])
    
    # クリスマス削除
    total_sales_noXmas = total_sales.drop(index=[dt.datetime(2011,12,25), 
                                                 dt.datetime(2012,12,25), 
                                                 dt.datetime(2013,12,25), 
                                                 dt.datetime(2014,12,25), 
                                                 dt.datetime(2015,12,25)])

    df = total_sales_noXmas
    lowess = sm.nonparametric.lowess(df['total sales'], df.index, frac=.3) 
    lowess_x = list(zip(*lowess))[0] 
    lowess_y = list(zip(*lowess))[1] 
    
    f = interp1d(lowess_x, lowess_y, bounds_error=False)
    new_lowess_x = df.index
    new_lowess_y = f(new_lowess_x)
    
    total_sales_lowess = total_sales_noXmas
    total_sales_lowess['lowess'] = new_lowess_y
    total_sales_lowess['total sales-lowess'] = total_sales_lowess['total sales'] - total_sales_lowess['lowess']
    
    return total_sales_lowess


# In[18]:


total_sales_lowess = make_total_sales_lowess( past_sales )


# In[19]:


total_sales_lowess.plot(figsize=(15, 5), alpha=0.8)


# In[20]:


store_list


# In[21]:


y_bt = -5000
y_up = 10000
for store in store_list :
    total_sales_lowess_tmp = make_total_sales_lowess( past_sales.loc[:, past_sales.columns.str.contains(store)] )
    total_sales_lowess_tmp.plot(figsize=(15, 5), alpha=0.8, ylim=[y_bt, y_up],title=store)


# In[ ]:




