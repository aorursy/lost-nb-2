#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.preprocessing import StandardScaler 
from scipy.stats import describe
get_ipython().run_line_magic('matplotlib', 'inline')
import lightgbm as lgb
from sklearn.linear_model import Ridge
import time
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error
import xgboost as xgb

import plotly.offline as py
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")

pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)

import gc
from sklearn.tree import DecisionTreeRegressor
#from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, train_test_split

from sklearn.model_selection import RepeatedKFold 
from sklearn.ensemble import ExtraTreesRegressor
import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,BaggingRegressor
from sklearn.pipeline import make_pipeline
#from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
#from sklearn.linear_model import Lasso

#from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error 
from xgboost import XGBRegressor


# In[2]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


nmt = reduce_mem_usage(pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/new_merchant_transactions.csv',parse_dates=['purchase_date']))
ht = reduce_mem_usage(pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/historical_transactions.csv',parse_dates=['purchase_date']))


# In[4]:


def cleaning(df):
  scaler = StandardScaler()
  for col in ['authorized_flag', 'category_1']:
    df[col] = df[col].map({'Y':1, 'N':0})  
    df[col] = df[col].apply(pd.to_numeric, errors='coerce')
  for col in ['installments']:
    df[col] = df[col].map({-1:14, 0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,999:13})
    df[col] = df[col].apply(pd.to_numeric, errors='coerce')
  for col in ['category_3']:
    df[col] = df[col].map({'A':1, 'B':2,'C':3})
    df[col] = df[col].apply(pd.to_numeric, errors='coerce')
  for col in ['category_2']:
    df[col] = df[col].apply(pd.to_numeric, errors='coerce')   
  for col in ['purchase_amount']:        
    df[col] = scaler.fit_transform(df[[col]])      
  return df

ht = cleaning(ht)
nmt = cleaning(nmt)

from datetime import datetime
# Missing values handling
for df in [ht, nmt]: # Filling with most common value
  df['category_2'].fillna(1,inplace=True)
  df['category_3'].fillna(1,inplace=True)
  df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
  # Purchase date - year, month, week, hour
  df['purchase_date'] = pd.to_datetime(df['purchase_date'])
  df['year'] = df['purchase_date'].dt.year
  df['weekofyear'] = df['purchase_date'].dt.weekofyear
  df['weekday'] = df['purchase_date'].dt.weekday
  df['month'] = df['purchase_date'].dt.month
  df['dayofweek'] = df['purchase_date'].dt.dayofweek
  df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
  df['hour'] = df['purchase_date'].dt.hour  
  df['month_diff'] = ((datetime.today() - df['purchase_date']).dt.days)//30
  df['month_diff'] += df['month_lag']

  df['duration'] = df['purchase_amount']*df['month_diff']
  df['amount_month_ratio'] = df['purchase_amount']/df['month_diff']
  df['price'] = df['purchase_amount'] / df['installments']

gc.collect()
len(ht.columns), len(nmt.columns)


# In[5]:


from datetime import datetime
# Here we are trying to calculate recency, frequency, monetary and age.
# Recency is how many days back did customer perform a last transaction.
# Frequency is how many transactions are performed in time period from dataset.
# Monetary is how much was spent in all the transactions.

hist = ht[['card_id','purchase_date','purchase_amount']]
hist = hist.sort_values(by=['card_id', 'purchase_date'], ascending=[True, True])
print(hist.head())

z = hist.groupby('card_id')['purchase_date'].max().reset_index()
q = hist.groupby('card_id')['purchase_date'].min().reset_index()

z.columns = ['card_id', 'Max']
q.columns = ['card_id', 'Min']

## Extracting current timestamp
now = datetime.now()
curr_date = now.strftime("%m-%d-%Y, %H:%M:%S")
curr_date = pd.to_datetime(curr_date)

rec = pd.merge(z,q,how = 'left',on = 'card_id')
rec['Min'] = pd.to_datetime(rec['Min'])
rec['Max'] = pd.to_datetime(rec['Max'])

## Recency value 
rec['Recency'] = (curr_date - rec['Max']).astype('timedelta64[D]') ## current date - most recent date

## Age value
rec['Age'] = (rec['Max'] - rec['Min']).astype('timedelta64[D]') ## Age of customer, MAX - MIN

rec = rec[['card_id','Age','Recency']]

## Frequency
freq = hist.groupby('card_id').size().reset_index()
freq.columns = ['card_id', 'Frequency']

## Monetary
mon = hist.groupby('card_id')['purchase_amount'].sum().reset_index()
mon.columns = ['card_id', 'Monetary']

final = pd.merge(freq,mon,how = 'left', on = 'card_id')
final = pd.merge(final,rec,how = 'left', on = 'card_id')

final['AvOrderValue'] = final['Monetary']/final['Frequency'] ## AOV - Average order value (i.e) total_purchase_amt/total_trans
final['AgeRecencyRatio'] = final['Age']/final['Recency'] ## 
final = final[['card_id','AvOrderValue','AgeRecencyRatio']]
final.head()


# In[6]:


def aggregate_transactions_hist(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = { 
    'purchase_date': ['max', 'min'],            
    'merchant_category_id': ['nunique'],
    'month': ['mean','nunique','min'],     
    'hour': ['mean','min','max'],    
    'weekofyear': ['mean','nunique','min','max'],     
    'month_diff': ['mean'],     
    'weekend': ['sum'],
    'weekday': ['sum','mean'],   
    'card_id': ['size','count'],
    'category_1' : ['sum', 'mean'],   
    'purchase_amount': ['sum', 'mean', 'max', 'min','median'],
    'installments': ['sum', 'mean', 'max', 'min','median'],     
    'authorized_flag': ['sum'],
    'subsector_id': ['nunique'],
    'month_lag': ['mean'],
    'price' :['sum','mean','min','var'],
    'duration' : ['mean','min','max','var','skew'],
    'amount_month_ratio':['mean','min','max','var','skew']        
    }
    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size().reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')    
    return agg_history
  
history = aggregate_transactions_hist(ht)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]

agg_func = {'mean': ['mean'], }

for col in ['category_2','category_3']:
    history[col+'_mean'] = ht['purchase_amount'].groupby(ht[col]).agg('mean')
    history[col+'_max'] = ht['purchase_amount'].groupby(ht[col]).agg('max')
    history[col+'_min'] = ht['purchase_amount'].groupby(ht[col]).agg('min')
    history[col+'_sum'] = ht['purchase_amount'].groupby(ht[col]).agg('sum')
    agg_func[col+'_mean'] = ['mean']
    
gc.collect()
len(ht.columns), len(history.columns)


# In[7]:


def aggregate_transaction_new(trans):  
        
    agg_func = {
        'purchase_amount' : ['sum','max','min','mean'],
        'installments' : ['sum','max','mean'],
        'purchase_date' : ['max','min'],
        'month_lag' : ['mean'],
        'month_diff' : ['mean'],
        'weekend' : ['sum'],
        'weekday' : ['sum', 'mean'],
        'authorized_flag': ['sum'],
        'category_1': ['sum','mean'],
        'card_id' : ['size', 'count'],
        'month': ['nunique', 'mean', 'min'],
        'hour': ['mean', 'min', 'max'],
        'weekofyear': ['nunique', 'mean', 'min', 'max'],       
        'subsector_id': ['nunique'],
        'merchant_category_id' : ['nunique'],
        'price' :['sum','mean','min','var'],
        'duration' : ['mean','min','max','var','skew'],
        'amount_month_ratio':['mean','min','max','var','skew']
    }
    
    agg_history = trans.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size().reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')    
    return agg_history


new = aggregate_transaction_new(nmt)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]

agg_func = {'mean': ['mean'], }

for col in ['category_2','category_3']:
    new[col+'_mean'] = nmt['purchase_amount'].groupby(ht[col]).agg('mean')
    new[col+'_max'] = nmt['purchase_amount'].groupby(nmt[col]).agg('max')
    new[col+'_min'] = nmt['purchase_amount'].groupby(nmt[col]).agg('min')
    new[col+'_sum'] = nmt['purchase_amount'].groupby(nmt[col]).agg('sum')
    agg_func[col+'_mean'] = ['mean']
    
gc.collect()
len(new.columns)


# In[8]:


test = reduce_mem_usage(pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/test.csv'))
print(test.isnull().sum())
date_time_str = '2017-04-01'
print(test[test['first_active_month'].isnull() ] )
test.loc[11578,'first_active_month'] = date_time_str
print(test.isnull().sum() )
test['first_active_month'] = test['first_active_month'].astype('datetime64[ns]')
test.info()


# In[9]:


train = reduce_mem_usage(pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/train.csv',parse_dates=['first_active_month']))

# Merge all dataframes based on card_id
train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')

train = pd.merge(train, final, on='card_id', how='left')
test = pd.merge(test, final, on='card_id', how='left')

train = pd.merge(train, new, on='card_id', how='left')
test = pd.merge(test, new, on='card_id', how='left')
train.shape, test.shape


# In[10]:


#Feature Engineering - Adding new features inspired by Chau's first kernel
train['new_purchase_date_max'] = pd.to_datetime(train['new_purchase_date_max'])
train['new_purchase_date_min'] = pd.to_datetime(train['new_purchase_date_min'])
train['new_purchase_date_diff'] = (train['new_purchase_date_max'] - train['new_purchase_date_min']).dt.days
train['new_purchase_date_average'] = train['new_purchase_date_diff']/train['new_card_id_size']
train['new_purchase_date_uptonow'] = (datetime.today() - train['new_purchase_date_max']).dt.days
train['new_purchase_date_uptomin'] = (datetime.today() - train['new_purchase_date_min']).dt.days
train['new_first_buy'] = (train['new_purchase_date_min'] - train['first_active_month']).dt.days
train['new_last_buy'] = (train['new_purchase_date_max'] - train['first_active_month']).dt.days


#Feature Engineering - Adding new features inspired by Chau's first kernel
test['new_purchase_date_max'] = pd.to_datetime(test['new_purchase_date_max'])
test['new_purchase_date_min'] = pd.to_datetime(test['new_purchase_date_min'])
test['new_purchase_date_diff'] = (test['new_purchase_date_max'] - test['new_purchase_date_min']).dt.days
test['new_purchase_date_average'] = test['new_purchase_date_diff']/test['new_card_id_size']
test['new_purchase_date_uptonow'] = (datetime.today() - test['new_purchase_date_max']).dt.days
test['new_purchase_date_uptomin'] = (datetime.today() - test['new_purchase_date_min']).dt.days
test['new_first_buy'] = (test['new_purchase_date_min'] - test['first_active_month']).dt.days
test['new_last_buy'] = (test['new_purchase_date_max'] - test['first_active_month']).dt.days

gc.collect()


# In[11]:


train['card_id_total'] = train['new_card_id_size']+train['hist_card_id_size']
train['card_id_cnt_total'] = train['new_card_id_count']+train['hist_card_id_count']
train['card_id_cnt_ratio'] = train['new_card_id_count']/train['hist_card_id_count']
train['purchase_amount_total'] = train['new_purchase_amount_sum']+train['hist_purchase_amount_sum']
train['purchase_amount_mean'] = train['new_purchase_amount_mean']+train['hist_purchase_amount_mean']
train['purchase_amount_max'] = train['new_purchase_amount_max']+train['hist_purchase_amount_max']
train['purchase_amount_min'] = train['new_purchase_amount_min']+train['hist_purchase_amount_min']
train['purchase_amount_ratio'] = train['new_purchase_amount_sum']/train['hist_purchase_amount_sum']
train['month_diff_mean'] = train['new_month_diff_mean']+train['hist_month_diff_mean']
train['month_diff_ratio'] = train['new_month_diff_mean']/train['hist_month_diff_mean']
train['month_lag_mean'] = train['new_month_lag_mean']+train['hist_month_lag_mean']
#train['month_lag_max'] = train['new_month_lag_max']+train['hist_month_lag_max']
#train['month_lag_min'] = train['new_month_lag_min']+train['hist_month_lag_min']
train['category_1_mean'] = train['new_category_1_mean']+train['hist_category_1_mean']
#train['category_2_mean'] = train['new_category_2_mean']+train['hist_category_2_mean']
#train['category_3_mean'] = train['new_category_3_mean']+train['hist_category_3_mean']


train['installments_total'] = train['new_installments_sum']+train['hist_installments_sum']
train['installments_mean'] = train['new_installments_mean']+train['hist_installments_mean']
train['installments_max'] = train['new_installments_max']+train['hist_installments_max']
train['installments_ratio'] = train['new_installments_sum']/train['hist_installments_sum']
train['price_total'] = train['purchase_amount_total'] / train['installments_total']
train['price_mean'] = train['purchase_amount_mean'] / train['installments_mean']
train['price_max'] = train['purchase_amount_max'] / train['installments_max']
train['duration_mean'] = train['new_duration_mean']+train['hist_duration_mean']
train['duration_min'] = train['new_duration_min']+train['hist_duration_min']
train['duration_max'] = train['new_duration_max']+train['hist_duration_max']
train['amount_month_ratio_mean']=train['new_amount_month_ratio_mean']+train['hist_amount_month_ratio_mean']
train['amount_month_ratio_min']=train['new_amount_month_ratio_min']+train['hist_amount_month_ratio_min']
train['amount_month_ratio_max']=train['new_amount_month_ratio_max']+train['hist_amount_month_ratio_max']
train['new_CLV'] = train['new_card_id_count'] * train['new_purchase_amount_sum'] / train['new_month_diff_mean']
train['hist_CLV'] = train['hist_card_id_count'] * train['hist_purchase_amount_sum'] / train['hist_month_diff_mean']
train['CLV_ratio'] = train['new_CLV'] / train['hist_CLV']

#test['card_id_total'] = test['new_card_id_size']+test['hist_card_id_size']
#test['card_id_cnt_total'] = test['new_card_id_count']+test['hist_card_id_count']
#test['card_id_cnt_ratio'] = test['new_card_id_count']/test['hist_card_id_count']
test['purchase_amount_total'] = test['new_purchase_amount_sum']+test['hist_purchase_amount_sum']
test['purchase_amount_mean'] = test['new_purchase_amount_mean']+test['hist_purchase_amount_mean']
test['purchase_amount_max'] = test['new_purchase_amount_max']+test['hist_purchase_amount_max']
test['purchase_amount_min'] = test['new_purchase_amount_min']+test['hist_purchase_amount_min']
test['purchase_amount_ratio'] = test['new_purchase_amount_sum']/test['hist_purchase_amount_sum']
test['month_diff_mean'] = test['new_month_diff_mean']+test['hist_month_diff_mean']
test['month_diff_ratio'] = test['new_month_diff_mean']/test['hist_month_diff_mean']
test['month_lag_mean'] = test['new_month_lag_mean']+test['hist_month_lag_mean']
#test['month_lag_max'] = test['new_month_lag_max']+test['hist_month_lag_max']
#test['month_lag_min'] = test['new_month_lag_min']+test['hist_month_lag_min']
test['category_1_mean'] = test['new_category_1_mean']+test['hist_category_1_mean']
test['installments_total'] = test['new_installments_sum']+test['hist_installments_sum']
test['installments_mean'] = test['new_installments_mean']+test['hist_installments_mean']
test['installments_max'] = test['new_installments_max']+test['hist_installments_max']
test['installments_ratio'] = test['new_installments_sum']/test['hist_installments_sum']
test['price_total'] = test['purchase_amount_total'] / test['installments_total']
test['price_mean'] = test['purchase_amount_mean'] / test['installments_mean']
test['price_max'] = test['purchase_amount_max'] / test['installments_max']
test['duration_mean'] = test['new_duration_mean']+test['hist_duration_mean']
test['duration_min'] = test['new_duration_min']+test['hist_duration_min']
test['duration_max'] = test['new_duration_max']+test['hist_duration_max']
test['amount_month_ratio_mean']=test['new_amount_month_ratio_mean']+test['hist_amount_month_ratio_mean']
test['amount_month_ratio_min']=test['new_amount_month_ratio_min']+test['hist_amount_month_ratio_min']
test['amount_month_ratio_max']=test['new_amount_month_ratio_max']+test['hist_amount_month_ratio_max']
test['new_CLV'] = test['new_card_id_count'] * test['new_purchase_amount_sum'] / test['new_month_diff_mean']
test['hist_CLV'] = test['hist_card_id_count'] * test['hist_purchase_amount_sum'] / test['hist_month_diff_mean']
test['CLV_ratio'] = test['new_CLV'] / test['hist_CLV']


# In[12]:


print(train.shape, test.shape)
train_back = train.copy()
test_back = test.copy()

train = train.drop(['card_id', 'first_active_month'], axis = 1)
test = test.drop(['card_id', 'first_active_month'], axis = 1)
print(train.shape, test.shape)


# In[13]:


train["new_transactions_count"].fillna(train["new_transactions_count"].median(),inplace=True)
train["new_purchase_amount_sum"].fillna(train["new_purchase_amount_sum"].median(),inplace=True)
train["new_purchase_amount_max"].fillna(train["new_purchase_amount_max"].median(),inplace=True)
train["new_purchase_amount_min"].fillna(train["new_purchase_amount_min"].median(),inplace=True)
train["new_purchase_amount_mean"].fillna(train["new_purchase_amount_mean"].median(),inplace=True)
train["new_installments_sum"].fillna(train["new_installments_sum"].median(),inplace=True)
train["new_installments_max"].fillna(train["new_installments_max"].median(),inplace=True)
train["new_installments_mean"].fillna(train["new_installments_mean"].median(),inplace=True)
train["new_purchase_date_max"].fillna(train["new_purchase_date_max"].mean(),inplace=True)
train["new_purchase_date_min"].fillna(train["new_purchase_date_min"].mean(),inplace=True)
#train["new_month_lag_max"].fillna(train["new_month_lag_max"].median(),inplace=True)
#train["new_month_lag_min"].fillna(train["new_month_lag_min"].median(),inplace=True)
train["new_month_lag_mean"].fillna(train["new_month_lag_mean"].median(),inplace=True)
#train["new_month_diff_max"].fillna(train["new_month_diff_max"].median(),inplace=True)
#train["new_month_diff_min"].fillna(train["new_month_diff_min"].median(),inplace=True)
train["new_month_diff_mean"].fillna(train["new_month_diff_mean"].median(),inplace=True)
train["new_weekend_sum"].fillna(train["new_weekend_sum"].median(),inplace=True)
#train["new_weekend_mean"].fillna(train["new_weekend_mean"].median(),inplace=True)
train["new_weekday_sum"].fillna(train["new_weekday_sum"].median(),inplace=True)
train["new_weekday_mean"].fillna(train["new_weekday_mean"].median(),inplace=True)
train["new_authorized_flag_sum"].fillna(train["new_authorized_flag_sum"].median(),inplace=True)
#train["new_authorized_flag_mean"].fillna(train["new_authorized_flag_mean"].median(),inplace=True)
train["new_category_1_sum"].fillna(train["new_category_1_sum"].median(),inplace=True)
train["new_category_1_mean"].fillna(train["new_category_1_mean"].median(),inplace=True)
#train["new_category_1_max"].fillna(train["new_category_1_max"].median(),inplace=True)
#train["new_category_1_min"].fillna(train["new_category_1_min"].median(),inplace=True)

train["new_card_id_size"].fillna(train["new_card_id_size"].median(),inplace=True)
train["new_card_id_count"].fillna(train["new_card_id_count"].median(),inplace=True)
train["new_month_nunique"].fillna(train["new_month_nunique"].median(),inplace=True)
train["new_month_mean"].fillna(train["new_month_mean"].median(),inplace=True)
train["new_month_min"].fillna(train["new_month_min"].median(),inplace=True)
#train["new_month_max"].fillna(train["new_month_max"].median(),inplace=True)
#train["new_hour_nunique"].fillna(train["new_hour_nunique"].median(),inplace=True)
train["new_hour_mean"].fillna(train["new_hour_mean"].median(),inplace=True)
train["new_hour_min"].fillna(train["new_hour_min"].median(),inplace=True)
train["new_hour_max"].fillna(train["new_hour_max"].median(),inplace=True)

train["new_weekofyear_nunique"].fillna(train["new_weekofyear_nunique"].median(),inplace=True)
train["new_weekofyear_max"].fillna(train["new_weekofyear_max"].median(),inplace=True)
train["new_weekofyear_mean"].fillna(train["new_weekofyear_mean"].median(),inplace=True)
train["new_weekofyear_min"].fillna(train["new_weekofyear_min"].median(),inplace=True)

train["new_subsector_id_nunique"].fillna(train["new_subsector_id_nunique"].median(),inplace=True)
train["new_merchant_category_id_nunique"].fillna(train["new_merchant_category_id_nunique"].median(),inplace=True)

train["new_price_sum"].fillna(train["new_price_sum"].median(),inplace=True)
train["new_price_min"].fillna(train["new_price_min"].median(),inplace=True)
#train["new_price_max"].fillna(train["new_price_max"].median(),inplace=True)
train["new_price_mean"].fillna(train["new_price_mean"].median(),inplace=True)
train["new_price_var"].fillna(train["new_price_var"].median(),inplace=True)

train["new_duration_mean"].fillna(train["new_duration_mean"].median(),inplace=True)
train["new_duration_min"].fillna(train["new_duration_min"].median(),inplace=True)
train["new_duration_max"].fillna(train["new_duration_max"].median(),inplace=True)
train["new_duration_var"].fillna(train["new_duration_var"].median(),inplace=True)
train["new_duration_skew"].fillna(train["new_duration_skew"].median(),inplace=True)

train["new_amount_month_ratio_mean"].fillna(train["new_amount_month_ratio_mean"].median(),inplace=True)
train["new_amount_month_ratio_min"].fillna(train["new_amount_month_ratio_min"].median(),inplace=True)
train["new_amount_month_ratio_max"].fillna(train["new_amount_month_ratio_max"].median(),inplace=True)
train["new_amount_month_ratio_var"].fillna(train["new_amount_month_ratio_var"].median(),inplace=True)
train["new_amount_month_ratio_skew"].fillna(train["new_amount_month_ratio_skew"].median(),inplace=True)

train["category_1_mean"].fillna(train["category_1_mean"].median(),inplace=True)
#train["category_2_mean"].fillna(train["category_2_mean"].median(),inplace=True)
#train["category_3_mean"].fillna(train["category_3_mean"].median(),inplace=True)

train["category_2_min_y"].fillna(train["category_2_min_y"].median(),inplace=True)
train["category_2_mean_y"].fillna(train["category_2_mean_y"].median(),inplace=True)
train["category_2_max_y"].fillna(train["category_2_max_y"].median(),inplace=True)
train["category_2_sum_y"].fillna(train["category_2_sum_y"].median(),inplace=True)

train["category_2_min_x"].fillna(train["category_2_min_y"].median(),inplace=True)
train["category_2_mean_x"].fillna(train["category_2_mean_y"].median(),inplace=True)
train["category_2_max_x"].fillna(train["category_2_max_y"].median(),inplace=True)
train["category_2_sum_x"].fillna(train["category_2_sum_y"].median(),inplace=True)

train["category_3_min_y"].fillna(train["category_3_min_y"].median(),inplace=True)
train["category_3_mean_y"].fillna(train["category_3_mean_y"].median(),inplace=True)
train["category_3_max_y"].fillna(train["category_3_max_y"].median(),inplace=True)
train["category_3_sum_y"].fillna(train["category_3_sum_y"].median(),inplace=True)

train["category_3_min_x"].fillna(train["category_3_min_x"].median(),inplace=True)
train["category_3_sum_x"].fillna(train["category_3_sum_x"].median(),inplace=True)
train["category_3_mean_x"].fillna(train["category_3_mean_x"].median(),inplace=True)
train["category_3_max_x"].fillna(train["category_3_max_x"].median(),inplace=True)

train["new_first_buy"].fillna(train["new_first_buy"].median(),inplace=True)
train["new_last_buy"].fillna(train["new_last_buy"].median(),inplace=True)

train["hist_price_mean"].fillna(train["hist_price_mean"].median(),inplace=True)
train["hist_price_sum"].fillna(train["hist_price_sum"].median(),inplace=True)
train["hist_price_var"].fillna(train["hist_price_var"].median(),inplace=True)
train["hist_duration_skew"].fillna(train["hist_duration_skew"].median(),inplace=True)
train["hist_amount_month_ratio_skew"].fillna(train["hist_amount_month_ratio_skew"].median(),inplace=True)

train["price_mean"].fillna(train["price_mean"].median(),inplace=True)
train["price_max"].fillna(train["price_max"].median(),inplace=True)
train["price_total"].fillna(train["price_total"].median(),inplace=True)

train["installments_total"].fillna(train["installments_total"].median(),inplace=True)
train["installments_mean"].fillna(train["installments_mean"].median(),inplace=True)
train["installments_max"].fillna(train["installments_max"].median(),inplace=True)
train["installments_ratio"].fillna(train["installments_ratio"].median(),inplace=True)

train["new_purchase_date_diff"].fillna(train["new_purchase_date_diff"].median(),inplace=True)
train["new_purchase_date_average"].fillna(train["new_purchase_date_average"].median(),inplace=True)
train["new_purchase_date_uptonow"].fillna(train["new_purchase_date_uptonow"].median(),inplace=True)
train["new_purchase_date_uptomin"].fillna(train["new_purchase_date_uptomin"].median(),inplace=True)

train["month_diff_mean"].fillna(train["month_diff_mean"].median(),inplace=True)
train["month_diff_ratio"].fillna(train["month_diff_ratio"].median(),inplace=True)

train["month_lag_mean"].fillna(train["month_lag_mean"].median(),inplace=True)
#train["month_lag_min"].fillna(train["month_lag_min"].median(),inplace=True)
#train["month_lag_max"].fillna(train["month_lag_max"].median(),inplace=True)

train["card_id_total"].fillna(train["card_id_total"].median(),inplace=True)
train["card_id_cnt_total"].fillna(train["card_id_cnt_total"].median(),inplace=True)
train["card_id_cnt_ratio"].fillna(train["card_id_cnt_ratio"].median(),inplace=True)

train["purchase_amount_total"].fillna(train["purchase_amount_total"].median(),inplace=True)
train["purchase_amount_mean"].fillna(train["purchase_amount_mean"].median(),inplace=True)
train["purchase_amount_max"].fillna(train["purchase_amount_max"].median(),inplace=True)
train["purchase_amount_min"].fillna(train["purchase_amount_min"].median(),inplace=True)
train["purchase_amount_ratio"].fillna(train["purchase_amount_ratio"].median(),inplace=True)

train["duration_mean"].fillna(train["duration_mean"].median(),inplace=True)
train["duration_min"].fillna(train["duration_min"].median(),inplace=True)
train["duration_max"].fillna(train["duration_max"].median(),inplace=True)

train["CLV_ratio"].fillna(train["CLV_ratio"].median(),inplace=True)
train["new_CLV"].fillna(train["new_CLV"].median(),inplace=True)

train["amount_month_ratio_mean"].fillna(train["amount_month_ratio_mean"].median(),inplace=True)
train["amount_month_ratio_min"].fillna(train["amount_month_ratio_min"].median(),inplace=True)
train["amount_month_ratio_max"].fillna(train["amount_month_ratio_max"].median(),inplace=True)


# In[14]:



test["new_transactions_count"].fillna(test["new_transactions_count"].median(),inplace=True)
test["new_purchase_amount_sum"].fillna(test["new_purchase_amount_sum"].median(),inplace=True)
test["new_purchase_amount_max"].fillna(test["new_purchase_amount_max"].median(),inplace=True)
test["new_purchase_amount_min"].fillna(test["new_purchase_amount_min"].median(),inplace=True)
test["new_purchase_amount_mean"].fillna(test["new_purchase_amount_mean"].median(),inplace=True)
test["new_installments_sum"].fillna(test["new_installments_sum"].median(),inplace=True)
test["new_installments_max"].fillna(test["new_installments_max"].median(),inplace=True)
test["new_installments_mean"].fillna(test["new_installments_mean"].median(),inplace=True)
test["new_purchase_date_max"].fillna(test["new_purchase_date_max"].mean(),inplace=True)
test["new_purchase_date_min"].fillna(test["new_purchase_date_min"].mean(),inplace=True)
#test["new_month_lag_max"].fillna(test["new_month_lag_max"].median(),inplace=True)
#test["new_month_lag_min"].fillna(test["new_month_lag_min"].median(),inplace=True)
test["new_month_lag_mean"].fillna(test["new_month_lag_mean"].median(),inplace=True)
#test["new_month_diff_max"].fillna(test["new_month_diff_max"].median(),inplace=True)
#test["new_month_diff_min"].fillna(test["new_month_diff_min"].median(),inplace=True)
test["new_month_diff_mean"].fillna(test["new_month_diff_mean"].median(),inplace=True)
test["new_weekend_sum"].fillna(test["new_weekend_sum"].median(),inplace=True)
#test["new_weekend_mean"].fillna(test["new_weekend_mean"].median(),inplace=True)
test["new_weekday_sum"].fillna(test["new_weekday_sum"].median(),inplace=True)
test["new_weekday_mean"].fillna(test["new_weekday_mean"].median(),inplace=True)
test["new_authorized_flag_sum"].fillna(test["new_authorized_flag_sum"].median(),inplace=True)
#test["new_authorized_flag_mean"].fillna(test["new_authorized_flag_mean"].median(),inplace=True)
test["new_category_1_sum"].fillna(test["new_category_1_sum"].median(),inplace=True)
test["new_category_1_mean"].fillna(test["new_category_1_mean"].median(),inplace=True)
#test["new_category_1_max"].fillna(test["new_category_1_max"].median(),inplace=True)
#test["new_category_1_min"].fillna(test["new_category_1_min"].median(),inplace=True)

test["new_card_id_size"].fillna(test["new_card_id_size"].median(),inplace=True)
test["new_card_id_count"].fillna(test["new_card_id_count"].median(),inplace=True)
test["new_month_nunique"].fillna(test["new_month_nunique"].median(),inplace=True)
test["new_month_mean"].fillna(test["new_month_mean"].median(),inplace=True)
test["new_month_min"].fillna(test["new_month_min"].median(),inplace=True)
#test["new_month_max"].fillna(test["new_month_max"].median(),inplace=True)
#test["new_hour_nunique"].fillna(test["new_hour_nunique"].median(),inplace=True)
test["new_hour_mean"].fillna(test["new_hour_mean"].median(),inplace=True)
test["new_hour_min"].fillna(test["new_hour_min"].median(),inplace=True)
test["new_hour_max"].fillna(test["new_hour_max"].median(),inplace=True)

test["new_weekofyear_nunique"].fillna(test["new_weekofyear_nunique"].median(),inplace=True)
test["new_weekofyear_max"].fillna(test["new_weekofyear_max"].median(),inplace=True)
test["new_weekofyear_mean"].fillna(test["new_weekofyear_mean"].median(),inplace=True)
test["new_weekofyear_min"].fillna(test["new_weekofyear_min"].median(),inplace=True)

test["new_subsector_id_nunique"].fillna(test["new_subsector_id_nunique"].median(),inplace=True)
test["new_merchant_category_id_nunique"].fillna(test["new_merchant_category_id_nunique"].median(),inplace=True)

test["new_price_sum"].fillna(test["new_price_sum"].median(),inplace=True)
test["new_price_min"].fillna(test["new_price_min"].median(),inplace=True)
#test["new_price_max"].fillna(test["new_price_max"].median(),inplace=True)
test["new_price_mean"].fillna(test["new_price_mean"].median(),inplace=True)
test["new_price_var"].fillna(test["new_price_var"].median(),inplace=True)

test["new_duration_mean"].fillna(test["new_duration_mean"].median(),inplace=True)
test["new_duration_min"].fillna(test["new_duration_min"].median(),inplace=True)
test["new_duration_max"].fillna(test["new_duration_max"].median(),inplace=True)
test["new_duration_var"].fillna(test["new_duration_var"].median(),inplace=True)
test["new_duration_skew"].fillna(test["new_duration_skew"].median(),inplace=True)

test["new_amount_month_ratio_mean"].fillna(test["new_amount_month_ratio_mean"].median(),inplace=True)
test["new_amount_month_ratio_min"].fillna(test["new_amount_month_ratio_min"].median(),inplace=True)
test["new_amount_month_ratio_max"].fillna(test["new_amount_month_ratio_max"].median(),inplace=True)
test["new_amount_month_ratio_var"].fillna(test["new_amount_month_ratio_var"].median(),inplace=True)
test["new_amount_month_ratio_skew"].fillna(test["new_amount_month_ratio_skew"].median(),inplace=True)

test["category_1_mean"].fillna(test["category_1_mean"].median(),inplace=True)
#test["category_2_mean"].fillna(test["category_2_mean"].median(),inplace=True)
#test["category_3_mean"].fillna(test["category_3_mean"].median(),inplace=True)

test["category_2_min_y"].fillna(test["category_2_min_y"].median(),inplace=True)
test["category_2_mean_y"].fillna(test["category_2_mean_y"].median(),inplace=True)
test["category_2_max_y"].fillna(test["category_2_max_y"].median(),inplace=True)
test["category_2_sum_y"].fillna(test["category_2_sum_y"].median(),inplace=True)

test["category_2_min_x"].fillna(test["category_2_min_y"].median(),inplace=True)
test["category_2_mean_x"].fillna(test["category_2_mean_y"].median(),inplace=True)
test["category_2_max_x"].fillna(test["category_2_max_y"].median(),inplace=True)
test["category_2_sum_x"].fillna(test["category_2_sum_y"].median(),inplace=True)

test["category_3_min_y"].fillna(test["category_3_min_y"].median(),inplace=True)
test["category_3_mean_y"].fillna(test["category_3_mean_y"].median(),inplace=True)
test["category_3_max_y"].fillna(test["category_3_max_y"].median(),inplace=True)
test["category_3_sum_y"].fillna(test["category_3_sum_y"].median(),inplace=True)

test["category_3_min_x"].fillna(test["category_3_min_x"].median(),inplace=True)
test["category_3_sum_x"].fillna(test["category_3_sum_x"].median(),inplace=True)
test["category_3_mean_x"].fillna(test["category_3_mean_x"].median(),inplace=True)
test["category_3_max_x"].fillna(test["category_3_max_x"].median(),inplace=True)

test["new_first_buy"].fillna(test["new_first_buy"].median(),inplace=True)
test["new_last_buy"].fillna(test["new_last_buy"].median(),inplace=True)

test["hist_price_mean"].fillna(test["hist_price_mean"].median(),inplace=True)
test["hist_price_sum"].fillna(test["hist_price_sum"].median(),inplace=True)
test["hist_price_var"].fillna(test["hist_price_var"].median(),inplace=True)
test["hist_duration_skew"].fillna(test["hist_duration_skew"].median(),inplace=True)
test["hist_amount_month_ratio_skew"].fillna(test["hist_amount_month_ratio_skew"].median(),inplace=True)

test["price_mean"].fillna(test["price_mean"].median(),inplace=True)
test["price_max"].fillna(test["price_max"].median(),inplace=True)
test["price_total"].fillna(test["price_total"].median(),inplace=True)

test["installments_total"].fillna(test["installments_total"].median(),inplace=True)
test["installments_mean"].fillna(test["installments_mean"].median(),inplace=True)
test["installments_max"].fillna(test["installments_max"].median(),inplace=True)
test["installments_ratio"].fillna(test["installments_ratio"].median(),inplace=True)

test["new_purchase_date_diff"].fillna(test["new_purchase_date_diff"].median(),inplace=True)
test["new_purchase_date_average"].fillna(test["new_purchase_date_average"].median(),inplace=True)
test["new_purchase_date_uptonow"].fillna(test["new_purchase_date_uptonow"].median(),inplace=True)
test["new_purchase_date_uptomin"].fillna(test["new_purchase_date_uptomin"].median(),inplace=True)

test["month_diff_mean"].fillna(test["month_diff_mean"].median(),inplace=True)
test["month_diff_ratio"].fillna(test["month_diff_ratio"].median(),inplace=True)

test["month_lag_mean"].fillna(test["month_lag_mean"].median(),inplace=True)
#test["month_lag_min"].fillna(test["month_lag_min"].median(),inplace=True)
#test["month_lag_max"].fillna(test["month_lag_max"].median(),inplace=True)

#test["card_id_total"].fillna(test["card_id_total"].median(),inplace=True)
#test["card_id_cnt_total"].fillna(test["card_id_cnt_total"].median(),inplace=True)
#test["card_id_cnt_ratio"].fillna(test["card_id_cnt_ratio"].median(),inplace=True)

test["purchase_amount_total"].fillna(test["purchase_amount_total"].median(),inplace=True)
test["purchase_amount_mean"].fillna(test["purchase_amount_mean"].median(),inplace=True)
test["purchase_amount_max"].fillna(test["purchase_amount_max"].median(),inplace=True)
test["purchase_amount_min"].fillna(test["purchase_amount_min"].median(),inplace=True)
test["purchase_amount_ratio"].fillna(test["purchase_amount_ratio"].median(),inplace=True)

test["duration_mean"].fillna(test["duration_mean"].median(),inplace=True)
test["duration_min"].fillna(test["duration_min"].median(),inplace=True)
test["duration_max"].fillna(test["duration_max"].median(),inplace=True)

test["CLV_ratio"].fillna(test["CLV_ratio"].median(),inplace=True)
test["new_CLV"].fillna(test["new_CLV"].median(),inplace=True)

test["amount_month_ratio_mean"].fillna(test["amount_month_ratio_mean"].median(),inplace=True)
test["amount_month_ratio_min"].fillna(test["amount_month_ratio_min"].median(),inplace=True)
test["amount_month_ratio_max"].fillna(test["amount_month_ratio_max"].median(),inplace=True)
train.shape, test.shape


# In[15]:


train['card_id_total'] = train['new_card_id_size']+train['hist_card_id_size']
train['card_id_cnt_total'] = train['new_card_id_count']+train['hist_card_id_count']
train['card_id_cnt_ratio'] = train['new_card_id_count']/train['hist_card_id_count']
train['purchase_amount_total'] = train['new_purchase_amount_sum']+train['hist_purchase_amount_sum']
train['purchase_amount_mean'] = train['new_purchase_amount_mean']+train['hist_purchase_amount_mean']
train['purchase_amount_max'] = train['new_purchase_amount_max']+train['hist_purchase_amount_max']
train['purchase_amount_min'] = train['new_purchase_amount_min']+train['hist_purchase_amount_min']
train['purchase_amount_ratio'] = train['new_purchase_amount_sum']/train['hist_purchase_amount_sum']
train['month_diff_mean'] = train['new_month_diff_mean']+train['hist_month_diff_mean']
train['month_diff_ratio'] = train['new_month_diff_mean']/train['hist_month_diff_mean']
train['month_lag_mean'] = train['new_month_lag_mean']+train['hist_month_lag_mean']
#train['month_lag_max'] = train['new_month_lag_max']+train['hist_month_lag_max']
#train['month_lag_min'] = train['new_month_lag_min']+train['hist_month_lag_min']
train['category_1_mean'] = train['new_category_1_mean']+train['hist_category_1_mean']


train['installments_total'] = train['new_installments_sum']+train['hist_installments_sum']
train['installments_mean'] = train['new_installments_mean']+train['hist_installments_mean']
train['installments_max'] = train['new_installments_max']+train['hist_installments_max']
train['installments_ratio'] = train['new_installments_sum']/train['hist_installments_sum']
train['price_total'] = train['purchase_amount_total'] / train['installments_total']
train['price_mean'] = train['purchase_amount_mean'] / train['installments_mean']
train['price_max'] = train['purchase_amount_max'] / train['installments_max']
train['duration_mean'] = train['new_duration_mean']+train['hist_duration_mean']
train['duration_min'] = train['new_duration_min']+train['hist_duration_min']
train['duration_max'] = train['new_duration_max']+train['hist_duration_max']
train['amount_month_ratio_mean']=train['new_amount_month_ratio_mean']+train['hist_amount_month_ratio_mean']
train['amount_month_ratio_min']=train['new_amount_month_ratio_min']+train['hist_amount_month_ratio_min']
train['amount_month_ratio_max']=train['new_amount_month_ratio_max']+train['hist_amount_month_ratio_max']
train['new_CLV'] = train['new_card_id_count'] * train['new_purchase_amount_sum'] / train['new_month_diff_mean']
train['hist_CLV'] = train['hist_card_id_count'] * train['hist_purchase_amount_sum'] / train['hist_month_diff_mean']
train['CLV_ratio'] = train['new_CLV'] / train['hist_CLV']

test['card_id_total'] = test['new_card_id_size']+test['hist_card_id_size']
test['card_id_cnt_total'] = test['new_card_id_count']+test['hist_card_id_count']
test['card_id_cnt_ratio'] = test['new_card_id_count']/test['hist_card_id_count']
test['purchase_amount_total'] = test['new_purchase_amount_sum']+test['hist_purchase_amount_sum']
test['purchase_amount_mean'] = test['new_purchase_amount_mean']+test['hist_purchase_amount_mean']
test['purchase_amount_max'] = test['new_purchase_amount_max']+test['hist_purchase_amount_max']
test['purchase_amount_min'] = test['new_purchase_amount_min']+test['hist_purchase_amount_min']
test['purchase_amount_ratio'] = test['new_purchase_amount_sum']/test['hist_purchase_amount_sum']
test['month_diff_mean'] = test['new_month_diff_mean']+test['hist_month_diff_mean']
test['month_diff_ratio'] = test['new_month_diff_mean']/test['hist_month_diff_mean']
test['month_lag_mean'] = test['new_month_lag_mean']+test['hist_month_lag_mean']
#test['month_lag_max'] = test['new_month_lag_max']+test['hist_month_lag_max']
#test['month_lag_min'] = test['new_month_lag_min']+test['hist_month_lag_min']
test['category_1_mean'] = test['new_category_1_mean']+test['hist_category_1_mean']

test['installments_total'] = test['new_installments_sum']+test['hist_installments_sum']
test['installments_mean'] = test['new_installments_mean']+test['hist_installments_mean']
test['installments_max'] = test['new_installments_max']+test['hist_installments_max']
test['installments_ratio'] = test['new_installments_sum']/test['hist_installments_sum']
test['price_total'] = test['purchase_amount_total'] / test['installments_total']
test['price_mean'] = test['purchase_amount_mean'] / test['installments_mean']
test['price_max'] = test['purchase_amount_max'] / test['installments_max']
test['duration_mean'] = test['new_duration_mean']+test['hist_duration_mean']
test['duration_min'] = test['new_duration_min']+test['hist_duration_min']
test['duration_max'] = test['new_duration_max']+test['hist_duration_max']
test['amount_month_ratio_mean']=test['new_amount_month_ratio_mean']+test['hist_amount_month_ratio_mean']
test['amount_month_ratio_min']=test['new_amount_month_ratio_min']+test['hist_amount_month_ratio_min']
test['amount_month_ratio_max']=test['new_amount_month_ratio_max']+test['hist_amount_month_ratio_max']
test['new_CLV'] = test['new_card_id_count'] * test['new_purchase_amount_sum'] / test['new_month_diff_mean']
test['hist_CLV'] = test['hist_card_id_count'] * test['hist_purchase_amount_sum'] / test['hist_month_diff_mean']
test['CLV_ratio'] = test['new_CLV'] / test['hist_CLV']
train.shape, test.shape


# In[16]:


train.replace({np.inf: 0, -np.inf: 0}, inplace=True)
test.replace({np.inf: 0, -np.inf: 0}, inplace=True)

train = train.drop(['new_purchase_date_max', 'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_price_min'], axis =1 )
test = test.drop(['new_purchase_date_max','hist_purchase_date_max', 'hist_purchase_date_min', 'hist_price_min'], axis =1 )

print("Train/Test Shape: ",train.shape,test.shape)

y = train['target']
print("y shape: ",y.shape)
print("Before drop - X shape: ",train.shape)
X = train.drop(['target'], axis=1)
print("After drop - X shape: ",X.shape)
print("Nulls in X: ",(X.isnull().sum() > 0 ).sum())
print("Nulls in y: ",(y.isnull().sum() > 0 ).sum())

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print("X_train, X_test, y_train, y_test shape:",X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[17]:


X["installments_ratio"].fillna(X["installments_ratio"].median(),inplace=True)
df= X.isnull().sum() > 0
print(df[130:160])


# In[18]:


del history;
gc.collect() 
del new;
gc.collect() 
del final;
gc.collect() 
del ht;
gc.collect() 
del nmt;
gc.collect() 
train.head(5)


# In[19]:


train_cols= ['card_id_total', 'card_id_cnt_total', 'card_id_cnt_ratio', 'purchase_amount_total' , 'purchase_amount_mean' ,'purchase_amount_max',
'purchase_amount_min', 'purchase_amount_ratio', 'month_diff_mean' , 'month_diff_ratio', 'month_lag_mean', 'category_1_mean', 'installments_total',
'installments_mean', 'installments_max', 'installments_ratio', 'price_total', 'price_mean','price_max', 'duration_mean', 'duration_min' , 'duration_max',
'amount_month_ratio_mean', 'amount_month_ratio_min','amount_month_ratio_max', 'new_CLV', 'hist_CLV', 'CLV_ratio','AvOrderValue','AgeRecencyRatio']


# In[20]:



df = test_min.isnull().sum() > 0
df[:20]


# In[21]:


from sklearn.linear_model import LinearRegression

X_min = X[train_cols]
y_min = y
test_min = test[train_cols]
X_min.replace({np.inf: 0, -np.inf: 0}, inplace=True)
test_min.replace({np.inf: 0, -np.inf: 0}, inplace=True)
test_min["installments_ratio"].fillna(test_min["installments_ratio"].median(),inplace=True)

# Fitting Linear Regression to the dataset
regressor = LinearRegression() 
regressor.fit(X_min, y_min) #training the algorithm

y_pred_linear = regressor.predict(test_min)
print(y_pred_linear.shape)

submission = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv') 
print(submission.head())
submission['target'] = y_pred_linear 
print(submission.head())
submission.to_csv('linear.csv', index=False)


# In[22]:


# Using Ridge Regression 
ridge=Ridge() 

parameters={'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]} 
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_root_mean_squared_error',cv=5) 
ridge_regressor.fit(X_min, y_min)
y_pred_ridge = regressor.predict(test_min)
print(y_pred_ridge.shape)

submission = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv') 
print(submission.head())
submission['target'] = y_pred_ridge 
print(submission.head())
submission.to_csv('ridge.csv', index=False)


# In[23]:


from sklearn.linear_model import Lasso 

lasso=Lasso() 
parameters={'alpha':[1e-3]} 
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_root_mean_squared_error',cv=5)
lasso_regressor.fit(X_min,y_min)
prediction_lasso = lasso_regressor.predict(test_min) 

submission = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv') 
print(submission.head())
submission['target'] = prediction_lasso 
print(submission.head())
submission.to_csv('lasso.csv', index=False)


# In[24]:


treeRegressor = DecisionTreeRegressor() 

param_grid = {"criterion": ["mse"], "max_depth": [5], "min_samples_split": [8], "max_leaf_nodes": [15], "max_features": [25], "min_impurity_decrease":[0.1] }
grid_decision = GridSearchCV(treeRegressor, param_grid, cv=3,verbose=1,n_jobs=-1) 
grid_decision.fit(X_min, y_min)
y_pred_decision = grid_decision.predict(test_min)

submission = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv') 
print(submission.head())
submission['target'] = y_pred_decision 
print(submission.head())
submission.to_csv('decision.csv', index=False)


# In[25]:


param_grid = {"criterion": ["mse"], 'n_estimators': [1000], "max_depth": [5], "max_leaf_nodes" : [5], "min_samples_split":[8] , "max_features": [25], "min_impurity_decrease":[0.1] } 
forestRegressor = RandomForestRegressor(random_state = 10) 

grid_forest = GridSearchCV(forestRegressor, param_grid, cv=3, verbose=1,n_jobs=-1) 
grid_forest.fit(X_min, y_min)
y_pred_forest = grid_forest.predict(test_min)

submission = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv') 
print(submission.head())
submission['target'] = y_pred_forest 
print(submission.head())
submission.to_csv('forest.csv', index=False)


# In[26]:


# Fitting Extra Trees regressor to the dataset 
extraRegressor = ExtraTreesRegressor(random_state = 10)
param_grid = {"criterion": ["mse"], 'n_estimators': [500], "max_depth": [5], "max_leaf_nodes" : [5], "min_samples_leaf":[2], "min_samples_split":[2] , "max_features": [25], "min_impurity_decrease":[0.1] }

grid_extra = GridSearchCV(extraRegressor, param_grid, cv=3, verbose=1,n_jobs=-1)
grid_extra.fit(X_min, y_min) 
y_pred_extra = grid_extra.predict(test_min)

submission = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv') 
print(submission.head())
submission['target'] = y_pred_extra 
print(submission.head())
submission.to_csv('extra.csv', index=False)


# In[27]:


from sklearn.ensemble import AdaBoostRegressor
param = { 'n_estimators':[50], 'learning_rate':[1e-2], 'loss':['exponential'] }
adaRegressor = AdaBoostRegressor(random_state = 10) 

grid_ada = GridSearchCV(adaRegressor, param, cv = 3, n_jobs = -1, verbose=1)
grid_ada.fit(X_min, y_min) 
y_pred_ada = grid_ada.predict(test_min)

submission = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv') 
print(submission.head())
submission['target'] = y_pred_ada 
print(submission.head())
submission.to_csv('ada.csv', index=False)


# In[28]:


SEED = 10

# Using Decision Tree as a base model
def get_model():   
  dt = DecisionTreeRegressor(criterion = 'mse', max_depth = 5, max_features = 25, max_leaf_nodes = 15, min_impurity_decrease = 0.1, min_samples_split = 8, random_state=SEED)    
  return dt


# In[29]:


from random import choices
def get_samples(D1_train, D2_train, n_estimators):
  print("\nCalculating Samples with replacement...")
  samples_train_appender = []
  samples_test_appender = []
  all_indexes = D1_train.index
  population_size = D1_train.shape[0]  
  sample_size = round(population_size/n_estimators)  
  bumpedup_sample_size = int(sample_size * 1.85)  

  for s in range(n_estimators):
    samples_index = choices(all_indexes, k = sample_size)        
    sample_train_df = D1_train[D1_train.index.isin(samples_index)]
    sample_test_df = D2_train[D2_train.index.isin(samples_index)]   
    samples_train_appender.append(sample_train_df)
    samples_test_appender.append(sample_test_df)

  all_train_samples = pd.concat(samples_train_appender,ignore_index=True)
  all_test_samples = pd.concat(samples_test_appender,ignore_index=True)
  print("Samples calculation Done.")    
  return all_train_samples, all_test_samples, sample_size

# Calculate RMSE
def evaluate_model(y_pred, y_actual):
  print("Evaluating Score...\n")
  mse = mean_squared_error(y_actual, y_pred)
  print('RMSE %.3f' % (np.sqrt(mse)))

def train_predict(n_estimators, all_train_samples, all_test_samples, D1_test,D2_test, sample_size):
  """Fit models in list on training set and return preds"""
  P = np.zeros((D2_test.shape[0], n_estimators))
  P = pd.DataFrame(P)
  models_list = []

  print("Sample size: ", sample_size)
  cols = list()
  print("Base models {} - fitting and predicting ...".format(n_estimators))
  
  for i in range(0,n_estimators):    
    j = sample_size * i
    k = sample_size * (i + 1)
    x_data = all_train_samples[j:k]
    y_data = all_test_samples[j:k]
    model = get_model() 
    
    model.fit(x_data, y_data)  
    models_list.append(model)           
    pred = (model.predict(D1_test))        
    P.iloc[:,i] = pred
    cols.append(i)
    
  P.columns = cols
  print("Base models done.")
  return P, models_list


# In[30]:


def custom_ensemble(X_train,y_train,n_estimators):
  print("Preparing Custom Ensemble...")
  #Split X_train into D1,D2 (50-50)
  D1_train, D1_test, D2_train, D2_test = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
  print("D1_train, D1_test, D2_train, D2_test: ",D1_train.shape, D1_test.shape, D2_train.shape, D2_test.shape)
  
  #Get Samples
  all_samples_train, all_samples_test, sample_size = get_samples(D1_train, D2_train, n_estimators)
  
  #Get predictions
  P, models_list = train_predict(n_estimators, all_samples_train, all_samples_test,D1_test, D2_test, sample_size)
  print("Custom Ensemble Done.")
  return P, models_list, D2_test

def super_train_predict(n_estimators, models_list, test_set):
  """Fit models in list on training set and return preds"""
  meta_pred = np.zeros((test_set.shape[0] , n_estimators))
  meta_pred = pd.DataFrame(meta_pred)
    
  print("Predicting {} models from metalearner...".format(n_estimators))
  cols = list()
  for i in range(0,n_estimators):    
    model = models_list[i]                    
    pred = (model.predict(test_set))          
    meta_pred.iloc[:,i] = pred
    cols.append(i)
    
  meta_pred.columns = cols       
    
  print("Meta Learner prediction Done.")
  return meta_pred


# In[31]:


X_train.replace({np.inf: 0, -np.inf: 0}, inplace=True)
X_test.replace({np.inf: 0, -np.inf: 0}, inplace=True)
X_train["installments_ratio"].fillna(X_train["installments_ratio"].median(),inplace=True)
X_test["installments_ratio"].fillna(X_test["installments_ratio"].median(),inplace=True)

X_t = X_train.copy()
print(X_t.shape)
X_t = X_t[train_cols]
print(X_t.shape)

X_tt = X_test.copy()
print(X_tt.shape)
X_tt = X_tt[train_cols]
print(X_tt.shape)


# In[32]:


test.replace({np.inf: 0, -np.inf: 0}, inplace=True)
test["installments_ratio"].fillna(test["installments_ratio"].median(),inplace=True)
te = test.copy()
te = te[train_cols]
te.shape


# In[33]:


df = te.isnull().sum()
df[0:35]


# In[34]:


import timeit
from sklearn.model_selection import StratifiedKFold

start = timeit.default_timer()

xgb = XGBRegressor()
parameters = {   'gamma': [8],  'eval_metric' :['rmse'],'eta': [0.5], 'colsample_bytree':[0.3],
               'min_child_weight': [3], 'max_depth' :[3], 'max_features':[5],'subsample': [0.7],'tree_method':['auto'],
               'reg_alpha':[1000], "criterion": ["mse"],'n_estimators': [1000] ,'seed':[11] }


meta_learner = GridSearchCV(xgb, parameters, cv = 6, n_jobs = -1, verbose=1)


n_estimators = 100
print("Fitting models to meta-learner.")
P, models_list, D2_test = custom_ensemble(X_t,y_train,n_estimators)
meta_learner.fit(P, D2_test)

# Ensemble final prediction and evaluation
meta_pred = super_train_predict(n_estimators, models_list, X_tt) # X_test brought from first split
pred_final = meta_learner.predict(meta_pred)

rmse = evaluate_model(pred_final,y_test)

stop = timeit.default_timer()
execution_time = (stop - start)/60

print("Ensemble Executed in {} minutes".format(str(execution_time)))


# In[35]:


train.tail()


# In[36]:


print("Best Score = ",meta_learner.best_score_)
print("Best Params = ",meta_learner.best_params_)


# In[37]:


start = timeit.default_timer()

test_meta_pred = super_train_predict(n_estimators, models_list, te) # test.csv 
test_pred_final = meta_learner.predict(test_meta_pred)

stop = timeit.default_timer()
execution_time = (stop - start)/60

print("Test Ensemble Executed in {} minutes".format(str(execution_time)))


# In[38]:


test_pred_final.shape


# In[39]:


submission = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv') 
print(submission.head())
submission['target'] = test_pred_final 
print(submission.head())
submission.to_csv('sub.csv', index=False)


# In[40]:


print(train.shape, test.shape)

