#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import gc
import os

import re

from sklearn.model_selection import KFold, RepeatedKFold

import time
import lightgbm as lgb

from sklearn.metrics import mean_squared_error,roc_auc_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor, Pool, CatBoost
from sklearn.linear_model import LinearRegression,BayesianRidge
from boruta import BorutaPy
#from keras.utils import to_categorical

import matplotlib.pyplot as plt 
import seaborn as sns 
from pylab import rcParams

import datetime

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')


# In[2]:


def rmse(y_true,y_pred):
    return np.sqrt(np.power(y_true-y_pred,2).sum()/len(y_true))

def replace_id(df,dictionary = False):
    
    df['card_id2']  = df.card_id.str.replace('C','')
    df['card_id2']  = df.card_id2.str.replace('_','')
    df['card_id2']  = df.card_id2.str.replace('I','')
    df['card_id2']  = df.card_id2.str.replace('D','')
    df['card_id2']  = df.card_id2.str.replace('a','10')
    df['card_id2']  = df.card_id2.str.replace('b','11')
    df['card_id2']  = df.card_id2.str.replace('c','12')
    df['card_id2']  = df.card_id2.str.replace('d','13')
    df['card_id2']  = df.card_id2.str.replace('e','14')
    df['card_id2']  = df.card_id2.str.replace('f','15').astype(np.float64)   
    
    if dictionary:
        d = df[['card_id','card_id2']]
        df.drop('card_id',axis=1,inplace=True)
        df = df.rename(columns = {'card_id2':'card_id'})
        return df,d
    df.drop('card_id',axis=1,inplace=True)
    df = df.rename(columns = {'card_id2':'card_id'})
    return df
    

def replace_month(df):
    df['month_id']  = df.first_active_month.str.replace("-",'').fillna(201802).astype(int)
    
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 8, 1) - df['first_active_month'].dt.date).dt.days
    return df

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

def dist_holiday(df, col_name, date_holiday, date_ref, period=100):
    df[col_name] = np.maximum(np.minimum((pd.to_datetime(date_holiday) - df[date_ref]).dt.days, period), 0)
    
def submit(df,name,score):
    arr = []
    path = '../'
    for a in os.listdir(path):
        arr.append(a)
    arr = [re.sub("[^0-9_]",'',a) for a in arr ]
    arr = [a.split('_') for a in arr if a not in ['']]
    arr = [int(a[0]) for a in arr if a[0] not in ['']]
    pre = str(max(arr))
    str_dat = str(time.localtime().tm_year)+'_'+str(time.localtime().tm_mon)+'_'+str(time.localtime().tm_mday)
    str_dat
    df[['card_id','target']].to_csv(path+pre+'_'+str_dat+'_CV_'+str(score)+'.csv')


# In[3]:



def deanon_purchase(purchase):
    return np.round(purchase/0.00150265118 + 497.06,8)

def deanon_target(target):
    return np.exp2(target)

def anon_target(target):
    return np.log2(target)


# In[4]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv(\'../inpt/train.csv\')\ntest_final = pd.read_csv(\'../input/test.csv\')\nhist = pd.read_csv(\'../input/historical_transactions.csv\',parse_dates=["purchase_date"])\nmerch = pd.read_csv(\'../input/merchants.csv\')\nnew_t = pd.read_csv(\'../input/new_merchant_transactions.csv\',parse_dates=["purchase_date"])')


# In[5]:


train = reduce_mem_usage(train)
test_final = reduce_mem_usage(test_final)
hist = reduce_mem_usage(hist)
merch = reduce_mem_usage(merch)
new_t = reduce_mem_usage(new_t)


# In[6]:


plt.hist(train['target']);


# In[7]:


print('Sample with target less then -33: ',train[train.target < -33.].card_id.nunique(),'rows')


# In[8]:


card_with_large_negative = train[train.target < -33.].card_id.unique()
print('Cards with large negative:',train[train.target < -33.].card_id.nunique())


# In[9]:


get_ipython().run_cell_magic('time', '', 'train = replace_month(train)\ntest_final = replace_month(test_final)\ngc.collect()')


# In[10]:


merch.category_1 = merch.category_1.replace({'Y':'1','N':'0'}).astype(int)
merch.category_2.fillna(1,inplace=True)
merch.category_4 = merch.category_4.replace({'Y':'1','N':'0'}).astype(int)
merch.most_recent_sales_range = merch.most_recent_sales_range.replace({'A':'1','B':'2','C':'3','D':'4','E':'5'}).astype(int)
merch.most_recent_purchases_range = merch.most_recent_purchases_range.replace({'A':'1','B':'2','C':'3','D':'4','E':'5'}).astype(int)


# In[11]:


merch['numerical_1'] = np.round(merch['numerical_1'] / 0.009914905 + 5.79639, 0)
merch['numerical_2'] = np.round(merch['numerical_2'] / 0.009914905 + 5.79639, 0)
merch['avg_purchases_lag3' ] = deanon_purchase(merch['avg_purchases_lag3' ])
merch['avg_purchases_lag6' ] = deanon_purchase(merch['avg_purchases_lag6' ])
merch['avg_purchases_lag12'] = deanon_purchase(merch['avg_purchases_lag12'])


# In[12]:


merch.describe().T


# In[13]:


merch.loc[(merch.avg_purchases_lag3  == np.inf),'avg_purchases_lag3'] = merch[(merch.avg_purchases_lag3  != np.inf)]['avg_purchases_lag3'].mean()
merch.loc[(merch.avg_purchases_lag6  == np.inf),'avg_purchases_lag6'] = merch[(merch.avg_purchases_lag6  != np.inf)]['avg_purchases_lag6'].mean()
merch.loc[(merch.avg_purchases_lag12 == np.inf),'avg_purchases_lag12'] = merch[(merch.avg_purchases_lag12 != np.inf)]['avg_purchases_lag12'].mean()


# In[14]:


print(len(merch['merchant_id'].unique()))
print(merch['merchant_id'].count())


# In[15]:


merch = merch.groupby('merchant_id').mean().reset_index()
merch.columns = ['merch_'+a for a in merch.columns]
merch = merch.rename(columns={'merch_merchant_id':'merchant_id'})


# In[16]:


merch = merch[['merchant_id','merch_avg_sales_lag3',
       'merch_avg_purchases_lag3','merch_numerical_1','merch_numerical_2']]


# In[17]:


hist  = pd.merge(hist ,merch,on='merchant_id',how='left')
new_t = pd.merge(new_t,merch,on='merchant_id',how='left')


# In[18]:


del merch;
gc.collect()


# In[19]:


auth = hist[hist['authorized_flag']== 'Y']
non_auth = hist[hist['authorized_flag']== 'N']
card_ln = hist[hist.card_id.isin(card_with_large_negative)]


# In[20]:


get_ipython().run_cell_magic('time', '', "def format_transaction(added_1,pre=''):\n    print('Starting formating for',pre)\n    added_1['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True) #one missing merchant\n    orig = added_1.copy()\n    #replace object data\n    print('replace object data',end=' - ')\n    added_1.authorized_flag = added_1.authorized_flag.replace({'Y':'1','N':'0'}).astype(int)\n    added_1.category_1 = added_1.category_1.replace({'Y':'1','N':'0'}).astype(int)\n    added_1.category_3 = added_1.category_3.replace({'A':'1','B':'2','C':'3',np.nan:'1'}).astype(int)\n    print('ready')    \n    #circled days, weeks and month\n    print('Circle dates',end=' - ')\n    added_1['cos_doy'  ] = np.cos(added_1.purchase_date.dt.dayofyear*2*np.pi/365)\n    added_1['sin_doy'  ] = np.cos(added_1.purchase_date.dt.dayofyear*2*np.pi/365)    \n    added_1['cos_dow'  ] = np.cos(added_1.purchase_date.dt.dayofweek*2*np.pi/7)\n    added_1['sin_dow'  ] = np.cos(added_1.purchase_date.dt.dayofweek*2*np.pi/7)    \n    added_1['cos_month'] = np.cos(added_1.purchase_date.dt.month*2*np.pi/12)\n    added_1['sin_month'] = np.cos(added_1.purchase_date.dt.month*2*np.pi/12)    \n    added_1['cos_week' ] = np.cos(added_1.purchase_date.dt.week*2*np.pi/52)\n    added_1['sin_week' ] = np.cos(added_1.purchase_date.dt.week*2*np.pi/52)\n    print('ready')\n    #added year, month, and yearmonth\n    print('added year, month, and yearmonth',end=' - ')\n    added_1['year'] = added_1.purchase_date.dt.year.astype(str)\n    added_1['weekofyear'] = added_1.purchase_date.dt.weekofyear\n    added_1['dayofweek'] = added_1.purchase_date.dt.dayofweek\n    added_1['month'] = added_1['purchase_date'].dt.month\n    added_1['weekend'] = (added_1.purchase_date.dt.weekday >=5).astype(int)\n    added_1['not_weekend'] = (added_1.purchase_date.dt.weekday <5).astype(int)\n    added_1['hour'] = added_1.purchase_date.dt.hour  \n    added_1.loc[added_1['month']<10,'month'] = '0'+added_1.loc[added_1['month']<10,'month'].astype(str)\n    added_1['month_id'] = added_1['year'].astype(str)+added_1['month'].astype(str)\n    \n    added_1['month_diff']    = ((datetime.datetime.today() - added_1['purchase_date']).dt.days)//30\n    added_1['month_diff']   += added_1['month_lag']  \n    added_1['day']           = added_1['purchase_date'].dt.day\n    added_1['dayofyear']     = added_1['purchase_date'].dt.dayofyear\n    added_1['quarter']       = added_1['purchase_date'].dt.quarter\n    \n    added_1['price']              = added_1['purchase_amount'] / added_1['installments']\n    added_1['duration']           = added_1['purchase_amount'] * added_1['month_diff']\n    added_1['amount_month_ratio'] = added_1['purchase_amount'] / added_1['month_diff']\n    \n    print('ready')\n    #add holidays    \n    holidays = [\n        ('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017\n        ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017\n        ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017\n        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017\n        ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017\n        ('Black_Friday_2017', '2017-11-24'),  # Black Friday: 24th November 2017\n        ('Mothers_Day_2018', '2018-05-13'),\n    ]\n    \n    for d_name, d_day in holidays:\n        dist_holiday(added_1, d_name, d_day, 'purchase_date')\n    \n    #deanon purchase\n    print('deanon purchase',end=' - ')\n    added_1['purchase_amount'] = deanon_purchase(added_1['purchase_amount'])\n    print('ready')\n    #add category pairs\n    print('add category pairs',end=' - ')\n    added_1['category_12'] = added_1['category_1']*10 + added_1['category_2']\n    added_1['category_23'] = added_1['category_2']*10 + added_1['category_3']\n    added_1['category_13'] = added_1['category_1']*10 + added_1['category_3'] \n    \n    added_1['category_all'] = added_1['category_1']*100 + added_1['category_2']*10 + added_1['category_3'] \n    \n    added_1['month_diff'] = ((datetime.datetime.today() - added_1['purchase_date']).dt.days)//30\n    added_1['month_diff'] += added_1['month_lag']\n    \n    added_1['month_diff_days'] = ((datetime.datetime.today() - added_1['purchase_date']).dt.days)//1\n    added_1['month_diff_days'] += added_1['month_lag']*30.5\n    \n    added_1 = pd.get_dummies(added_1, columns=['category_3','category_2'])\n    \n    added_1.loc[:, 'purchase_date'] = pd.DatetimeIndex(added_1['purchase_date']).\\\n                                      astype(np.int64) * 1e-9    \n    \n    \n    print('ready')    \n    \n    aggregate = {'installments':['sum','mean','median','std','var','nunique'],\n                 'merchant_category_id':['nunique'],\n                 'merchant_id':['nunique'],\n                 'city_id':['nunique'],\n                 'state_id':['nunique'],\n                 'subsector_id':['nunique'],\n                 \n                 'purchase_amount': ['sum', 'mean','median', 'max', 'min', 'std','var'],\n                 'authorized_flag':['sum'],\n                 \n                 'month_id':['nunique'],\n                 'purchase_date':[np.ptp,'min','max'],\n                 'month_lag':['max','min','mean','median','sum','std','var'],\n                 'weekofyear':['max','min','mean','median','sum','std','var'],\n                 'dayofweek':['max','min','mean','median','sum','std','var'],\n                 'weekend':['sum','mean','median'],\n                 'day':['sum','mean','median'],\n                 'dayofyear':['sum','mean','median'],\n                 'quarter':['sum','mean','median'],\n                 'not_weekend':['sum','mean','median'],\n                 'hour':['mean','median','sum'],\n                 'month_diff':['mean','median','var','min','max'],     \n                 'cos_doy'  :['mean','median','var','min','max'],    \n                 'sin_doy'  :['mean','median','var','min','max'],    \n                 'cos_dow'  :['mean','median','var','min','max'],    \n                 'sin_dow'  :['mean','median','var','min','max'],    \n                 'cos_month':['mean','median','var','min','max'],    \n                 'sin_month':['mean','median','var','min','max'],    \n                 'cos_week' :['mean','median','var','min','max'],    \n                 'sin_week' :['mean','median','var','min','max'],    \n                 \n                 'category_1':['sum'],                 \n                 'category_2_1.0': ['mean','median'],\n                 'category_2_2.0': ['mean','median'],\n                 'category_2_3.0': ['mean','median'],\n                 'category_2_4.0': ['mean','median'],\n                 'category_2_5.0': ['mean','median'],                 \n                 'category_3_1': ['mean','median'],\n                 'category_3_2': ['mean','median'],\n                 'category_3_3': ['mean','median'],                 \n                 'category_12':['mean','median'],\n                 'category_23':['mean','median'],\n                 'category_13':['mean','median'],                  \n                 'category_all':['mean','median'],\n                 'price': ['sum', 'mean','median', 'std','var'],           \n                 'duration': ['sum', 'mean','median', 'std','var'],    \n                 'amount_month_ratio': ['sum', 'mean','median', 'std','var'],\n                 \n                 'merch_avg_sales_lag3':['sum', 'mean','median', 'std','var'],\n                 'merch_avg_purchases_lag3':['sum', 'mean','median', 'std','var'],\n                 'merch_numerical_1':['sum', 'mean','median', 'std','var'],\n                 'merch_numerical_2':['sum', 'mean','median', 'std','var'],\n                 \n                 'Christmas_Day_2017': ['mean', 'sum'],\n                 'Mothers_Day_2017': ['mean', 'sum'],\n                 'fathers_day_2017': ['mean', 'sum'],\n                 'Children_day_2017': ['mean', 'sum'],\n                 'Valentine_Day_2017': ['mean', 'sum'],\n                 'Black_Friday_2017': ['mean', 'sum'],\n                 'Mothers_Day_2018': ['mean', 'sum']\n    }\n\n    agg_month = {\n                'purchase_amount': ['sum', 'mean','median', 'min', 'max', 'std','var'],\n                'installments': ['sum','mean','median','nunique'],                \n                'merch_avg_sales_lag3':['sum', 'mean','median', 'std'],\n                'merch_avg_purchases_lag3':['sum', 'mean','median', 'std'],\n                'merch_numerical_1':['sum', 'mean','median', 'std','var'],\n                'merch_numerical_2':['sum', 'mean','median', 'std','var'],\n                'weekofyear':['max','min','mean','median','sum','std','var'],\n                'dayofweek':['max','min','mean','median','sum','std','var'],\n                'weekend':['sum','mean','median'],\n                'not_weekend':['sum','mean','median'],\n                'hour':['mean','median','sum'],\n                'day':['mean','median','sum']\n                }\n    \n    agg_month2 = {\n                'purchase_amount': ['sum', 'mean','median', 'min', 'max', 'std','var'],\n                'installments': ['sum','mean','median','nunique'],                \n                'merch_avg_sales_lag3':['sum', 'mean','median', 'std'],\n                'merch_avg_purchases_lag3':['sum', 'mean','median', 'std'],   \n                'merch_numerical_1':['sum', 'mean','median', 'std','var'],\n                'merch_numerical_2':['sum', 'mean','median', 'std','var'],\n                'weekofyear':['max','min','mean','median','sum','std','var'],\n                'dayofweek':['max','min','mean','median','sum','std','var'],\n                'weekend':['sum','mean','median'],\n                'not_weekend':['sum','mean','median'],\n                'hour':['mean','median','sum'],\n                'month_diff':['mean','median'],\n                'day':['mean','median','sum']\n                }\n    \n    print('Start_aggr_month',end=' - ')\n    #----\n    #dont forget do it in cicle\n    agg_month_1 = added_1.groupby(['card_id','month_lag']).agg(agg_month)\n    agg_month_1.columns = ['ad1_'.join(col).strip() for col in agg_month_1.columns.values]\n    agg_month_1.reset_index(inplace=True)\n\n    agg_month_1 = agg_month_1.groupby(['card_id']).agg(['mean','median'])\n    agg_month_1.columns = ['ad1_'.join(col).strip() for col in agg_month_1.columns.values]\n    agg_month_1.reset_index(inplace=True)\n    #----\n    agg_month_2 = added_1.groupby(['card_id','month_diff']).agg(agg_month)\n    agg_month_2.columns = ['ad2_'.join(col).strip() for col in agg_month_2.columns.values]\n    agg_month_2.reset_index(inplace=True)\n\n    agg_month_2 = agg_month_2.groupby(['card_id']).agg(['mean','median'])\n    agg_month_2.columns = ['ad2_'.join(col).strip() for col in agg_month_2.columns.values]\n    agg_month_2.reset_index(inplace=True)\n    #----\n    agg_month_3 = added_1.groupby(['card_id','installments']).agg(agg_month2)\n    agg_month_3.columns = ['ad3_'.join(col).strip() for col in agg_month_3.columns.values]\n    agg_month_3.reset_index(inplace=True)\n\n    agg_month_3 = agg_month_3.groupby(['card_id']).agg(['mean','median'])\n    agg_month_3.columns = ['ad3_'.join(col).strip() for col in agg_month_3.columns.values]\n    agg_month_3.reset_index(inplace=True)\n    \n    #----\n    agg_month_4 = added_1.groupby(['card_id','state_id']).agg(agg_month)\n    agg_month_4.columns = ['ad4_'.join(col).strip() for col in agg_month_4.columns.values]\n    agg_month_4.reset_index(inplace=True)\n\n    agg_month_4 = agg_month_4.groupby(['card_id']).agg(['mean','median'])\n    agg_month_4.columns = ['ad4_'.join(col).strip() for col in agg_month_4.columns.values]\n    agg_month_4.reset_index(inplace=True)\n    \n    #----\n    agg_month_5 = added_1.groupby(['card_id','category_all']).agg(agg_month)\n    agg_month_5.columns = ['ad5_'.join(col).strip() for col in agg_month_5.columns.values]\n    agg_month_5.reset_index(inplace=True)\n\n    agg_month_5 = agg_month_5.groupby(['card_id']).agg(['mean','median'])\n    agg_month_5.columns = ['ad5_'.join(col).strip() for col in agg_month_5.columns.values]\n    agg_month_5.reset_index(inplace=True)\n    print ('ready')\n    \n    print (added_1.columns)\n    print('Start_aggr',end=' - ')\n    added_1 = added_1.groupby(['card_id']).agg(aggregate)\n    print ('ready')\n    added_1.columns = [pre+'_'.join(col).strip() for col in added_1.columns.values]\n    added_1.reset_index(inplace=True)\n    print('Start merge',end=' - ')\n    #added_1 = pd.merge(added_1,dummys_to_card,on='card_id',how='left')\n    print ('ready')\n\n    print('Start merge month',end=' - ')\n    added_1 = pd.merge(added_1,agg_month_1,on='card_id',how='left')\n    added_1 = pd.merge(added_1,agg_month_2,on='card_id',how='left')\n    added_1 = pd.merge(added_1,agg_month_3,on='card_id',how='left')\n    added_1 = pd.merge(added_1,agg_month_4,on='card_id',how='left')\n    added_1 = pd.merge(added_1,agg_month_5,on='card_id',how='left')\n    print ('ready')\n    \n    del agg_month_1,agg_month_2,agg_month_3,agg_month_4,agg_month_5;\n    gc.collect();\n    \n    agr_feat = ['category_1','installments','state_id']\n    \n    #orig['category_all'] = orig['category_1']*100 + orig['category_2']*10 + orig['category_3'] \n    \n    for a in agr_feat:\n        print('Start gen features from '+a,end=' - ')\n        agr = orig.groupby(['card_id',a])['purchase_amount'].mean()\n        agr = pd.DataFrame(agr).reset_index().groupby('card_id')['purchase_amount'].agg(['mean','median'])\n        agr.columns = [a+'_purchase_amount_'+col for col in agr.columns.values]\n        agr.reset_index(inplace=True)\n        added_1 = pd.merge(added_1,agr,on='card_id', how='left')\n        print('ready')\n        \n    agr_feat = ['category_1','state_id']\n    for a in agr_feat:\n        print('Start gen features from '+a,end=' - ')\n        agr = orig.groupby(['card_id',a])['installments'].mean()\n        agr = pd.DataFrame(agr).reset_index().groupby('card_id')['installments'].agg(['mean','median'])\n        agr.columns = [a+'_installments_'+col for col in agr.columns.values]\n        agr.reset_index(inplace=True)\n        added_1 = pd.merge(added_1,agr,on='card_id', how='left')\n        print('ready')\n    del orig;\n    return added_1")


# In[21]:


get_ipython().run_cell_magic('time', '', "added_2 = format_transaction(auth.copy(),'auth')\nadded_3 = format_transaction(non_auth.copy(),'non_auth')")


# In[22]:


get_ipython().run_cell_magic('time', '', "added_4 = format_transaction(new_t.copy(),'new_')")


# In[23]:


added_5 = card_ln.copy()
added_5.authorized_flag = added_5.authorized_flag.replace({'Y':'1','N':'0'}).astype(int)
added_5.category_1 = added_5.category_1.replace({'Y':'1','N':'0'}).astype(int)
added_5.category_3 = added_5.category_3.replace({'A':'1','B':'2','C':'3',np.nan:'4'}).astype(int)

added_5 = added_5.groupby(['card_id']).agg({'authorized_flag':'mean',
                                            'city_id':'nunique',
                                            'category_1':'mean',
                                            'installments':'mean',
                                            'category_3':'mean',
                                            'merchant_category_id':'nunique',
                                            'merchant_id':'nunique',
                                            'month_lag':'mean',
                                            'purchase_amount':'mean',
                                            'category_2':'mean',
                                            'state_id':'nunique',
                                            'subsector_id':'nunique'
    
}).reset_index()
added_5 = added_5.mean()
columns = added_5.reset_index().T.head(1).values[0]
added_5 = added_5.reset_index().T
added_5.columns = columns
# added_5 = added_5[added_5.card_id!='card_id']
added_5.columns = ['MEAN_LN_'+col for col in added_5.columns.values]


# In[24]:


LN_col = added_5.columns


# In[25]:


train = pd.merge(train,added_2,on=['card_id'],how='left')
test_final = pd.merge(test_final,added_2,on=['card_id'],how='left')

train = pd.merge(train,added_3,on=['card_id'],how='left')
test_final = pd.merge(test_final,added_3,on=['card_id'],how='left')

train = pd.merge(train,added_4,on=['card_id'],how='left')
test_final = pd.merge(test_final,added_4,on=['card_id'],how='left')


train.fillna(0,inplace=True)
test_final.fillna(0,inplace=True)


# In[26]:


train['ELO_auth_sum'] = 1/(1+pow(10,(train.authpurchase_amount_sum-train.non_authpurchase_amount_sum)/400))
train['ELO_purchase_sum'] = 1/(1+pow(10,(train.authpurchase_amount_sum+train.non_authpurchase_amount_sum-abs(train.non_authpurchase_amount_sum))/400))

train['ELO_auth_mean'] = 1/(1+pow(10,(train.authpurchase_amount_mean-train.non_authpurchase_amount_mean)/400))
train['ELO_purchase_mean'] = 1/(1+pow(10,(train.authpurchase_amount_mean+train.non_authpurchase_amount_mean-abs(train.non_authpurchase_amount_mean))/400))


# In[27]:


test_final['ELO_auth_sum'] = 1/(1+pow(10,(test_final.authpurchase_amount_sum-test_final.non_authpurchase_amount_sum)/400))
test_final['ELO_purchase_sum'] = 1/(1+pow(10,(test_final.authpurchase_amount_sum+test_final.non_authpurchase_amount_sum-abs(test_final.non_authpurchase_amount_sum))/400))

test_final['ELO_auth_mean'] = 1/(1+pow(10,(test_final.authpurchase_amount_mean-test_final.non_authpurchase_amount_mean)/400))
test_final['ELO_purchase_mean'] = 1/(1+pow(10,(test_final.authpurchase_amount_mean+test_final.non_authpurchase_amount_mean-abs(test_final.non_authpurchase_amount_mean))/400))


# In[28]:


train['ELO_auth_month_diff'] = 1/(1+pow(10,(train.authmonth_diff_mean-train.non_authmonth_diff_mean)/400))
train['ELO_auth_new_month_diff'] = 1/(1+pow(10,(train.authmonth_diff_mean-train.new_month_diff_mean)/400))

train['ELO_auth_month_lag'] = 1/(1+pow(10,(train.authmonth_lag_mean-train.non_authmonth_lag_mean)/400))
train['ELO_auth_new_month_lag'] = 1/(1+pow(10,(train.authmonth_lag_mean-train.new_month_lag_mean)/400))


# In[29]:


test_final['ELO_auth_month_diff'] = 1/(1+pow(10,(test_final.authmonth_diff_mean-test_final.non_authmonth_diff_mean)/400))
test_final['ELO_auth_new_month_diff'] = 1/(1+pow(10,(test_final.authmonth_diff_mean-test_final.new_month_diff_mean)/400))

test_final['ELO_auth_month_lag'] = 1/(1+pow(10,(test_final.authmonth_lag_mean-test_final.non_authmonth_lag_mean)/400))
test_final['ELO_auth_new_month_lag'] = 1/(1+pow(10,(test_final.authmonth_lag_mean-test_final.new_month_lag_mean)/400))


# In[30]:


rating = pd.concat((hist[['card_id','merchant_id','purchase_amount','purchase_date','month_lag','authorized_flag']],new_t[['card_id','merchant_id','purchase_amount','purchase_date','month_lag','authorized_flag']]))
rating['month_diff'] = ((datetime.datetime.today() - rating['purchase_date']).dt.days)//30
rating['month_diff'] += rating['month_lag']
rating.head()


# In[31]:


rating = rating.sort_values(['card_id','purchase_date'])
rating['num_of_purchase'] = rating.groupby(['card_id']).cumcount()+1

rating['K_koef'] = 0
rating.loc[(rating['authorized_flag']=='Y')&(rating['num_of_purchase']<=30),'K_koef'] = 40
rating.loc[(rating['authorized_flag']=='Y')&(rating['num_of_purchase']>30),'K_koef'] = 20

rating['S_koef'] = 0
rating.loc[(rating['authorized_flag']=='Y'),'S_koef'] = 1

rating['change_rating'] = rating['K_koef'] * (rating['S_koef'] - rating['purchase_amount'])
rating['rating_of_card'] = rating.groupby(['card_id'])['change_rating'].cumsum()

rating['change_rating2'] = rating['K_koef'] * (rating['S_koef'] - rating['month_diff'])
rating['rating_of_card2'] = rating.groupby(['card_id'])['change_rating2'].cumsum()

rating['change_rating3'] = rating['K_koef'] * (rating['S_koef'] - abs(rating['purchase_amount']))
rating['rating_of_card3'] = rating.groupby(['card_id'])['change_rating3'].cumsum()

rating['change_rating4'] = rating['K_koef'] * (rating['S_koef'] - abs(rating['month_diff']))
rating['rating_of_card4'] = rating.groupby(['card_id'])['change_rating4'].cumsum()


# In[32]:


rating = rating.sort_values(['merchant_id','purchase_date'])
rating['num_of_purchase'] = rating.groupby(['merchant_id']).cumcount()+1

rating['K_koef'] = 0
rating.loc[(rating['authorized_flag']=='Y')&(rating['num_of_purchase']<=30),'K_koef'] = 40
rating.loc[(rating['authorized_flag']=='Y')&(rating['num_of_purchase']>30),'K_koef'] = 20

rating['S_koef'] = 0
rating.loc[(rating['authorized_flag']=='Y'),'S_koef'] = 1

rating['change_rating_merch'] = rating['K_koef'] * (rating['S_koef'] - (rating['purchase_amount']*(-1)))
rating['rating_of_merch'] = rating.groupby(['merchant_id'])['change_rating_merch'].cumsum()

rating['change_rating_merch2'] = rating['K_koef'] * (rating['S_koef'] - rating['month_diff'])
rating['rating_of_merch2'] = rating.groupby(['merchant_id'])['change_rating_merch2'].cumsum()

rating['change_rating_merch3'] = rating['K_koef'] * (rating['S_koef'] - abs(rating['purchase_amount']))
rating['rating_of_merch3'] = rating.groupby(['merchant_id'])['change_rating_merch3'].cumsum()

rating['change_rating_merch4'] = rating['K_koef'] * (rating['S_koef'] - abs(rating['month_diff']))
rating['rating_of_merch4'] = rating.groupby(['merchant_id'])['change_rating_merch4'].cumsum()


# In[33]:


rating['ELO_1'] = 1/(1+pow(10,(rating.rating_of_card -rating.rating_of_merch )/400))
rating['ELO_2'] = 1/(1+pow(10,(rating.rating_of_card2-rating.rating_of_merch2)/400))
rating['ELO_3'] = 1/(1+pow(10,(rating.rating_of_card3-rating.rating_of_merch3)/400))
rating['ELO_4'] = 1/(1+pow(10,(rating.rating_of_card4-rating.rating_of_merch4)/400))

rating['ELO_11'] = 1/(1+pow(10,(rating.rating_of_card -rating.rating_of_merch )/400))
rating['ELO_21'] = 1/(1+pow(10,(rating.rating_of_card2-rating.rating_of_merch2)/400))
rating['ELO_31'] = 1/(1+pow(10,(rating.rating_of_card3-rating.rating_of_merch3)/400))
rating['ELO_41'] = 1/(1+pow(10,(rating.rating_of_card4-rating.rating_of_merch4)/400))

rating['ELO_12'] = 1/(1+pow(10,(rating.rating_of_card -rating.rating_of_merch )/400))
rating['ELO_22'] = 1/(1+pow(10,(rating.rating_of_card2-rating.rating_of_merch2)/400))
rating['ELO_32'] = 1/(1+pow(10,(rating.rating_of_card3-rating.rating_of_merch3)/400))
rating['ELO_42'] = 1/(1+pow(10,(rating.rating_of_card4-rating.rating_of_merch4)/400))


# In[34]:


#rating = replace_id(rating)
rating_agg = {'S_koef':'mean',
                 'change_rating'   :'mean',
                 'rating_of_card' :'max',
                 'rating_of_card2':'max',
                 'rating_of_card3':'max',
                 'rating_of_card4':'max',
                 'rating_of_merch':'max',
                 'rating_of_merch2':'max',
                 'rating_of_merch3':'max',
                 'rating_of_merch4':'max',
                 'ELO_1':'mean',
                 'ELO_2':'mean',
                 'ELO_3':'mean',
                 'ELO_4':'mean',
                 'ELO_11':'max',
                 'ELO_21':'max',
                 'ELO_31':'max',
                 'ELO_41':'max',
                 'ELO_12':'sum',
                 'ELO_22':'sum',
                 'ELO_32':'sum',
                 'ELO_42':'sum'
             }
rating_card = rating.groupby('card_id').agg(rating_agg).reset_index()
rating_card.head()


# In[35]:


train = pd.merge(train,rating_card,on='card_id',how='left')
test_final = pd.merge(test_final,rating_card,on='card_id',how='left')

train.fillna(0,inplace=True)
test_final.fillna(0,inplace=True)


# In[36]:


union = pd.concat((hist,new_t))
union[union.installments==-1]
union = union[union.installments != 0].sort_values(['purchase_date'])
union['installments_lag'] = union.month_lag+union.installments
union['installments_lag2'] = union.month_lag+union.installments
union.loc[union.installments_lag<0,'installments_lag'] = 0
union['credit_now'] = union.installments_lag*union.purchase_amount
union['credit_all'] = union.installments*union.purchase_amount
union2 = union.groupby('card_id').agg({'installments_lag':['min','max','sum','mean','var','median']
                                ,'installments_lag2':['min','max','sum','mean','var','median']
                            ,'credit_now':['min','max','sum','mean','var','median']
                            ,'credit_all':['min','max','sum','mean','var','median']})
union2.columns = ['install_'.join(col).strip() for col in union2.columns.values]
union2.reset_index(inplace=True)
union2.fillna(0,inplace=True)

train = pd.merge(train,union2,on='card_id',how='left')
test_final = pd.merge(test_final,union2,on='card_id',how='left')
train.fillna(0,inplace=True)
test_final.fillna(0,inplace=True)


# In[37]:


get_ipython().run_cell_magic('time', '', "train.to_csv(r'F:\\Work\\RepoSVNwn\\Data\\train.csv')\ntest_final.to_csv(r'F:\\Work\\RepoSVNwn\\Data\\test_final.csv')")


# In[38]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv(r'F:\\Work\\RepoSVNwn\\Data\\train.csv')\ntest_final = pd.read_csv(r'F:\\Work\\RepoSVNwn\\Data\\test_final.csv')")


# In[39]:


get_ipython().run_cell_magic('time', '', "print('Search correlations - ',end='')\nfeatures = [a for a in train.columns if a not in ['target','first_active_month','card_id']] \ncorr_matrix = train[features].corr()\nprint('ready')\ni = 1\nar = []\nfor a in corr_matrix.index:\n    ar.append(i)\n    i+=1\ncorr_matrix['rank'] = ar")


# In[40]:


fe_no_corr = []
corr_coef = 0.90
for a in corr_matrix.columns:
    if a == 'rank':
        continue
    r1 = corr_matrix[(corr_matrix.index ==a)]['rank'].values[0]
    try:
        r2 = corr_matrix[(abs(corr_matrix[a]) >= corr_coef)&(corr_matrix.index !=a)]['rank'].values[0]
    except:
        r2 = 100000
    #print(a,'with rank',corr_matrix[(corr_matrix.index ==a)]['rank'].values[0])
    #print('Correlation with',corr_matrix[(corr_matrix[a] >= corr_coef)&(corr_matrix.index !=a)].index.values,
    #     'with ranks:',corr_matrix[(corr_matrix[a] >= corr_coef)&(corr_matrix.index !=a)]['rank'].values)
    if r1<r2:
        fe_no_corr.append(a)
print('='*10)
print('Features is',len(fe_no_corr),':',fe_no_corr)
        


# In[41]:


get_ipython().run_cell_magic('time', '', "f_save = []\nfor a in fe_no_corr:\n    f_save.append(a)\nf_save.append('card_id')\nf_save.append('target')\nf_save.append('first_active_month')\ntrain[f_save].to_csv(r'F:\\Work\\RepoSVNwn\\Data\\train_clear.csv')\ntest_final[[a for a in f_save if a !='target']].to_csv(r'F:\\Work\\RepoSVNwn\\Data\\test_final_clear.csv')")


# In[42]:


train = train[f_save]
test_final = test_final[[a for a in f_save if a !='target']]


# In[43]:


+-----------------+   +------------------+   +------------------+               +----------------+
|                 |   |                  |   |                  |               |                |
| new_transaction |   | hist_transaction +-->+ auth_transaction |               |   merch_data   |
|                 |   |                  |   |                  |               |                |
+--------+--------+   +--------+---------+   +----+-------------+               +--------+-------+
         |                     |                  |                                      |
         |                     v                  |                                      |
         |            +--------+---------+        |                                      |
         |            |                  |        |                                      |
         |            |noauth_transaction|        |                                      |
         |            |                  |        |                                      |
         |            +--------+---------+        v                                      |
         |                     |       +----------+------+                               |
         |                     +------>+                 |                               |
         |                             |   train_data    +<------------------------------+
         +---------------------------->+ ~ 2500 features |
                                       +---------+-------+
                                                 |
                                                 |
                                                 v
                                          +------+------+
                                          |             |
                                          |    lgbm     |
                                          |             |
                                          +------+------+
                                                 |
                                                 v
                                +----------------+-----------------+
                                |                                  |
                                | FI best [50,100,150,200,300,400] |
                                |                                  |
                                +----------------+-----------------+
                                                 |
                                                 |
                                                 |
                                                 |
        +---------------+---------------+--------+--------+---------------+---------------+
        |               |               |                 |               |               |
        v               v               v                 v               v               v
 +------+------+ +------+------+ +------+------+   +------+------+ +------+------+ +------+------+
 |  K-fold 10  | |  K-fold 10  | |  K-fold 10  |   |  K-fold 10  | |  K-fold 10  | |  K-fold 10  |
 |  goss lgbm  | |  gbdt lgbm  | |  dart lgbm  |   |  lgbm tuned | |  lgbm tuned | |  lgbm tuned |
 |             | |             | |             |   |     goss    | |     dart    | |     gbrt    |
 +------+------+ +------+------+ +----------+--+   +--+----------+ +------+------+ +------+------+
        |               |                   |         |                   |               |
        |               |                   |         |                   |               |
        |               |                   v         v                   |               |
        |               |               +---+---------+----+              |               |
        |               +-------------->+                  +<-------------+               |
        |                               |   ByesianRidge   |                              |
        +------------------------------>+                  +<-----------------------------+
                                        +--------+---------+
                                                 |
                                                 v
                                        +--------+---------+
                                        |     Submit       |
                                        | param:outlier if |
                                        |   target <-15    |
                                        +------------------+


# In[44]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv')\ntrain_cards = train['card_id']\ntrain = pd.read_csv('../input/train_clear.csv')\ntrain['card_id'] = train_cards\ntest_final = pd.read_csv('../input/test_final_clear.csv')")


# In[45]:


cards = test_final[['card_id']]
train['outlier'] = 0
train.loc[train.target<-22.,'outlier'] = 1
outlier = train['outlier']


# In[46]:


#train.target = deanon_target(train.target)
target = train.target


# In[47]:


features = [a for a in train.columns if a not in ['target','first_active_month','card_id']] 
purchase_column = [a for a in features if 'purchase_amount' in a]
for a in [a for a in features if 'rating' in a]:
    purchase_column.append(a)


# In[48]:


for a in purchase_column:
    train['anon_'+a] = anon_target(train[a])
    test_final['anon_'+a] = anon_target(test_final[a])
print('Done')


# In[49]:


features = [a for a in train.columns if a not in ['card_id','target','first_active_month','outlier','p1','p1_5','p2_5','pred_valid','pred_valid2','pred_valid3']]


# In[50]:


get_ipython().run_cell_magic('time', '', 'cat_features = [\'feature_1\', \'feature_2\', \'feature_3\']\n\nfolds = KFold(n_splits=5, shuffle=True, random_state=15)\noof = np.zeros(len(train))\npredictions = np.zeros(len(test_final))\nstart = time.time()\n\nprint(\'ready\')\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features].values, target.values)):\n    param = {\'task\': \'train\',\n        \'boosting\': \'gbdt\',\n        \'objective\': \'regression\',\n        \'metric\': \'rmse\',\n        \'learning_rate\': 0.01,\n        \'subsample\': 0.9855232997390695,\n        \'max_depth\': 7,\n        \'top_rate\': 0.9064148448434349,\n        \'num_leaves\': 20,\n        \'min_child_weight\': 41.9612869171337,\n        \'other_rate\': 0.0721768246018207,\n        \'reg_alpha\': 9.677537745007898,\n        \'colsample_bytree\': 0.5665320670155495,\n        \'min_split_gain\': 9.820197773625843,\n        \'reg_lambda\': 8.2532317400459,\n        \'min_data_in_leaf\': 21,\n        \'verbose\': -1,\n        \'seed\':int(2**fold_*2),\n        \'bagging_seed\':int(2**fold_*2),\n        \'drop_seed\':int(2**fold_*2),\n        \'lambda_l2\':5,\n        \'lambda_l1\':5,\n        "feature_fraction": 0.85,\n        "bagging_freq": 1,\n        "bagging_fraction": 0.9 ,\n        "bagging_seed": int(2**fold_*2),\n        "max_bin":4,\n        "n_jobs":6\n        }\n    print("fold n°{}".format(fold_))\n    trn_data = lgb.Dataset(train.iloc[trn_idx][features],\n                           label=target.iloc[trn_idx],\n                           categorical_feature=cat_features\n                          )\n    val_data = lgb.Dataset(train.iloc[val_idx][features],\n                           label=target.iloc[val_idx],\n                           categorical_feature=cat_features\n                          )\n    print(\'Data ready\')\n    num_round = 10000\n    clf = lgb.train(param,\n                    trn_data,\n                    num_round,\n                    valid_sets = [trn_data, val_data],\n                    verbose_eval=100,\n                    early_stopping_rounds = 200)\n    \n    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n    \n    predictions += clf.predict(test_final[features], num_iteration=clf.best_iteration) / folds.n_splits\n\nprint("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))\n#print("CV score: {:<8.5f}".format(mean_squared_error(anon_target(np.where(oof<0,0.0000001,oof)), anon_target(np.where(target<0,0.0000001,target)))**0.5))')


# In[51]:


dictionary = cards
dictionary['target'] = predictions
#dictionary.loc[dictionary['target']<-12,'target'] = -33.21928095
dictionary[['card_id','target']].to_csv('/all_feature_210219.csv',index=False)
dictionary[['card_id','target']].head()


# In[52]:


fi2 = pd.DataFrame(clf.feature_importance())
fi2.columns = ['importance']
fi2['feature'] = features

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=fi2.sort_values(by="importance",
                                           ascending=False)[:100])


# In[53]:


get_ipython().run_cell_magic('time', '', "cat_col = []\nfor a in train.columns:\n    if train[a].nunique() < 5:\n        if a != 'outlier':\n            cat_col.append(a)\ncat_col")


# In[54]:


get_ipython().run_cell_magic('time', '', 'oof = []\npredictions = []\n\n\nfor a in [50,100,150,200,300,400]:\n    oof.append(np.zeros(len(train)))\n    predictions.append(np.zeros(len(test_final)))\n    \ni=0\nfor a in [50,100,150,200,300,400]:                       \n    features = fi2.sort_values(by="importance",ascending=False)[\'feature\'][:a]\n    cat_features = []\n\n    folds = KFold(n_splits=8, shuffle=True, random_state=a*2)\n    start = time.time()\n    print(\'ready\')\n\n    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features].values, target.values)):\n        print("fold n°{}".format(fold_))\n        param = {\'task\': \'train\',\n            \'boosting\': \'goss\',\n            \'objective\': \'regression\',\n            \'metric\': \'rmse\',\n            \'learning_rate\': 0.005,\n            \'subsample\': 0.9855232997390695,\n            \'max_depth\': 7,\n            \'top_rate\': 0.9064148448434349,\n            \'num_leaves\': 63,\n            \'min_child_weight\': 41.9612869171337,\n            \'other_rate\': 0.0721768246018207,\n            \'reg_alpha\': 9.677537745007898,\n            \'colsample_bytree\': 0.5665320670155495,\n            \'min_split_gain\': 9.820197773625843,\n            \'reg_lambda\': 8.2532317400459,\n            \'min_data_in_leaf\': 21,\n            \'verbose\': -1,\n            \'seed\':int(2**fold_*a),\n            \'bagging_seed\':int(2**fold_/a),\n            \'drop_seed\':int(2**fold_+a),\n            "n_jobs":20,\n            "lambda_l1": 0.05,\n            }\n        trn_data = lgb.Dataset(train.iloc[trn_idx][features],\n                               label=target.iloc[trn_idx],\n                               categorical_feature=cat_features\n                              )\n        val_data = lgb.Dataset(train.iloc[val_idx][features],\n                               label=target.iloc[val_idx],\n                               categorical_feature=cat_features\n                              )\n        print(\'Data ready\')\n        num_round = 10000\n        clf = lgb.train(param,\n                        trn_data,\n                        num_round,\n                        valid_sets = [trn_data, val_data],\n                        verbose_eval=100,\n                        #feval = minowski_dist,\n                        early_stopping_rounds = 200)\n\n        oof[i][val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n\n        predictions[i] += clf.predict(test_final[features], num_iteration=clf.best_iteration) / folds.n_splits\n\n    print("CV score: {:<8.5f}".format(mean_squared_error(oof[i], target)**0.5))\n    train[\'p1_\'+str(a)] = oof[i]\n    i+=1\nresult = np.zeros(len(train))\nfor a in range(0,len(oof)):\n    result += oof[a]/len(oof)\nprint("CV score: {:<8.5f}".format(mean_squared_error(result, target)**0.5))')


# In[55]:


get_ipython().run_cell_magic('time', '', 'oof2 = []\npredictions2 = []\n\nfor a in [50,100,150,200,300,400]:\n    oof2.append(np.zeros(len(train)))\n    predictions2.append(np.zeros(len(test_final)))\n    \ni=0\nfor a in [50,100,150,200,300,400]:                       \n    features = fi2.sort_values(by="importance",ascending=False)[\'feature\'][:a]\n    cat_features = []\n\n    folds = KFold(n_splits = 3,shuffle = True, random_state = 12*a)\n    start = time.time()\n\n    param = {\'num_leaves\': 31,\n              \'min_data_in_leaf\': 32, \n              \'objective\':\'regression\',\n              \'max_depth\': -1,\n              \'learning_rate\': 0.004,\n              "min_child_samples": 20,\n              "boosting": "dart",\n              "feature_fraction": 0.9,\n              "bagging_freq": 1,\n              "bagging_fraction": 0.9 ,\n              "bagging_seed": 11,\n              "metric": \'rmse\',\n              "lambda_l1": 0.18,\n              "nthread": 24,\n              "verbosity": -1}\n    print(\'ready\')\n    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features].values, target.values)):\n        print("fold n°{}".format(fold_))\n        trn_data = lgb.Dataset(train.iloc[trn_idx][features],\n                               label=target.iloc[trn_idx],\n                               categorical_feature=cat_features\n                              )\n        val_data = lgb.Dataset(train.iloc[val_idx][features],\n                               label=target.iloc[val_idx],\n                               categorical_feature=cat_features\n                              )\n        print(\'Data ready\')\n        num_round = 10000\n        clf = lgb.train(param,\n                        trn_data,\n                        num_round,\n                        valid_sets = [trn_data, val_data],\n                        verbose_eval=100,\n                        #feval = minowski_dist,\n                        early_stopping_rounds = 200)\n\n        oof2[i][val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n\n        predictions2[i] += clf.predict(test_final[features], num_iteration=clf.best_iteration) / folds.n_splits\n\n    print("CV score: {:<8.5f}".format(mean_squared_error(oof2[i], target)**0.5))\n    train[\'p2_\'+str(a)] = oof2[i]\n    i+=1\n    \nresult2 = np.zeros(len(train))\nfor a in range(0,len(oof2)):\n    result2 += oof2[a]/len(oof2)\nprint("CV score: {:<8.5f}".format(mean_squared_error(result2, target)**0.5))')


# In[56]:


dictionary = cards
dictionary['target'] = predictions
dictionary[['card_id','target']].to_csv('/dart_150219_cv_3.65983.csv',index=False)
dictionary[['card_id','target']].head()


# In[57]:


get_ipython().run_cell_magic('time', '', 'oof3 = []\npredictions3 = []\n\nfor a in [50,100,150,200,300,400]:\n    oof3.append(np.zeros(len(train)))\n    predictions3.append(np.zeros(len(test_final)))\n    \ni=0\nfor a in [50,100,150,200,300,400]:                       \n    features = fi2.sort_values(by="importance",ascending=False)[\'feature\'][:a]\n    cat_features = []\n\n    folds = KFold(n_splits = 7,shuffle = True, random_state = 12*a)\n    start = time.time()\n\n    print(\'ready\')\n\n    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features].values, target.values)):\n        print("fold n°{}".format(fold_))\n        param = {\'num_leaves\': 10,\n             \'min_data_in_leaf\': 32, \n             \'objective\':\'regression\',\n             \'max_depth\': -1,\n             \'learning_rate\': 0.005,\n             "min_child_samples": 20,\n             "boosting": "gbdt",\n             "feature_fraction": 0.9,\n             "bagging_freq": 1,\n             "bagging_fraction": 0.9 ,\n             "bagging_seed": int(2**fold_*a),\n             "metric": \'rmse\',\n             "lambda_l1": 0.11,\n             "nthread": 10,\n             "verbosity": -1}\n        trn_data = lgb.Dataset(train.iloc[trn_idx][features],\n                               label=target.iloc[trn_idx],\n                               categorical_feature=cat_features\n                              )\n        val_data = lgb.Dataset(train.iloc[val_idx][features],\n                               label=target.iloc[val_idx],\n                               categorical_feature=cat_features\n                              )\n        print(\'Data ready\')\n        num_round = 10000\n        clf = lgb.train(param,\n                        trn_data,\n                        num_round,\n                        valid_sets = [trn_data, val_data],\n                        verbose_eval=100,\n                        #feval = minowski_dist,\n                        early_stopping_rounds = 200)\n\n        oof3[i][val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n\n        predictions3[i] += clf.predict(test_final[features], num_iteration=clf.best_iteration) / folds.n_splits\n\n    print("CV score: {:<8.5f}".format(mean_squared_error(oof3[i], target)**0.5))\n    train[\'p3_\'+str(a)] = oof3[i]\n    i+=1\nresult3 = np.zeros(len(train))\nfor a in range(0,len(oof3)):\n    result3 += oof3[a]/len(oof3)\nprint("CV score: {:<8.5f}".format(mean_squared_error(result3, target)**0.5))')


# In[58]:


dictionary = cards
dictionary['target'] = predictions
dictionary[['card_id','target']].to_csv('/gbdt_150219_cv_3.65867.csv',index=False)
dictionary[['card_id','target']].head()


# In[59]:


get_ipython().run_cell_magic('time', '', 'oof5 = []\npredictions5 = []\n\nfor a in [50,100,150,200,300,400]:\n    oof5.append(np.zeros(len(train)))\n    predictions5.append(np.zeros(len(test_final)))\n    \ni=0\nfor a in [50,100,150,200,300,400]:                       \n    features = fi2.sort_values(by="importance",ascending=False)[\'feature\'][:a]\n    cat_features = []\n\n    folds = KFold(n_splits=5, shuffle=True, random_state=a*2)\n    start = time.time()\n    print(\'ready\')\n\n    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features].values, target.values)):\n        print("fold n°{}".format(fold_))\n        param = {\n        #Core Parameters\n            \'task\': \'train\',\n            \'objective\': \'regression\',\n            \'boosting\': \'gbrt\',\n            \'learning_rate\': .01,\n            \'num_leaves\': 114, #2^max_depth*70%\n            \'num_threads\': 24,\n            \'device_type\': \'cpu\',\n            \'seed\': int(2**a*fold_),\n        #Learning Control Parameters\n            \'max_depth\': 3,\n            \'min_data_in_leaf\': 20,\n            \'min_sum_hessian_in_leaf\': 1e-3,\n            \'bagging_fraction\': .75,\n            \'bagging_freq\': 2,\n            \'bagging_seed\': int(2**a*fold_),\n            \'feature_fraction\': .75,\n            \'feature_fraction_seed\': int(2**a*fold_),\n            \'max_delta_step\': 0.0,\n            \'lambda_l1\': 170.,\n            \'lambda_l2\': 227.,\n            \'min_gain_to_split\': 0.0,\n            \'min_data_per_group\': 100,\n            \'max_cat_threshold\': 32,\n            \'cat_l2\': 10.0,\n            \'cat_smooth\': 10.0,\n            \'max_cat_to_onehot\': 4,\n            \'monotone_constraints\': None,\n            \'feature_contri\': None,\n            \'forcedsplits_filename\': \'\',\n            \'refit_decay_rate\': .9,\n            \'verbosity\': 1,\n            \'max_bin\': 171,\n            \'min_data_in_bin\': 3,\n            \'bin_construct_sample_cnt\': 200000,\n            \'histogram_pool_size\': -1.0,\n            \'data_random_seed\': int(2**a*fold_),\n            \'snapshot_freq\': -1,\n            \'iniscore_filename\': \'\',\n            \'valid_data_iniscore\': \'\',\n            \'pre_partition\': False,\n            \'enable_bundle\': True,\n            \'max_conflict\': 0.0,\n            \'is_enable_sparse\': True,\n            \'sparse_threshold\': .8,\n            \'use_missing\': True,\n            \'zero_as_missing\': True,\n            \'two_round\': False,\n            \'save_binary\': False,\n            \'enable_load_from_binary_file\': True,\n            \'header\': False,\n            \'label_column\': \'\',\n            \'weight_columns\': \'\',\n            \'group_column\': \'\',\n            \'ignore_columns\': \'\',\n            \'boost_from_average\': True,\n            \'reg_sqrt\': False,\n            \'alpha\': .9,\n            \'fair_c\': 1.0,\n            \'poisson_max_delta_step\': .7,\n            \'tweedie_variance_power\': 1.5,\n        #Metric Parameters\n            \'metric\': \'l2_root\',\n            \'metric_freq\': 1,\n            \'is_provide_training_metric\': False\n        }\n        trn_data = lgb.Dataset(train.iloc[trn_idx][features],\n                               label=target.iloc[trn_idx],\n                               categorical_feature=cat_features\n                              )\n        val_data = lgb.Dataset(train.iloc[val_idx][features],\n                               label=target.iloc[val_idx],\n                               categorical_feature=cat_features\n                              )\n        print(\'Data ready\')\n        num_round = 10000\n        clf = lgb.train(param,\n                        trn_data,\n                        num_round,\n                        valid_sets = [trn_data, val_data],\n                        verbose_eval=100,\n                        #feval = minowski_dist,\n                        early_stopping_rounds = 200)\n\n        oof5[i][val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n\n        predictions5[i] += clf.predict(test_final[features], num_iteration=clf.best_iteration) / folds.n_splits\n\n    print("CV score: {:<8.5f}".format(mean_squared_error(oof5[i], target)**0.5))\n    train[\'p5_\'+str(a)] = oof5[i]\n    i+=1\nresult = np.zeros(len(train))\nfor a in range(0,len(oof5)):\n    result += oof5[a]/len(oof5)\nprint("CV score: {:<8.5f}".format(mean_squared_error(result, target)**0.5))')


# In[60]:


get_ipython().run_cell_magic('time', '', 'oof6 = []\npredicions6 = []\n\nfor a in [50,100,150,200,300,400]:\n    oof6.append(np.zeros(len(train)))\n    predicions6.append(np.zeros(len(test_final)))\n    \ni=0\nfor a in [50,100,150,200,300,400]:                       \n    features = fi2.sort_values(by="importance",ascending=False)[\'feature\'][:a]\n    cat_features = []\n\n    folds = KFold(n_splits=5, shuffle=True, random_state=a*2)\n    start = time.time()\n    print(\'ready\')\n\n    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features].values, target.values)):\n        print("fold n°{}".format(fold_))\n        param = {\n        #Core Parameters\n            \'task\': \'train\',\n            \'objective\': \'regression\',\n            \'boosting\': \'goss\',\n            \'learning_rate\': .01,\n            \'num_leaves\': 171, #2^max_depth*70%\n            \'num_threads\': 24,\n            \'device_type\': \'cpu\',\n            \'seed\': int(2**a*fold_),\n        #Learning Control Parameters\n            \'max_depth\': 3,\n            \'min_data_in_leaf\': 20,\n            \'min_sum_hessian_in_leaf\': 1e-3,\n            \'feature_fraction\': .75,\n            \'feature_fraction_seed\': int(2**a*fold_),\n            \'max_delta_step\': 0.0,\n            \'lambda_l1\': 114.,\n            \'lambda_l2\': 199.,\n            \'min_gain_to_split\': 0.0,\n            \'top_rate\': .4, #goss\n            \'oter_rate\': .4, #goss\n            \'min_data_per_group\': 100,\n            \'max_cat_threshold\': 32,\n            \'cat_l2\': 10.0,\n            \'cat_smooth\': 10.0,\n            \'max_cat_to_onehot\': 4,\n            \'monotone_constraints\': None,\n            \'feature_contri\': None,\n            \'forcedsplits_filename\': \'\',\n            \'refit_decay_rate\': .9,\n        #IO Parameters\n            \'verbosity\': 1,\n            \'max_bin\': 171,\n            \'min_data_in_bin\': 3,\n            \'bin_construct_sample_cnt\': 200000,\n            \'histogram_pool_size\': -1.0,\n            \'data_random_seed\': int(2**a*fold_),\n            \'snapshot_freq\': -1,\n            \'iniscore_filename\': \'\',\n            \'valid_data_iniscore\': \'\',\n            \'pre_partition\': False,\n            \'enable_bundle\': True,\n            \'max_conflict\': 0.0,\n            \'is_enable_sparse\': True,\n            \'sparse_threshold\': .8,\n            \'use_missing\': True,\n            \'zero_as_missing\': True,\n            \'two_round\': False,\n            \'save_binary\': False,\n            \'enable_load_from_binary_file\': True,\n            \'header\': False,\n            \'label_column\': \'\',\n            \'weight_columns\': \'\',\n            \'group_column\': \'\',\n            \'ignore_columns\': \'\',\n            \'boost_from_average\': True,\n            \'reg_sqrt\': False,\n            \'alpha\': .9,\n            \'fair_c\': 1.0,\n            \'poisson_max_delta_step\': .7,\n            \'tweedie_variance_power\': 1.5,\n        #Metric Parameters\n            \'metric\': \'l2_root\',\n            \'metric_freq\': 1,\n            \'is_provide_training_metric\': False,\n        #Network Parameters\n            \'num_machines\': 1,\n            \'local_listen_port\': 12400,\n            \'time-out\': 120,\n            \'machine_list_filename\': \'\',\n            \'machines\': \'\',\n        #GPU Parameters\n            \'gpu_platform_id\': 0,\n            \'gpu_device_id\': 0,\n            \'gpu_use_dp\': True\n        }\n        \n        trn_data = lgb.Dataset(train.iloc[trn_idx][features],\n                               label=target.iloc[trn_idx],\n                               categorical_feature=cat_features\n                              )\n        val_data = lgb.Dataset(train.iloc[val_idx][features],\n                               label=target.iloc[val_idx],\n                               categorical_feature=cat_features\n                              )\n        print(\'Data ready\')\n        num_round = 10000\n        clf = lgb.train(param,\n                        trn_data,\n                        num_round,\n                        valid_sets = [trn_data, val_data],\n                        verbose_eval=100,\n                        #feval = minowski_dist,\n                        early_stopping_rounds = 200)\n\n        oof6[i][val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n\n        predicions6[i] += clf.predict(test_final[features], num_iteration=clf.best_iteration) / folds.n_splits\n\n    print("CV score: {:<8.5f}".format(mean_squared_error(oof6[i], target)**0.5))\n    train[\'p6_\'+str(a)] = oof6[i]\n    i+=1\nresult = np.zeros(len(train))\nfor a in range(0,len(oof6)):\n    result += oof6[a]/len(oof6)\nprint("CV score: {:<8.5f}".format(mean_squared_error(result, target)**0.5))')


# In[61]:


get_ipython().run_cell_magic('time', '', 'oof7 = []\npredicions7 = []\n\nfor a in [50,100,150,200,300,400]:\n    oof7.append(np.zeros(len(train)))\n    predicions7.append(np.zeros(len(test_final)))\n    \ni=0\nfor a in [50,100,150,200,300,400]:                       \n    features = fi2.sort_values(by="importance",ascending=False)[\'feature\'][:a]\n    cat_features = []\n\n    folds = KFold(n_splits=5, shuffle=True, random_state=a*2)\n    start = time.time()\n    print(\'ready\')\n\n    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features].values, target.values)):\n        print("fold n°{}".format(fold_))\n        param = {\n        #Core Parameters\n            \'task\': \'train\',\n            \'objective\': \'regression\',\n            \'boosting\': \'dart\',\n            \'learning_rate\': .1,\n            \'num_leaves\': 114, #2^max_depth*70%\n            \'num_threads\': 24,\n            \'device_type\': \'cpu\',\n            \'seed\': int(2**a*fold_),\n        #Learning Control Parameters\n            \'max_depth\': 3,\n            \'min_data_in_leaf\': 20,\n            \'min_sum_hessian_in_leaf\': 1e-3,\n            \'bagging_fraction\': .75,\n            \'bagging_freq\': 2,\n            \'bagging_seed\': int(2**a*fold_),\n            \'feature_fraction\': .75,\n            \'feature_fraction_seed\': int(2**a*fold_),\n            \'max_delta_step\': 0.0,\n            \'lambda_l1\': 0.,\n            \'lambda_l2\': 114.,\n            \'min_gain_to_split\': 0.0,\n            \'drop_rate\': .55, #dart\n            \'max_dop\': 64, #dart\n            \'skip_drop\': .1, #dart\n            \'xgboost_dart_mode\': False, #dart\n            \'uniform_drop\': True, #dart\n            \'drop_seed\': int(2**a*fold_), #dart\n            \'min_data_per_group\': 100,\n            \'max_cat_threshold\': 32,\n            \'cat_l2\': 10.0,\n            \'cat_smooth\': 10.0,\n            \'max_cat_to_onehot\': 4,\n            \'monotone_constraints\': None,\n            \'feature_contri\': None,\n            \'forcedsplits_filename\': \'\',\n            \'refit_decay_rate\': .9,\n        #IO Parameters\n            \'verbosity\': 1,\n            \'max_bin\': 255,\n            \'min_data_in_bin\': 3,\n            \'bin_construct_sample_cnt\': 200000,\n            \'histogram_pool_size\': -1.0,\n            \'data_random_seed\': int(2**a*fold_),\n            \'snapshot_freq\': -1,\n            \'iniscore_filename\': \'\',\n            \'valid_data_iniscore\': \'\',\n            \'pre_partition\': False,\n            \'enable_bundle\': True,\n            \'max_conflict\': 0.0,\n            \'is_enable_sparse\': True,\n            \'sparse_threshold\': .8,\n            \'use_missing\': True,\n            \'zero_as_missing\': True,\n            \'two_round\': False,\n            \'save_binary\': False,\n            \'enable_load_from_binary_file\': True,\n            \'header\': False,\n            \'label_column\': \'\',\n            \'weight_columns\': \'\',\n            \'group_column\': \'\',\n            \'ignore_columns\': \'\',\n            \'boost_from_average\': True,\n            \'reg_sqrt\': False,\n            \'alpha\': .9,\n            \'fair_c\': 1.0,\n            \'poisson_max_delta_step\': .7,\n            \'tweedie_variance_power\': 1.5,\n        #Metric Parameters\n            \'metric\': \'l2_root\',\n            \'metric_freq\': 1,\n            \'is_provide_training_metric\': False,\n        #Network Parameters\n            \'num_machines\': 1,\n            \'local_listen_port\': 12400,\n            \'time-out\': 120,\n            \'machine_list_filename\': \'\',\n            \'machines\': \'\',\n        #GPU Parameters\n            \'gpu_platform_id\': 0,\n            \'gpu_device_id\': 0,\n            \'gpu_use_dp\': True\n        }\n        trn_data = lgb.Dataset(train.iloc[trn_idx][features],\n                               label=target.iloc[trn_idx],\n                               categorical_feature=cat_features\n                              )\n        val_data = lgb.Dataset(train.iloc[val_idx][features],\n                               label=target.iloc[val_idx],\n                               categorical_feature=cat_features\n                              )\n        print(\'Data ready\')\n        num_round = 10000\n        clf = lgb.train(param,\n                        trn_data,\n                        num_round,\n                        valid_sets = [trn_data, val_data],\n                        verbose_eval=100,\n                        #feval = minowski_dist,\n                        early_stopping_rounds = 200)\n\n        oof7[i][val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n\n        predicions7[i] += clf.predict(test_final[features], num_iteration=clf.best_iteration) / folds.n_splits\n\n    print("CV score: {:<8.5f}".format(mean_squared_error(oof7[i], target)**0.5))\n    train[\'p1_\'+str(a)] = oof7[i]\n    i+=1\nresult = np.zeros(len(train))\nfor a in range(0,len(oof7)):\n    result += oof7[a]/len(oof7)\nprint("CV score: {:<8.5f}".format(mean_squared_error(result, target)**0.5))')


# In[62]:


train_stack = []
test_stack = []
for a in range(0,len(oof)):
    train_stack.append(oof[a])
    train_stack.append(oof2[a])
    train_stack.append(oof3[a])
    
    train_stack.append(oof5[a])
    train_stack.append(oof6[a])
    train_stack.append(oof7[a])
    
    test_stack.append(predictions[a])
    test_stack.append(predictions2[a])
    test_stack.append(predictions3[a])
    
    test_stack.append(predictions5[a])
    test_stack.append(predicions6[a])
    test_stack.append(predicions7[a])
#train_stack.append(oof4)
#test_stack.append(predictions4)

train_stack = np.vstack(train_stack).transpose()
test_stack = np.vstack(test_stack).transpose()

folds = KFold(n_splits=20,shuffle=True,random_state=4520)
oof_stack = np.zeros(train_stack.shape[0])
predictions_stack = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, target)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    print("-" * 10 + "Stacking " + str(fold_) + "-" * 10)
    clf = BayesianRidge()
    clf.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf.predict(val_data)
    print("CV score fold: {:<8.5f}".format(mean_squared_error(oof_stack[val_idx], target[val_idx])**0.5))
    predictions_stack += clf.predict(test_stack) / folds.n_splits

print("CV score summary:",np.sqrt(mean_squared_error(target.values, oof_stack)))


# In[63]:


dictionary = cards
dictionary['target'] = predictions_stack
border = -14.5
print('Count outliers:',len(dictionary[dictionary.target<border]))
dictionary[['card_id','target']].to_csv('../stuck_models_no_cat_220219_cv_'+str(round(np.sqrt(mean_squared_error(target.values, oof_stack)),5))+'.csv',index=False)
dictionary[['card_id','target']].head()


# In[64]:




