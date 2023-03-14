#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from datetime import datetime

import os
from os.path import join as pjoin

data_root = '../input/make-data-ready'
print(os.listdir(data_root))

pd.set_option('display.max_rows',200)

from sklearn.preprocessing import LabelEncoder

from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def load_data(data='train',n=2):
    df = pd.DataFrame()
    for i in range(n) :
        if data=='train':
            if i > 8 :
                break
            dfpart = pd.read_pickle(pjoin(data_root,f'train_{i}.pkl'))
        elif data=='test':
            if i > 2 :
                break
            dfpart = pd.read_pickle(pjoin(data_root,f'test_{i}.pkl'))
        df = pd.concat([df,dfpart])
        del dfpart
    return df
        


# In[3]:


df_train = load_data(n=9)
df_test = load_data('test',n=4)


# In[4]:


df = pd.concat([df_train, df_test])


# In[5]:


col_drop = ['Date_Year', 'Date_Month', 'Date_Week','Date_Hour','device_isMobile','device_deviceCategory',
       'Date_Day', 'Date_Dayofweek', 'Date_Dayofyear', 'Date_Is_month_end',
       'Date_Is_month_start', 'Date_Is_quarter_end', 'Date_Is_quarter_start',
       'Date_Is_year_end', 'Date_Is_year_start','totals_visits',
           'date','visitId','totals_totalTransactionRevenue','geoNetwork_city','geoNetwork_continent',
            'geoNetwork_metro','geoNetwork_networkDomain',
'geoNetwork_region','geoNetwork_subContinent','trafficSource_adContent',
            'trafficSource_adwordsClickInfo.adNetworkType','trafficSource_adwordsClickInfo.gclId',
'trafficSource_adwordsClickInfo.slot','trafficSource_campaign',
            'trafficSource_keyword','trafficSource_referralPath','trafficSource_medium',
            'customDimensions_value','customDimensions_index','trafficSource_source',
           'totals_bounces','visitNumber','totals_newVisits']
df.drop(col_drop, axis=1, inplace=True)


# In[6]:


country_drop=df.groupby('geoNetwork_country')['totals_transactions'].sum()[df.groupby('geoNetwork_country')['totals_transactions'].sum().sort_values()<4].index.tolist()
df.loc[df[df.geoNetwork_country.isin(country_drop)].index,'geoNetwork_country'] = 'NaN'

df.loc[df[~df.device_browser.isin(['Edge', 'Internet Explorer', 'Firefox', 'Safari', 'Chrome'])].index,'device_browser'] = 'NaN'
df.loc[df[~df.device_operatingSystem.isin(['Android', 'iOS', 'Linux', 'Chrome OS', 'Windows', 'Macintosh'])].index,'device_operatingSystem'] = 'NaN'


# In[7]:


col_lb = ['channelGrouping','device_browser','device_operatingSystem', 'geoNetwork_country',
          'trafficSource_adwordsClickInfo.isVideoAd','trafficSource_isTrueDirect']
for col in col_lb:
    lb = LabelEncoder()
    df[col]=lb.fit_transform(df[col])


# In[8]:


to_median = ['channelGrouping','device_browser','device_operatingSystem','geoNetwork_country','trafficSource_adwordsClickInfo.isVideoAd','trafficSource_isTrueDirect','trafficSource_adwordsClickInfo.page']
to_sum =['totals_hits','totals_pageviews','totals_timeOnSite','totals_transactionRevenue', 'totals_transactions']
to_mean =['totals_hits','totals_pageviews','totals_sessionQualityDim']
to_std = ['totals_hits','totals_pageviews']
to_time = 'visitStartTime'


# In[9]:


target_period = pd.date_range(start='2016-08-01',end='2018-12-01', freq='2MS')
train_period = target_period.to_series().shift(periods=-210, freq='d',axis= 0)
time_to = train_period[train_period.index>np.datetime64('2016-08-01')].astype('int')//10**9
time_end = target_period.to_series().shift(periods=-45, freq='d',axis= 0)[4:]


# In[10]:


user_x = df.iloc[df_train.shape[0]:,:]
test_x = pd.concat([user_x.groupby('fullVisitorId')[to_median].median().add_suffix('_median'),
user_x.groupby('fullVisitorId')['visitStartTime'].agg(['first','last']).add_suffix('_time').sub(time_to.values[-1]).abs(),
user_x.groupby('fullVisitorId')['visitStartTime'].apply(lambda x: x.max() -x.min()).rename('time_diff'),
user_x.groupby('fullVisitorId')[to_sum].sum().add_suffix('_sum'),
user_x.groupby('fullVisitorId')[to_mean].mean().add_suffix('_mean'),
user_x.groupby('fullVisitorId')[to_std].std(ddof=0).add_suffix('_std')],axis=1).reset_index()

test_ID= test_x.fullVisitorId
test_x = test_x.drop(['fullVisitorId'], axis=1,errors='ignore')
test_x = test_x.astype('int')


# In[11]:


i=4
user_x = df[(df.visitStartTime>=(time_to.index.astype('int')//10**9)[i]) & (df.visitStartTime<(time_end.index.astype('int')//10**9)[i])]
user_y = df[(df.visitStartTime>=time_to.values[i]) & (df.visitStartTime<time_to.values[i+1])]
train_x = pd.concat([user_x.groupby('fullVisitorId')[to_median].median().add_suffix('_median'),
user_x.groupby('fullVisitorId')['visitStartTime'].agg(['first','last']).add_suffix('_time').sub(time_to.values[i]).abs(),
user_x.groupby('fullVisitorId')['visitStartTime'].apply(lambda x: x.max() -x.min()).rename('time_diff'),
user_x.groupby('fullVisitorId')[to_sum].sum().add_suffix('_sum'),
user_x.groupby('fullVisitorId')[to_mean].mean().add_suffix('_mean'),
user_x.groupby('fullVisitorId')[to_std].std(ddof=0).add_suffix('_std')],axis=1).reset_index()

merged=train_x.merge(user_y.groupby('fullVisitorId')['totals_transactionRevenue'].sum().reset_index(),                          how='left', on='fullVisitorId')
val_y = merged.totals_transactionRevenue
val_y.fillna(0, inplace=True)
val_x = merged.drop(['fullVisitorId','totals_transactionRevenue'], axis=1,errors='ignore')
val_x = val_x.astype('int')


# In[12]:


import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error


# In[13]:


params={'learning_rate': 0.01,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 31,
        'verbose': 1,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.9,
        "random_state":42,
        'max_depth': 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "bagging_frequency" : 5,
        'lambda_l2': 0.5,
        'lambda_l1': 0.5,
        'min_child_samples': 36
       }
xgb_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456,
        'importance_type': 'total_gain'
    }

cat_param = {
    'learning_rate' :0.03,
    'depth' :10,
    'eval_metric' :'RMSE',
    'od_type' :'Iter',
    'metric_period ' : 50,
    'od_wait' : 20,
    'seed' : 42
    
}


# In[14]:


oof_reg_preds = np.zeros(val_x.shape[0])
oof_reg_preds1 = np.zeros(val_x.shape[0])
oof_reg_preds2 = np.zeros(val_x.shape[0])
merge_pred = np.zeros(val_x.shape[0])
merge_preds = np.zeros(val_x.shape[0])
sub_preds = np.zeros(test_x.shape[0])
alist = list(range(time_to.shape[0]-1))
alist.remove(4)
folds = alist
folds=range(len(alist)-1)

for i in alist:
    print(i)
    user_x = df[(df.visitStartTime>=(time_to.index.astype('int')//10**9)[i]) & (df.visitStartTime<(time_end.index.astype('int')//10**9)[i])]
    user_y = df[(df.visitStartTime>=time_to.values[i]) & (df.visitStartTime<time_to.values[i+1])]
    train_x = pd.concat([user_x.groupby('fullVisitorId')[to_median].median().add_suffix('_median'),
    user_x.groupby('fullVisitorId')['visitStartTime'].agg(['first','last']).add_suffix('_time').sub(time_to.values[i]).abs(),
    user_x.groupby('fullVisitorId')['visitStartTime'].apply(lambda x: x.max() -x.min()).rename('time_diff'),
    user_x.groupby('fullVisitorId')[to_sum].sum().add_suffix('_sum'),
    user_x.groupby('fullVisitorId')[to_mean].mean().add_suffix('_mean'),
    user_x.groupby('fullVisitorId')[to_std].std(ddof=0).add_suffix('_std')],axis=1).reset_index()
    
    merged=train_x.merge(user_y.groupby('fullVisitorId')['totals_transactionRevenue'].sum().reset_index(),                              how='left', on='fullVisitorId')
    train_y = merged.totals_transactionRevenue
    train_y.fillna(0, inplace=True)
    train_x = merged.drop(['fullVisitorId','totals_transactionRevenue'], axis=1,errors='ignore')
    train_x = train_x.astype('int')    
    
    reg = lgb.LGBMRegressor(**params,n_estimators=1100)
    xgb = XGBRegressor(**xgb_params, n_estimators=1000)
    cat = CatBoostRegressor(iterations=1000,learning_rate=0.03,
                            depth=10,
                            eval_metric='RMSE',
                            random_seed = 42,
                            bagging_temperature = 0.2,
                            od_type='Iter',
                            metric_period = 50,
                            od_wait=20)
    print("-"* 20 + "LightGBM Training" + "-"* 20)
    reg.fit(train_x, np.log1p(train_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,verbose=100,eval_metric='rmse')
    print("-"* 20 + "XGboost Training" + "-"* 20)
    xgb.fit(train_x, np.log1p(train_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,eval_metric='rmse',verbose=100)
    print("-"* 20 + "Catboost Training" + "-"* 20)
    cat.fit(train_x, np.log1p(train_y), eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,use_best_model=True,verbose=100)
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_x.columns
    imp_df['gain_reg'] = np.zeros(train_x.shape[1])
    imp_df['gain_xgb'] = np.zeros(train_x.shape[1])
    imp_df['gain_cat'] = np.zeros(train_x.shape[1])
    imp_df['gain_reg'] += reg.booster_.feature_importance(importance_type='gain')/ len(folds)
    imp_df['gain_xgb'] += xgb.feature_importances_/ len(folds)
    imp_df['gain_cat'] += np.array(cat.get_feature_importance())/ len(folds)
    
    # LightGBM
    oof_reg_preds = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    lgb_preds = reg.predict(test_x, num_iteration=reg.best_iteration_)
    lgb_preds[lgb_preds < 0] = 0
    
    
    # Xgboost
    oof_reg_preds1 = xgb.predict(val_x)
    oof_reg_preds1[oof_reg_preds1 < 0] = 0
    xgb_preds = xgb.predict(test_x)
    xgb_preds[xgb_preds < 0] = 0
    
    # catboost
    oof_reg_preds2 = cat.predict(val_x)
    oof_reg_preds2[oof_reg_preds2 < 0] = 0
    cat_preds = cat.predict(test_x)
    cat_preds[cat_preds < 0] = 0
        
    #merge all prediction
    merge_pred = oof_reg_preds * 0.4 + oof_reg_preds1 * 0.3 +oof_reg_preds2 * 0.3
    merge_preds += (oof_reg_preds / len(folds)) * 0.4 + (oof_reg_preds1 / len(folds)) * 0.3 + (oof_reg_preds2 / len(folds)) * 0.3
    sub_preds += (lgb_preds / len(folds)) * 0.4 + (xgb_preds / len(folds)) * 0.3 + (cat_preds / len(folds)) * 0.3
    
    
print("LGBM  ", mean_squared_error(np.log1p(val_y), oof_reg_preds) ** .5)
print("XGBoost  ", mean_squared_error(np.log1p(val_y), oof_reg_preds1) ** .5)
print("CatBoost  ", mean_squared_error(np.log1p(val_y), oof_reg_preds2) ** .5)
print("merged  ", mean_squared_error(np.log1p(val_y), merge_pred) ** .5)
print("stack_merged  ", mean_squared_error(np.log1p(val_y), merge_preds) ** .5)
print("Zeros  ", mean_squared_error(np.log1p(val_y), np.zeros(val_x.shape[0])) ** .5)


# In[15]:


plt.figure(figsize=(8, 12))
sns.barplot(x='gain_reg', y='feature', data=imp_df.sort_values('gain_reg', ascending=False))


# In[16]:


plt.figure(figsize=(8, 12))
sns.barplot(x='gain_xgb', y='feature', data=imp_df.sort_values('gain_xgb', ascending=False))


# In[17]:


plt.figure(figsize=(8, 12))
sns.barplot(x='gain_cat', y='feature', data=imp_df.sort_values('gain_cat', ascending=False))


# In[18]:


sub_df = pd.DataFrame(test_ID)
sub_df["PredictedLogRevenue"] = sub_preds
sub_df.to_csv("stacked_result.csv", index=False)

