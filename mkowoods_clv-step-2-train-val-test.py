#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import datetime
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas.io.json import json_normalize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import gc
gc.enable()
color = sns.color_palette()


from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        


# In[2]:


NUMERIC_COLUMNS_TO_REFORMAT = [
    'totals_hits',
    'totals_pageviews',
    'totals_timeOnSite',
    'totals_totalTransactionRevenue', 
    'totals_transactions'
]

def type_correct_numeric(df):
    for col in NUMERIC_COLUMNS_TO_REFORMAT:
        df[col] = df[col].fillna(0).astype(int)
    
    return df

def process_date_time(df):
    print('process date')
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['year'] = df['date'].dt.year
    df['weekofyear'] = df['date'].dt.weekofyear
#    df['weekday'] = df['date'].dt.weekday
    return df

def add_index_and_deduplicate(df):
    n_rows, n_cols = df.shape

    df['unique_row_id'] = df.fullVisitorId.map(str) + '.' + df.visitId.map(str)
    df.index = df.unique_row_id
    deduped_df = df.loc[~df.index.duplicated(keep='first')]
    print('De dupliceated {} rows'.format(n_rows - deduped_df.shape[0]))
    return deduped_df

def fillnas(df):
    df = df['trafficSource_isTrueDirect'].fillna(False)
    return 


# In[3]:


get_ipython().run_cell_magic('time', '', 'path = "../input/google-analytics-preprocessed-dataset/"\ndf = pd.concat([\n    pd.read_pickle(path + \'train_v2_clean.pkl\'),\n    pd.read_pickle(path + \'test_v2_clean.pkl\')\n])\n')


# In[4]:


print('Processing Training Data...')
df =  process_date_time(df)
df = type_correct_numeric(df)
df = add_index_and_deduplicate(df)
# print()
# print('Processing Test Data...')
# test_df =  process_date_time(test_df)
# test_df = type_correct_numeric(test_df)
# test_df = add_index_and_deduplicate(test_df)

gc.collect()
# full_df = process_date_time(full_df)
# full_df = correct_dtypes(full_df)


# In[5]:


print('Date Range', df.date.min(), ' - ', df.date.max())
#print('Test Date Range', test_df.date.min(), ' - ',  test_df.date.max())


# In[6]:


print('Data Shape', df.shape, 'From:', df.date.min(), 'To:', df.date.max(), 'Duration:', df.date.max() - df.date.min())
#print('Trest Data Shape', test_df.shape, 'From:', test_df.date.min(), 'To:', test_df.date.max(), 'Duration:', test_df.date.max() - test_df.date.min())


# In[7]:


DAYS_LOOK_BACK = 365
DAYS_PREDICT_FORWARD = 90

#Features are just going to get the median, max, min, values
DATE_COLUMNS = [
    'day', #removing features to cut back on memory
    #consider making one hot if performance drops
]

ONE_HOT_COLUMNS = [
    'dayofweek',
    'month',
    #'hour', #removing features to cut back on memory
    #'weekofyear', 
    #'year' always the same....
    'channelGrouping',
    #'device_browser', #Removed: Too Many Features
    'device_deviceCategory',
    'device_isMobile',
#    'geoNetwork_country', 
#    'trafficSource_adwordsClickInfo.page', #these fell to the bottom of feature importances, sum and mean... just ditiching
    #'trafficSource_adwordsClickInfo.isVideoAd', #Removed: All False All True In Fold
    'trafficSource_isTrueDirect', #Removed: All True In Fold
]

NUMERIC_FEAT_COLUMNS = [
    'totals_hits',
    'totals_pageviews',
    'totals_timeOnSite',
    'totals_totalTransactionRevenue.div1M',
    'totals_totalTransactionRevenue.log1p', #Added feature and remove native values so that you can convert fetures to int32 
    'totals_transactions'
]

# for these columns will just choose the most frequently occuring one by user....
LABEL_ENCODE_COLUMNS = [
    'geoNetwork_country',
    'geoNetwork_subContinent',
    'device_operatingSystem'
]


# In[8]:


from sklearn.preprocessing import LabelEncoder

def get_label_encoded_features(df):
    tables = []
    le  = LabelEncoder()
    for column in LABEL_ENCODE_COLUMNS:
        encoded_labels = le.fit_transform(df[column])
        tables.append( pd.DataFrame({(column + '.encoded'): encoded_labels}).set_index(df.index) )
    return pd.concat(tables, axis=1)


# In[9]:


def get_one_hot_features(df):
    """
    One hot encode categorical features...
    """
    tables = []
    for col in ONE_HOT_COLUMNS:
        tables.append( pd.get_dummies(df[col].fillna('NA')).add_prefix(col + '.') )
    return pd.concat(tables,axis=1)

# def get_date_columns(df):
#     tables = []
#     for col in DATE_COLUMNS:
#         tables.append( pd.get_dummies(df[col].fillna('NA')).add_prefix(col + '.') )
#     return pd.concat(tables,axis=1)

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def get_rececency(df, end_of_training_window_date, skip_quantile_stats=False):
    df['session_recency'] = (end_of_training_window_date - df['date']).dt.days
    
    # These Stats are pretty slow to calculate, increaese run time 10x, all of the sorting??---
    
    quantiles_stats = []
    if not skip_quantile_stats:
        quantiles_stats = ['median', 'skew', percentile(25), percentile(75)]
    
    recency = df.groupby('fullVisitorId')['session_recency']             .agg(['min', 'max', 'mean', 'std'] + quantiles_stats)             .add_prefix('session_recency_')
    recency['session_recency_diff'] = recency['session_recency_max'] - recency['session_recency_min']
    
    if not skip_quantile_stats:
        recency['session_recency_iqr'] = recency['session_recency_percentile_75'] - recency['session_recency_percentile_25']
    
    return recency


def add_calculted_features(df):
    df['totals_totalTransactionRevenue.log1p'] = np.log1p(df['totals_totalTransactionRevenue'].values)
    df['totals_totalTransactionRevenue.div1M'] = df['totals_totalTransactionRevenue'].values/(10**6)
    return df


# In[10]:


def feat_targets(df, split_date, lookback_window=DAYS_LOOK_BACK, target_fwd_window=DAYS_PREDICT_FORWARD, skip_quantile_stats=False):
    """
    skip_quantile_stats: whether or not to include non-parametric stats, these have a pretty poor perforamnce on the overall dataset, assumedly because of the need to repeatedly sort?
    with the stats include it takes ~6Min per run, without it takes 40 seconds
    """
    target_col = 'totals_totalTransactionRevenue'
    train_start_date = split_date + pd.Timedelta(days=-lookback_window)
    target_end_date = split_date + pd.Timedelta(days=+target_fwd_window)
    print('Date Range of Dataset', df.date.min(), df.date.max())
    print('lookback_window', lookback_window, 'target_fwd_window', target_fwd_window)
    print('train_start_date', train_start_date)
    print('split_date', split_date)
    print('target_end_date', target_end_date)
    print()
    if (train_start_date < df.date.min()) or (target_end_date > df.date.max()):
        raise ValueError('Periods are outside of dataframe time range')
    fold_train = df[(df.date >= train_start_date) & (df.date < split_date)]
    print('train at sessions level shape', fold_train.shape)
    #print('removing duplicate sessions')
    
    fold_val = df[(df.date >= split_date) & (df.date <= target_end_date)]
    fold_val_target = fold_val.groupby('fullVisitorId')[target_col].sum().to_frame()
    print('val agg by user shape', fold_val_target.shape)
    del fold_val
    gc.collect()
    
    print('Encoding session level features')
    print('adding calculated features')
    fold_train = add_calculted_features(fold_train)
    print('one_hot_features')
    one_hot_features = get_one_hot_features(fold_train)
    print('label_encoded_features')
    label_encoded_features = get_label_encoded_features(fold_train)
    
    print('creating session level features')
    # get session level features
    session_x = pd.concat([
        fold_train[['fullVisitorId'] + NUMERIC_FEAT_COLUMNS + DATE_COLUMNS],
#         date_features, 
         one_hot_features, 
         label_encoded_features
        ], axis=1, sort=True)
    print('session_x', session_x.shape)
    
    sum_cols = one_hot_features.columns.tolist() + NUMERIC_FEAT_COLUMNS
    mean_cols = one_hot_features.columns.tolist() + NUMERIC_FEAT_COLUMNS
    min_cols = NUMERIC_FEAT_COLUMNS + DATE_COLUMNS
    max_cols = NUMERIC_FEAT_COLUMNS + DATE_COLUMNS
    std_cols = NUMERIC_FEAT_COLUMNS
    skew_cols = NUMERIC_FEAT_COLUMNS
    median_cols = label_encoded_features.columns.tolist() + DATE_COLUMNS #these should be the same for all users

    print('aggregating session level features to user level')
    print('calcing recency stats')
    recency = get_rececency(fold_train, split_date, skip_quantile_stats=skip_quantile_stats) #done to calculate recency stats
    print('finished recency')
    
    quantile_cols = []
    if not skip_quantile_stats:
        quantile_cols = [
            session_x.groupby('fullVisitorId')[median_cols].median().add_suffix('_median'),
            session_x.groupby('fullVisitorId')[skew_cols].skew().add_suffix('_skew'), #this made performance notably worse, for median/skew/percentile, it has to sort so O(n*log(n)

        ]
    #aggregate session features by user
    train_x = pd.concat([
        
        session_x['fullVisitorId'].value_counts().to_frame(name='session_count'), 
        #get_rececency(fold_train, split_date), #done to calculate recency stats
        recency,
        session_x.groupby('fullVisitorId')[sum_cols].sum().add_suffix('_sum'), #this will handle frequency/monetary vaue
        session_x.groupby('fullVisitorId')[mean_cols].mean().add_suffix('_mean'),
        session_x.groupby('fullVisitorId')[min_cols].max().add_suffix('_min'),
        session_x.groupby('fullVisitorId')[max_cols].max().add_suffix('_max'),
        session_x.groupby('fullVisitorId')[std_cols].std().add_suffix('_std'),
    ] + quantile_cols , axis = 1, sort=True) \
        .fillna(0) \
        .astype('int32') #this had a big effect on memory!!
    del session_x, one_hot_features, label_encoded_features
    gc.collect()
    
    print('getting target values')
    # get target for each user from fold_val, left join on a series from the train dataset to get all users in train and any target from fold_val
    merged=train_x['session_count'].to_frame().join(fold_val_target, how='left')
    train_y = merged[target_col].to_frame(name = 'target_revenue')
    train_y['is_returning'] = train_y.target_revenue.notna()
    train_y.fillna(0, inplace=True)
    
    print('Output shapes', 'X', train_x.shape, 'y', train_y.shape)
    gc.collect()
    return train_x, train_y


# In[11]:


get_ipython().run_cell_magic('time', '', "train_X, train_y = feat_targets(df, split_date=pd.Timestamp('2017-09-30'), skip_quantile_stats=True)")


# In[12]:


get_ipython().run_cell_magic('time', '', "val_X, val_y = feat_targets(df, split_date=pd.Timestamp('2017-12-31'), skip_quantile_stats=True)")


# In[13]:


get_ipython().run_cell_magic('time', '', "# Test Set starts on first date of test period\ntest_X, test_y = feat_targets(df, split_date=pd.Timestamp('2018-04-01'), skip_quantile_stats=True)")


# In[14]:


#TODO: correct for non-overlapping features, there are a handful of features in val not in train and vice versa
feature_overlap = sorted(list((set(train_X.columns).intersection(val_X.columns)).intersection(test_X.columns)))
val_X = val_X[feature_overlap]
train_X = train_X[feature_overlap]
test_X = test_X[feature_overlap]


# In[15]:


print('Val Baseline all zeros', mean_squared_error(np.log1p(val_y.target_revenue.values), np.zeros_like(val_y.target_revenue.values))**0.5)
print('Test Baseline all zeros', mean_squared_error(np.log1p(test_y.target_revenue.values), np.zeros_like(test_y.target_revenue.values))**0.5)

print('Val % non-zero', (val_y.target_revenue.values > 0).mean())
print('Test % non-zero', (test_y.target_revenue.values > 0).mean())


# In[16]:


#import lightgbm as lgb
# setting taken from here: https://www.kaggle.com/augustmarvel/base-model-v2-user-level-solution
from xgboost import XGBRegressor
xgb_params = {
        'objective': 'reg:squarederror',
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

xgb = XGBRegressor(**xgb_params, n_estimators=1500) #n_estimators determines number of rounds
xgb.fit(train_X, np.log1p(train_y.target_revenue.values),eval_set=[
    (train_X, np.log1p(train_y.target_revenue.values)),
    (val_X, np.log1p(val_y.target_revenue.values)),
],early_stopping_rounds=25,eval_metric='rmse',verbose=25)


# In[17]:


results = xgb.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')


# In[18]:


preds = xgb.predict(val_X)
print('Val RMSE From model', mean_squared_error(np.log1p(val_y.target_revenue.values), preds)**0.5)


# In[19]:


print('TEst RMSE From model', mean_squared_error(np.log1p(test_y.target_revenue.values), xgb.predict(test_X))**0.5)


# In[20]:


xgb2 = XGBRegressor(**xgb_params, n_estimators=xgb.best_iteration)
xgb2.fit(val_X, np.log1p(val_y.target_revenue.values), eval_set=[
    (val_X, np.log1p(val_y.target_revenue.values))
], eval_metric='rmse',verbose=25)


# In[21]:


results = xgb2.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
#ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')


# In[22]:


print('Test RMSE From model', mean_squared_error(np.log1p(test_y.target_revenue.values), xgb2.predict(test_X))**0.5)


# In[ ]:




