#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import datetime as dt
import gc
# from dateutil.relativedelta import relativedelta
import time
# from tqdm import tqdm
# from scipy import stats
from statsmodels.stats import proportion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from bayes_opt import BayesianOptimization
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error


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


# def percentile99(x):
#     return x[x<=np.percentile(x, 99)]
def get_mode(x):
    try:
        return int(x.mode())
    except:# in cases, when threre are only two values - just pick the second one
        return x.values[0]

def groupby(df, groupby_clmns, agg_clmns, agg_func, rel_calc=False, multigroup=False):
    if not agg_clmns:
        groupby = df.groupby(groupby_clmns).agg(agg_func).reset_index()
    else:
        groupby = df.groupby(groupby_clmns)[agg_clmns].agg(agg_func).reset_index()
    
    if multigroup:
        groupby1 = groupby.groupby(groupby_clmns[0])[list(list(zip(*agg_func))[0])].agg(sum).reset_index()
        groupby1.rename(columns={key[0]: key[0]+'sum' for key in agg_func}, inplace=True)
        groupby = merge_two_df(groupby, groupby1, groupby_clmns[0], groupby_clmns[0])
#         print(groupby.head())
        for agg_func_clmn in agg_func:
            groupby[agg_func_clmn[0]] = groupby[agg_func_clmn[0]]/groupby[agg_func_clmn[0]+'sum']
#         print(groupby.head())
        groupby = groupby[list(groupby_clmns) + list(list(zip(*agg_func))[0])]
        
    if rel_calc:
        for agg_func_clmn in agg_func:
            groupby[agg_func_clmn[0]] = groupby[agg_func_clmn[0]]/groupby[agg_func_clmn[0]].sum()
            
    return groupby


# def auth_mean_calculation(df_hist_trans):
#     def bern_conf_interv(vls, true_p=0.91354, mth='agresti_coull'):#possible methods = ['normal', 'agresti_coull', 'beta', 'wilson', 'jeffreys', 'binom_test']
#         positive = np.sum(vls)
#         n = len(vls)
#         if n<10:
#             interval = proportion.proportion_confint(positive, n, method=mth)
#             return interval[0] + true_p * (interval[1] - interval[0])
#         else:
#             return np.mean(vls)

#     gr = df_hist_trans.groupby('card_id')['authorized_flag'].agg(bern_conf_interv).reset_index()
#     return gr
    

def merge_two_df(df1, df2, left_on, right_on):
    return pd.merge(df1, df2, how='left', left_on=left_on, right_on=right_on)

# def save_df(filename='../data/new_hist.csv', drop_clmns=['date_min_dif', 'purchase_date_month_label']):
# #     df_hist_trans.drop(['date_min_dif', 'purchase_date_month_label'], axis=1)
#     with open(filename, 'w') as f:
#         if not drop_clmns:
#             df_hist_trans.to_csv(f, index=False, header=True)
#         else:
#             df_hist_trans.drop(drop_clmns, axis=1).to_csv(f, index=False, header=True)


# In[4]:


# for me, the loading takes about 50 sec
df_hist_trans = pd.read_csv("../data/historical_transactions.csv")

# make a little preprocessing
df_hist_trans['authorized_flag'] = df_hist_trans['authorized_flag'].map({'Y': 1, 'N': 0})
df_hist_trans['purchase_date'] = pd.to_datetime(df_hist_trans['purchase_date'])
df_hist_trans['category_1'] = df_hist_trans['category_1'].map({'Y': 1, 'N': 0})
df_hist_trans['category_2'].fillna(-1, inplace=True)
df_hist_trans['category_3'].fillna('Z', inplace=True)
df_hist_trans['category_3'] = df_hist_trans['category_3'].map({'A': 0, 'B': 1, 'C': 2, 'Z': -1})
df_hist_trans['date_min_dif'] = (df_hist_trans['purchase_date'] - df_hist_trans['purchase_date'].min()).dt.days

# Here, I delete the first five symbols from card_id and merchant_id, because they are the same and in order to save a little bit memory.
# Do it for all *.csv files - train, test, merchants, etc. And before making a submit I just add that five symbols to card_id submit file.

df_hist_trans['card_id'] = df_hist_trans['card_id'].str[5:]
df_hist_trans['merchant_id'] = df_hist_trans['merchant_id'].str[5:]

df_new_merchant_hist_trans = reduce_mem_usage(df_hist_trans)

gc.collect();
# df_hist_trans.head()


# In[5]:


# If we check the data after some preprocessing steps, we can notice that there are 138481 NaN merchant_id. 
# Also, there are some NaN merchant_id in new_merchant_transactions.csv
# In such a way we can do one of the following things:
# - delete these rows. But that would cause to the loss of information.
# - replace with some new specific value - '000000000' for example. But in that case, we would not be able to use information from merchants.csv
# - replace with the most familier merhcant_id. For me, that case looks like a little bit more apropriate than the previous one.

df_hist_trans.isnull().sum()


# In[6]:


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def merchant_id_impute(df):
    agg_func = {
        'purchase_amount': np.median,
        'category_3': lambda x: get_mode(x),
        'city_id': lambda x: get_mode(x),
        'state_id': lambda x: get_mode(x),
        'subsector_id': lambda x: get_mode(x), 
        'merchant_category_id': lambda x: get_mode(x),
        'date_min_dif': np.mean,
    }
    temp = df[~df['merchant_id'].isnull()].groupby(['merchant_id']).agg(agg_func).reset_index()
    
    scaler = StandardScaler()
    
    #build and fit 1-NearestNeighbors model
    clmns = ['subsector_id', 'merchant_category_id',
           'purchase_amount', 'date_min_dif', 
           'category_3', 'city_id', 'state_id']
    knnImputer = NearestNeighbors(n_neighbors=1)
    knnImputer.fit(scaler.fit_transform(temp[clmns].values))

    # predict for NaN merchants
    # knnImputer.kneighbors returns distance to the nearest neighbour and index of the nearest neighbour. We need only index
    index = knnImputer.kneighbors(scaler.transform(df[df['merchant_id'].isnull()][clmns].values))[1]
    
    #impute NaN merchants with predicted nearest neighbours
    df.loc[df['merchant_id'].isnull(), 'merchant_id'] = temp.iloc[index.reshape(-1)]['merchant_id'].values


# In[7]:


gc.collect()


# In[8]:


merchant_id_impute(df_hist_trans)


# In[9]:


# right after merhcnat_id NaN imputation we can notice that there are not NaN values at all
df_hist_trans.isnull().sum()


# In[10]:


df_new_merchant_hist_trans = pd.read_csv("../data/new_merchant_transactions.csv")
# In new_merchant_transactions.csv all transactions are authorized.

# Some default prerocessing
df_new_merchant_hist_trans['purchase_date'] = pd.to_datetime(df_new_merchant_hist_trans['purchase_date'])
df_new_merchant_hist_trans['category_1'] = df_new_merchant_hist_trans['category_1'].map({'Y': 1, 'N': 0})
df_new_merchant_hist_trans['category_2'].fillna(-1, inplace=True)
df_new_merchant_hist_trans['category_3'].fillna('Z', inplace=True)
df_new_merchant_hist_trans['category_3'] = df_new_merchant_hist_trans['category_3'].map({'A': 0, 'B': 1, 'C': 2, 'Z': -1})
df_new_merchant_hist_trans['date_min_dif'] = (df_new_merchant_hist_trans['purchase_date'] - df_new_merchant_hist_trans['purchase_date'].min()).dt.days

df_new_merchant_hist_trans['card_id'] = df_new_merchant_hist_trans['card_id'].str[5:]
df_new_merchant_hist_trans['merchant_id'] = df_new_merchant_hist_trans['merchant_id'].str[5:]

# i think that we can even drop authrized column from that dataframe, because the values is constant
df_new_merchant_hist_trans.drop('authorized_flag', axis=1, inplace=True)

# impute NaN merchant_id
merchant_id_impute(df_new_merchant_hist_trans)

df_new_merchant_hist_trans = reduce_mem_usage(df_new_merchant_hist_trans)
gc.collect();


# In[11]:


def fix_first_active_date(df1, df2):
    # Since some card_id 'first_active_date' featues are later than coresponding for that card_id purchase_date,
    # we can replace that first_active_dates with the coresponding for that card_id min purchase_date.
    # In such a way we fix that kinda "noise" or data mismatches
    ddd = merge_two_df(df1.groupby('card_id')['purchase_date'].min().reset_index(), df2[['card_id', 'first_active_date']], ['card_id'], ['card_id'])
    ddd.dropna(inplace=True)
    ddd['fault'] = ddd['purchase_date']<ddd['first_active_date']
    ddd.loc[ddd['fault'], 'first_active_date'] = ddd.loc[ddd['fault'], 'purchase_date'].values
    df2 = ddd.loc[ddd['fault'], ['card_id', 'first_active_date']].set_index('card_id').combine_first(df2.set_index('card_id')).reset_index()
    return df2

def cross_cat_features_calc(df, dct={}):
    # this function encodes combination of feature_1, ..._2, ..._3, but actualy that feature doesn't bring anything usefull,
    # but probably you would be able to extract some usefull information for your regression model and i just decided to leave it.
    cnt = 0
    lst = []
    for val in df[['feature_1', 'feature_2', 'feature_3']].values:
        key = tuple(val)
        if dct.get(key, ''):
            lst.append(dct.get(key))
            continue
        dct[key] = cnt
        cnt += 1
        lst.append(dct.get(key))
    return lst, dct

def load_df(df_history, isTrain=1, dct={}):
    if isTrain:
        df = reduce_mem_usage(pd.read_csv('../data/train.csv'))
        df['outlier'] = (df['target']<-19)*1
    else:
        df = reduce_mem_usage(pd.read_csv('../data/test.csv'))
        
    df['card_id'] = df['card_id'].str[5:]
    # There is only one card_id 'c27b4f80f7' with NaN first_active_month in test. 
    # Assign to this card_id min date year-moth from history_transaction.csv
    for card_id in df[df['first_active_month'].isnull()]['card_id'].values:
        mindate = df_history[df_history['card_id']==card_id]['purchase_date'].min()
        df['first_active_month'].fillna('-'.join(str(mindate).split('-')[:2]), inplace=True)#'2017-03'
        
    df['first_active_date'] = pd.to_datetime(df['first_active_month'])
    df = fix_first_active_date(df_history, df)
    df['first_active_date_elapsed_day'] = (dt.datetime(2018, 3, 1) - df['first_active_date']).dt.days # погрешность в 1 месяц - может купить в конце месяца, а может в начале
    # train['first_active_date_month'] = train['first_active_date'].dt.month
    # train['first_active_date_year'] = train['first_active_date'].dt.year

    df['cross_cat_features'], dct = cross_cat_features_calc(df, dct)
    return df, dct


# In[12]:


train, dct = load_df(df_hist_trans)
test, dct = load_df(df_hist_trans, isTrain=0, dct=dct)

# train.feature_2 = train.feature_2.map({1: 'A', 2: 'B', 3: 'C'})
train.head()


# In[13]:


merch = pd.read_csv('../data/merchants.csv')
merch['merchant_id'] = merch['merchant_id'].str[5:]
merch['category_1'] = merch['category_1'].map({'Y': 1, 'N': 0})
merch['category_4'] = merch['category_4'].map({'Y': 1, 'N': 0})
merch['most_recent_sales_range'] = merch['most_recent_sales_range'].map({'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0})
merch['most_recent_purchases_range'] = merch['most_recent_purchases_range'].map({'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0})
merch.drop_duplicates('merchant_id', inplace=True)# there are about 63 duplcated merchants but with some a little bit different features
merch = reduce_mem_usage(merch)
merch.head()


# In[14]:


# if we look at the min value of 'first_active_date_elapsed_day', then we can conclude that outliers are only the card with first_active_date_elapsed_day>150
train[train['outlier']==1]['first_active_date_elapsed_day'].describe()


# In[15]:


print(f"quantity of non_outliers in train based on train['first_active_date_elapsed_day']<=150 condition: {train[train['first_active_date_elapsed_day']<=150].shape[0]},\nquantity of non_outliers in test based on test['first_active_date_elapsed_day']<=150 condition: {test[test['first_active_date_elapsed_day']<=150].shape[0]}")


# In[16]:


df_hist_trans = merge_two_df(df_hist_trans, train[['card_id', 'outlier']], ['card_id'], ['card_id'])

_ = df_hist_trans[df_hist_trans['card_id'].isin(train['card_id'])].groupby(['card_id', 'merchant_id'])['outlier'].max().reset_index()
_ = merge_two_df(_, _.groupby('merchant_id')['outlier'].mean().reset_index().rename(columns={'outlier': 'outlier_merch_mean'}), ['merchant_id'], ['merchant_id'])
train = merge_two_df(train, _.groupby('card_id')['outlier_merch_mean'].mean().reset_index(), ['card_id'], ['card_id'])
del _
gc.collect()
train.corr()['outlier']


# In[17]:


# purchase_amount_rel - purchase_amount comparing with the other purchases in the same merchant_category.
# That feature is scaled in range of (0; 1];
# On my personal opinion that feature helps to avoid some outliers in original purchase_amount.
# That feature has very very little impact on the outlier prediction, but nonethelss i used it and probably that feature can bring a little LB improvement in predicting the target with regression
# Calculation of that feature takes about 2100 seconds
# The higher 'subshape' values - the more precise and accurate 'purchase_amount_rel' values
# If you get Memory Error - decrease 'subshape' value. That would also speed up calculation and cause to rougher values. 
# I have 16Gb RAM memory and subshape = 10000 for me is optimal - relativly fast and precise.
df_hist_trans['purchase_amount_rel'] = 0
df_hist_trans['merchant_category_id_count'] = 0
subshape = 10000
tm = time.time()

for glbl_cnt, merchant_category_id in enumerate(df_hist_trans['merchant_category_id'].unique()):#326 categories
    indx = df_hist_trans[df_hist_trans['merchant_category_id']==merchant_category_id].index.values
    vls = df_hist_trans.loc[indx, 'purchase_amount'].values
    shape = indx.shape[0]
    if shape>subshape:
        subvls = vls[np.random.choice(len(vls), size=subshape, replace=False)]
    df_hist_trans.loc[indx, 'merchant_category_id_count'] = shape
    print('counter:', glbl_cnt, 'merchant_category_id:', merchant_category_id, 'shape of merch_category:', shape, 'number of chunks:', max(shape//subshape, 1))
    cnt = 0
    for lcl_vls in np.array_split(vls, max(shape//subshape, 1)):
        if shape>subshape:
            df_hist_trans.loc[indx[cnt : cnt + lcl_vls.shape[0]], 'purchase_amount_rel'] = (lcl_vls.reshape(-1, 1)>=subvls).sum(axis=1)/subshape
        else:
            df_hist_trans.loc[indx[cnt : cnt + lcl_vls.shape[0]], 'purchase_amount_rel'] = (lcl_vls.reshape(-1, 1)>=vls).sum(axis=1)/shape
        cnt += lcl_vls.shape[0]
    
print(time.time()-tm)
gc.collect()


# In[18]:


def extract_date_features(df_hist_trans, train_df, test_df):
    tm = time.time()
    agg_func = [
        ('merchant_category_id_last_visited_ref_date_day_diff', lambda x: (dt.datetime(2018, 3, 1)-x.max()).days),
    ]
#     df = groupby(df_hist_trans, ['merchant_category_id', 'card_id'], 'purchase_date', agg_func)
    df = df_hist_trans.groupby(['merchant_category_id', 'card_id'])['purchase_date'].agg(agg_func).reset_index()
    print('merchant_category_id_last_visited_ref_date_day_diff passed', time.time()-tm)
    
    df = merge_two_df(df, train_df[['card_id', 'first_active_date_elapsed_day']], ['card_id'], ['card_id'])
    df = merge_two_df(df, test_df[['card_id', 'first_active_date_elapsed_day']], ['card_id'], ['card_id'])
    df.loc[df[(df['first_active_date_elapsed_day_x'].isnull()) & (~df['first_active_date_elapsed_day_y'].isnull())].index, 'first_active_date_elapsed_day_x'] =         df.loc[df[(df['first_active_date_elapsed_day_x'].isnull()) & (~df['first_active_date_elapsed_day_y'].isnull())].index, 'first_active_date_elapsed_day_y']
    df.drop('first_active_date_elapsed_day_y', axis=1, inplace=True)
    df.rename(columns={'first_active_date_elapsed_day_x': 'first_active_date_elapsed_day'}, inplace=True)
    agg_func = [
        ('last_visited_ref_date_day_diff', lambda x: (dt.datetime(2018, 3, 1)-x.max()).days)
    ]
#     last_visited_ref_date_day_diff = groupby(df_hist_trans, ['card_id'], 'purchase_date', agg_func)
    last_visited_ref_date_day_diff = df_hist_trans.groupby(['card_id'])['purchase_date'].agg(agg_func).reset_index()
    print('last_visited_ref_date_day_diff passed', time.time()-tm)
    df = merge_two_df(df, last_visited_ref_date_day_diff, ['card_id'], ['card_id'])
    
    df['last_vis_merch_last_vis_day_diff'] = df['merchant_category_id_last_visited_ref_date_day_diff'] - df['last_visited_ref_date_day_diff']
    return df


# In[19]:


#that takes about 700 seconds for me
temp = extract_date_features(df_hist_trans, train, test)
df_hist_trans = merge_two_df(df_hist_trans, temp, ['merchant_category_id', 'card_id'], ['merchant_category_id', 'card_id'])
del temp
gc.collect()


# In[20]:


lastShape = df_hist_trans.shape
# leave only 'first_active_date_elapsed_day'>150 card_id transactions
# cond = (df_hist_trans['card_id'].isin(train[train['first_active_date_elapsed_day']>150]['card_id'])) | \
#         (df_hist_trans['card_id'].isin(test[test['first_active_date_elapsed_day']>150]['card_id']))
# df_hist_trans = df_hist_trans[cond].reset_index(drop=True)#.copy()
# df_hist_trans['isTrain'] = (df_hist_trans['card_id'].isin(train[train['first_active_date_elapsed_day']>150]['card_id']))*1
df_hist_trans['isTrain'] = (df_hist_trans['card_id'].isin(train['card_id']))*1

# add from train and test 'first_active_date_elapsed_day' features into our transaction dataframe
df_hist_trans = merge_two_df(df_hist_trans, train[['card_id', 'first_active_date_elapsed_day']], ['card_id'], ['card_id'])
df_hist_trans = merge_two_df(df_hist_trans, test[['card_id', 'first_active_date_elapsed_day']], ['card_id'], ['card_id'])
df_hist_trans.loc[df_hist_trans['first_active_date_elapsed_day_x'].isnull(), 'first_active_date_elapsed_day_x'] = df_hist_trans.loc[df_hist_trans['first_active_date_elapsed_day_x'].isnull(), 'first_active_date_elapsed_day_y'].values
df_hist_trans.drop('first_active_date_elapsed_day_y', axis=1, inplace=True)
df_hist_trans.rename(columns={'first_active_date_elapsed_day_x': 'first_active_date_elapsed_day'}, inplace=True)
# fill NaN outliers from test with default value
df_hist_trans['outlier'].fillna(-1, inplace=True)
gc.collect()
print(lastShape, '->', df_hist_trans.shape)


# In[21]:


#save if right now your memory usage is pretty high in order to prevent recalculation in case of memory overflow during further calculation
with open('../data/temp_df_transaction_history.csv', 'w') as f:
    df_hist_trans.to_csv(f, header=True, index=False)


# In[22]:


df_hist_trans = pd.read_csv("../data/temp_df_transaction_history.csv")

# make a little preprocessing
df_hist_trans['purchase_date'] = pd.to_datetime(df_hist_trans['purchase_date'])
df_hist_trans = reduce_mem_usage(df_hist_trans)


# In[23]:


# for me it took about 9100 seconds
agg_func = {
    'outlier': max,
    'isTrain': max,
    'purchase_amount': np.median,
    'purchase_amount_rel': np.mean,
    'merchant_category_id_count': len,
    'city_id': lambda x: get_mode(x),
    'state_id': lambda x: get_mode(x),
    'installments': lambda x: get_mode(x),
    'category_3': lambda x: get_mode(x),
    'merchant_category_id': lambda x: get_mode(x),
    'subsector_id': lambda x: get_mode(x),
    'merchant_category_id_last_visited_ref_date_day_diff': min,
    'last_visited_ref_date_day_diff': min,
    'last_vis_merch_last_vis_day_diff': min,
    'first_active_date_elapsed_day': max,
    'purchase_date': lambda x: (dt.datetime(2018, 3, 1) - x.max()).days, 
}

tm = time.time()
ttt = df_hist_trans[df_hist_trans['first_active_date_elapsed_day']>150].groupby(['card_id', 'merchant_id']).agg(agg_func).reset_index()
ttt.rename(columns={'merchant_category_id_count': 'trans_count'}, inplace=True)
print(time.time()-tm)


# In[24]:


ttt.shape


# In[25]:


# defining only test merchant_id
onlyTestMerch = list(set(ttt[ttt['card_id'].isin(test[test['first_active_date_elapsed_day']>150]['card_id'])]['merchant_id'].unique()) -                       set(ttt[ttt['card_id'].isin(train[train['first_active_date_elapsed_day']>150]['card_id'])]['merchant_id'].unique()))
# defining only train merchant_id
onlyTrainMerch = list(set(ttt[ttt['card_id'].isin(train[train['first_active_date_elapsed_day']>150]['card_id'])]['merchant_id'].unique()) -                       set(ttt[ttt['card_id'].isin(test[test['first_active_date_elapsed_day']>150]['card_id'])]['merchant_id'].unique()))


# In[26]:


old_merch = merch[merch['merchant_id'].isin(ttt['merchant_id'].unique())][['merchant_id', 'merchant_category_id',
       'subsector_id', 'numerical_1', 
       'most_recent_sales_range', 'most_recent_purchases_range',
       'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12',
       'category_4', 'city_id', 'state_id']].reset_index(drop=True)


# In[27]:


old_merch['TestOrTrain'] = 0
old_merch.loc[old_merch['merchant_id'].isin(onlyTestMerch), 'TestOrTrain'] = -1
old_merch.loc[old_merch['merchant_id'].isin(onlyTrainMerch), 'TestOrTrain'] = 1


# In[28]:


old_merch['TestOrTrain'].value_counts()


# In[29]:


old_merch = merge_two_df(old_merch, ttt.groupby('merchant_id')['card_id'].nunique().reset_index().rename(columns={'card_id': 'merch_card_id_nunique'}), ['merchant_id'], ['merchant_id'])
old_merch = merge_two_df(old_merch, ttt[ttt['isTrain']==1].groupby('merchant_id')['outlier'].mean().reset_index().rename(columns={'outlier': 'merch_outlier_mean'}), ['merchant_id'], ['merchant_id'])
old_merch['merch_outlier_mean'] = old_merch['merch_outlier_mean'].fillna(-1)
old_merch['merch_card_id_nunique'] = old_merch['merch_card_id_nunique'].fillna(-1)


# In[30]:


gc.collect()


# In[31]:


# define condition for picking only potential outlier card_id - 'first_active_date_elapsed_day'>150
condition  = df_hist_trans['first_active_date_elapsed_day']>150
# define native city_id and state_id - city and state with the most frequent number of purchases
native_city_state = df_hist_trans[condition].groupby(['card_id'])[['city_id', 'state_id']].agg(lambda x: mode_func(x)).reset_index().rename(columns={'city_id': 'native_city', 'state_id': 'native_state'})
ttt = merge_two_df(ttt, native_city_state, ['card_id'], ['card_id'])
del native_city_state;

# quantity of uniquely visited 'city_id' and 'state_id' for each card_id
ttt = merge_two_df(ttt, df_hist_trans[condition].groupby('card_id')[['city_id', 'state_id']].nunique().reset_index().rename(columns={'city_id': 'card_city_nunique', 'state_id': 'card_state_nunique'}), ['card_id'], ['card_id'])
# quantity of uniquely visited 'city_id' for each card_id per state_id
ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['card_id', 'state_id'])[['city_id']].nunique().reset_index().rename(columns={'city_id': 'card_state_city_nunique'}), ['card_id', 'state_id'], ['card_id', 'state_id'])

# quantity of unique merchant in a specific merchant_category_id in a specific city_id
ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['city_id', 'merchant_category_id'])['merchant_id'].nunique().reset_index().rename(columns={'merchant_id': 'native_city_merch_cat_merch_nunique'}), ['native_city', 'merchant_category_id'], ['city_id', 'merchant_category_id'])
# quantity of unique merchant in a specific merchant_category_id in a specific state_id
ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['state_id', 'merchant_category_id'])['merchant_id'].nunique().reset_index().rename(columns={'merchant_id': 'native_state_merch_cat_merch_nunique'}), ['native_state', 'merchant_category_id'], ['state_id', 'merchant_category_id'])
ttt.drop(['city_id_y', 'state_id_y'], axis=1, inplace=True)
ttt.rename(columns={'city_id_x': 'city_id', 'state_id_x': 'state_id'}, inplace=True)

# quantity of unique merchant in a specific subsector_id in a specific city_id
ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['city_id', 'subsector_id'])['merchant_id'].nunique().reset_index().rename(columns={'merchant_id': 'native_city_subsector_merch_nunique'}), ['native_city', 'subsector_id'], ['city_id', 'subsector_id'])
# quantity of unique merchant in a specific subsector_id in a specific state_id
ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['state_id', 'subsector_id'])['merchant_id'].nunique().reset_index().rename(columns={'merchant_id': 'native_state_subsector_merch_nunique'}), ['native_state', 'subsector_id'], ['state_id', 'subsector_id'])
ttt.drop(['city_id_y', 'state_id_y'], axis=1, inplace=True)
ttt.rename(columns={'city_id_x': 'city_id', 'state_id_x': 'state_id'}, inplace=True)

# number of unique cities visited in a specific merchant_category_id for each card_id
ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['card_id', 'merchant_category_id'])['city_id'].nunique().reset_index().rename(columns={'city_id': 'card_merch_cat_city_nunique'}), ['card_id', 'merchant_category_id'], ['card_id', 'merchant_category_id'])
# number of unique cities visited in a specific merchant_category_id for each card_id in the native state
ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['card_id', 'state_id', 'merchant_category_id'])['city_id'].nunique().reset_index().rename(columns={'city_id': 'card_state_merch_cat_city_nunique'}), ['card_id', 'native_state', 'merchant_category_id'], ['card_id', 'state_id', 'merchant_category_id'])
ttt.drop(['state_id_y'], axis=1, inplace=True)
ttt.rename(columns={'state_id_x': 'state_id'}, inplace=True)

ttt['isNative_city'] = (ttt['city_id'] == ttt['native_city']) * 1
ttt['isNative_state'] = (ttt['state_id'] == ttt['native_state']) * 1

# tm = time.time()
# days difference since the last visit for a specific merch for earch card_id
# for me it takes about 1100 seconds
agg_func = [
    ('merch_card_last_date_diff', lambda x: (dt.datetime(2018, 3, 1) - x.max()).days)
]

ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['card_id', 'merchant_id'])['purchase_date'].agg(agg_func).reset_index(), ['card_id', 'merchant_id'], ['card_id', 'merchant_id'])
# print(time.time()-tm)

agg_func = [
    ('merch_last_date_diff_min', min),
    ('merch_last_date_diff_max', max),
    ('merch_last_date_diff_mean', np.mean),
    ('merch_last_date_diff_median', np.median),
]

old_merch = merge_two_df(old_merch, ttt.groupby(['merchant_id'])['merch_card_last_date_diff'].agg(agg_func).reset_index(), ['merchant_id'], ['merchant_id'])

agg_func = [
    ('merchant_category_id_card_id_purchase_amount_rel_mean', np.mean),
    ('merchant_category_id_card_id_purchase_amount_rel_median', np.median),
    ('merchant_category_id_card_id_purchase_amount_count', len),
]

ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['merchant_category_id', 'card_id'])['purchase_amount_rel'].agg(agg_func).reset_index(), ['merchant_category_id', 'card_id'], ['merchant_category_id', 'card_id'])

agg_func = [
    ('merchant_category_id_card_id_purchase_amount_abs_mean', np.mean),
    ('merchant_category_id_card_id_purchase_amount_abs_median', np.median),
]

ttt = merge_two_df(ttt, df_hist_trans[condition].groupby(['merchant_category_id', 'card_id'])['purchase_amount'].agg(agg_func).reset_index(), ['merchant_category_id', 'card_id'], ['merchant_category_id', 'card_id'])
gc.collect()


# In[32]:


# fillna for ['native_city_merch_cat_merch_nunique',  'native_state_merch_cat_merch_nunique', 'native_city_subsector_merch_nunique', 'native_state_subsector_merch_nunique', 'card_state_merch_cat_city_nunique']
# ttt.fillna(0, inplace=True)
ttt[['native_city_merch_cat_merch_nunique',
       'native_state_merch_cat_merch_nunique',
       'native_city_subsector_merch_nunique',
       'native_state_subsector_merch_nunique', 'card_state_merch_cat_city_nunique']] = ttt[['native_city_merch_cat_merch_nunique',
       'native_state_merch_cat_merch_nunique',
       'native_city_subsector_merch_nunique',
       'native_state_subsector_merch_nunique', 'card_state_merch_cat_city_nunique']].fillna(0)


# In[33]:


# calculating isTrain_mean is necessary to weighting samples while fitting for prediction outlier_merch_mean
# for example, we calculated outlier_merch_mean for one merchant_id - 2 outliers from train, 9 card_id from train and 1 card from test
# in that example outlier_merch_mean = 2/(9 + 1) = 0.2
# but we don't know whether card from test is outlier or not
# so we can only suppose that final outlier_merch_mean in this specific example  would be either 0.2 (if test card is not outlier) or 0.3 (if test card is outlier)
# so our outlier_merch_mean = 0.2 with weight = 9/10 - 9 cards from train and total 10 cards (train + test)
# isTrain_mean - is out outlier_merch_mean weight
_ = ttt.groupby(['merchant_id'])['isTrain'].mean().reset_index().rename(columns={'isTrain': 'isTrain_mean'})
ttt = merge_two_df(ttt, _, ['merchant_id'], ['merchant_id'])
old_merch = merge_two_df(old_merch, _, ['merchant_id'], ['merchant_id'])
del _; gc.collect()


# In[34]:


old_merch = merge_two_df(old_merch, old_merch.groupby('merchant_category_id')['merchant_id'].nunique().reset_index().rename(columns={'merchant_id': 'merch_cat_merch_nunique'}), ['merchant_category_id'], ['merchant_category_id'])
old_merch = merge_two_df(old_merch, old_merch.groupby(['city_id', 'merchant_category_id'])['merchant_id'].nunique().reset_index().rename(columns={'merchant_id': 'merch_cat_city_merch_nunique'}), ['city_id', 'merchant_category_id'], ['city_id', 'merchant_category_id'])
old_merch = merge_two_df(old_merch, old_merch.groupby(['state_id', 'merchant_category_id'])['merchant_id'].nunique().reset_index().rename(columns={'merchant_id': 'merch_cat_state_merch_nunique'}), ['state_id', 'merchant_category_id'], ['state_id', 'merchant_category_id'])
gc.collect()


# In[35]:


old_merch.fillna(0, inplace=True)


# In[36]:


# if all test card_ids are non_outliers
old_merch['merch_outlier_mean_min'] = old_merch['merch_outlier_mean']*old_merch['isTrain_mean']*old_merch['merch_card_id_nunique']/old_merch['merch_card_id_nunique']
# if all test card_ids are outliers
old_merch['merch_outlier_mean_max'] = ((1-old_merch['isTrain_mean'])*old_merch['merch_card_id_nunique'] + old_merch['merch_outlier_mean']*old_merch['isTrain_mean']*old_merch['merch_card_id_nunique'])/old_merch['merch_card_id_nunique']
# mean between 'merch_outlier_mean_min' and 'merch_outlier_mean_max'
old_merch['merch_outlier_mean_mean'] = (old_merch['merch_outlier_mean_min'] + old_merch['merch_outlier_mean_max'])/2.0


# In[37]:


old_merch[['merch_outlier_mean_min', 'merch_outlier_mean_max', 'merch_outlier_mean_mean', 'isTrain_mean']].describe()


# In[38]:


old_merch.columns


# In[39]:


clmns = ['numerical_1',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'avg_sales_lag3', 'avg_purchases_lag3', #'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', #'active_months_lag6',
       'avg_sales_lag12', 'avg_purchases_lag12',# 'active_months_lag12',
       'category_4', 'city_id', 'state_id', 
       'merch_card_id_nunique',
       'merch_last_date_diff_min', 'merch_last_date_diff_max',
       'merch_last_date_diff_mean', 'merch_last_date_diff_median',
       'merch_cat_merch_nunique', 'merch_cat_city_merch_nunique',
       'merch_cat_state_merch_nunique', 
#        'merch_outlier_mean_min', 'merch_outlier_mean_max',
#        'merch_outlier_mean_mean', 
        ]
clmns_weight = clmns + ['isTrain_mean']

test_size = 0.2
random_state = 241
cond = (old_merch['isTrain_mean']>0)# pick all merchants, which have been visited at least by one card_id from train
x_train, x_test, y_train, y_test = train_test_split(old_merch[cond][clmns_weight], old_merch[cond]['merch_outlier_mean'], test_size=test_size, random_state=random_state)


# In[40]:


# cat_features = ['most_recent_sales_range', 'most_recent_purchases_range', 
#                 'active_months_lag3', 'active_months_lag6', 'active_months_lag12', 
#                 'category_4']

train_ds = lgb.Dataset(x_train[clmns], y_train, weight=x_train['isTrain_mean'])#, categorical_feature=cat_features)
test_ds = lgb.Dataset(x_test[clmns], y_test, weight=x_test['isTrain_mean'])#, categorical_feature=cat_features)
num_leaves = 90
min_data_in_leaf = 30
params = {
        'objective' :'regression_l1',#regression_l2
        'bagging_fraction': 0.5, 'feature_fraction': 0.9, 'learning_rate': 0.1, 
    'max_bin': 100, 'min_data_in_leaf': 30, 'num_leaves': 90,
        'boosting_type' : 'gbdt',
        'metric': 'l1'#l2_root, l2
    }
num_boost = 450
clf = lgb.train(params, train_ds, num_boost, valid_sets=[test_ds], verbose_eval=25, early_stopping_rounds=50)

pred = clf.predict(x_test[clmns])
print('TEST MAE:', round(mean_absolute_error(y_test, pred), 8), 'MSE:', round(mean_squared_error(y_test, pred), 8))
pred = clf.predict(x_train[clmns])
print('TRAIN MAE:', round(mean_absolute_error(y_train, pred), 8), 'MSE:', round(mean_squared_error(y_train, pred), 8))


# In[41]:


clf.save_model('../models/merch_outlier_mean_lgb.model')


# In[42]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(), clmns)), columns=['Value','Feature']).sort_values(by='Value', ascending=False)
feature_imp.head(30)
# clf.feature_importance(), clmns


# In[43]:


old_merch['merch_outlier_mean_pred'] = clf.predict(old_merch[clmns])
# control that our predictions wouldn't go out of min and max values for a specific merchant_id
old_merch['merch_outlier_mean_pred_fixed'] = np.minimum(np.maximum(old_merch['merch_outlier_mean_pred'].values, old_merch['merch_outlier_mean_min'].values), old_merch['merch_outlier_mean_max'].values)


# In[44]:


gc.collect()


# In[45]:


'number of merchants that have been visited at least by one outlier according to the model:', old_merch[old_merch['merch_outlier_mean_pred_fixed']>0].shape[0]


# In[46]:


ttt.columns


# In[47]:


gc.collect()


# In[48]:


ttt = merge_two_df(ttt, old_merch[['merchant_id', 'numerical_1',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12',
       'category_4', 'merch_last_date_diff_min', 'merch_last_date_diff_max',
       'merch_last_date_diff_mean', 'merch_last_date_diff_median', 
        'merch_outlier_mean_min', 'merch_outlier_mean_max',
       'merch_outlier_mean_mean', 'merch_outlier_mean_pred',
       'merch_outlier_mean_pred_fixed']], ['merchant_id'], ['merchant_id'])


# In[49]:


ttt = merge_two_df(ttt, train[['card_id', 'feature_1', 'feature_2', 'feature_3']], ['card_id'], ['card_id'])
ttt = merge_two_df(ttt, test[['card_id', 'feature_1', 'feature_2', 'feature_3']], ['card_id'], ['card_id'])
ttt.loc[ttt['feature_1_x'].isnull(), ['feature_1_x', 'feature_2_x', 'feature_3_x']] = ttt.loc[ttt['feature_1_x'].isnull(), ['feature_1_y', 'feature_2_y', 'feature_3_y']].values
ttt.drop(['feature_1_y', 'feature_2_y', 'feature_3_y'], axis=1, inplace=True)
ttt.rename(columns={'feature_1_x': 'feature_1', 'feature_2_x': 'feature_2', 'feature_3_x': 'feature_3'}, inplace=True)


# In[50]:



clmns = ['purchase_amount',
       'purchase_amount_rel', 'installments', 'category_3',
       'merchant_category_id_last_visited_ref_date_day_diff',
       'last_visited_ref_date_day_diff', 'last_vis_merch_last_vis_day_diff',
       'first_active_date_elapsed_day', 
#          'city_id', 'state_id', 
         'card_city_nunique', 'card_state_nunique',
       'card_state_city_nunique', #'merchant_category_id', 'subsector_id',
#        'native_city', 'native_state', 
         'native_city_merch_cat_merch_nunique',
       'native_state_merch_cat_merch_nunique',
       'native_city_subsector_merch_nunique',
       'native_state_subsector_merch_nunique', 'card_merch_cat_city_nunique',
       'card_state_merch_cat_city_nunique', 
#        'isTrain_mean',
         'merch_card_last_date_diff',
       'merchant_category_id_card_id_purchase_amount_rel_mean',
       'merchant_category_id_card_id_purchase_amount_rel_median',
       'merchant_category_id_card_id_purchase_amount_count',
       'merchant_category_id_card_id_purchase_amount_abs_mean',
       'merchant_category_id_card_id_purchase_amount_abs_median',
       #'most_recent_sales_range', 'most_recent_purchases_range',
       'category_4', 'merch_last_date_diff_max',
       'merch_last_date_diff_mean', 'merch_last_date_diff_median',
#        'merch_outlier_mean_min', 
#          'merch_outlier_mean_max',
#        'merch_outlier_mean_mean',
#          'merch_outlier_mean_pred',
       'merch_outlier_mean_pred_fixed', 
         'feature_1', 'feature_2', 'feature_3']

print(len(clmns))
# cat_features = ['category_4', 'feature_1', 'feature_2', 'feature_3']

random_state = 241
test_size = 0.2
cnd = (ttt['isTrain']==1) & (ttt['merch_outlier_mean_pred_fixed']>0)
x_train, x_test, y_train, y_test = train_test_split(ttt[cnd][clmns], ttt[cnd]['outlier'], test_size=test_size, random_state=random_state)


# In[51]:


train_ds = lgb.Dataset(x_train, y_train)#, categorical_feature=cat_features)
test_ds = lgb.Dataset(x_test, y_test)#, categorical_feature=cat_features)
num_leaves = 150
min_data_in_leaf = 150
params = {
        'objective' :'binary',
        'learning_rate': 0.1,
        'max_bin': 50,
#         'max_depth': 10,
        'num_leaves' : num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'feature_fraction': 0.64, 
        'bagging_fraction': 0.8, 
        'bagging_freq': 1,
        'num_threads': 3,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'boosting_type' : 'gbdt',
        'metric': 'binary_logloss'#l2_root, l2
    }

# you can increase num_boost parameter, if you want, because the model doesn't stop fitting on 250 iteration for me
num_boost = 250
clf = lgb.train(params, train_ds, num_boost, valid_sets=[test_ds], verbose_eval=25, early_stopping_rounds=30)

pred = np.round(clf.predict(x_test))
print('TEST f1:', f1_score(pred, y_test), 'accuracy:', accuracy_score(y_test, pred))
del pred;
gc.collect()


# In[52]:


clf.save_model('../models/card_merch_outlier_lgb.model')


# In[53]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(), clmns)), columns=['Value','Feature']).sort_values(by='Value', ascending=False)
feature_imp.tail(31)
# clf.feature_importance(), clmns


# In[54]:


cond = (ttt['merch_outlier_mean_pred_fixed']>0)
ttt['isOutlier_pred'] = 0
ttt.loc[cond, 'isOutlier_pred'] = clf.predict(ttt[cond][clmns])
gc.collect()


# In[55]:


agg_func = {
    'merchant_category_id': 'nunique',
    'subsector_id': 'nunique',
    'purchase_amount': ['mean', ('median', np.median), ],
    'purchase_amount_rel': 'median',
    'trans_count': ['sum', ('merch_count', len), 'mean'],
    'installments': 'median',
    'category_3': 'median',
    'merchant_category_id_last_visited_ref_date_day_diff': 'mean',
    'last_visited_ref_date_day_diff': 'mean',
    'last_vis_merch_last_vis_day_diff': 'mean',
    'card_city_nunique': max, 
    'card_state_nunique': max,
    'card_state_city_nunique': max,
    'numerical_1': 'mean',
    'most_recent_sales_range': 'median',
    'most_recent_purchases_range': 'median',
    'numerical_1': 'mean',
    'category_4': 'mean',
    'merch_outlier_mean_min': 'mean', 
    'merch_outlier_mean_max': 'mean',
    'merch_outlier_mean_mean': 'mean', 
    'merch_outlier_mean_pred': 'mean',
    'merch_outlier_mean_pred_fixed': 'mean',
    'isOutlier_pred': ['min', 'max', 'mean', 'median']
}
_ = ttt.groupby('card_id').agg(agg_func).reset_index()
_.columns = ["_".join(x) if x[1] else x[0] for x in _.columns.ravel()]
train = merge_two_df(train, _, ['card_id'], ['card_id'])
test = merge_two_df(test, _, ['card_id'], ['card_id'])
_.head()


# In[56]:


train.corr()['outlier']


# In[57]:


train.columns


# In[58]:


clmns = ['feature_1', 'feature_2', 'feature_3',
       'first_active_date_elapsed_day', 
       'isOutlier_pred_min', 'isOutlier_pred_max', 'isOutlier_pred_mean',
       'isOutlier_pred_median', 'merchant_category_id_nunique',
       'subsector_id_nunique', 'purchase_amount_mean',
       'purchase_amount_median', 'purchase_amount_rel_median',
       'trans_count_sum', 'trans_count_merch_count', 'trans_count_mean',
       'installments_median', 'category_3_median',
       'merchant_category_id_last_visited_ref_date_day_diff_mean',
       'last_visited_ref_date_day_diff_mean',
       'last_vis_merch_last_vis_day_diff_mean', 'card_city_nunique_max',
       'card_state_nunique_max', 'card_state_city_nunique_max',
       'numerical_1_mean', 'most_recent_sales_range_median',
       'most_recent_purchases_range_median', 'category_4_mean',
#        'merch_outlier_mean_min_mean', 'merch_outlier_mean_max_mean',
#        'merch_outlier_mean_mean_mean', 'merch_outlier_mean_pred_mean',
         'merch_outlier_mean_pred_fixed_mean']

# clmns = [
#        'isOutlier_pred0_mean', 'isOutlier_pred0_median',
#     'merch_outlier_mean_pred_fixed',
#        'feature_1', 'feature_2', 'feature_3', 
# #     'first_active_date_elapsed_day', 
# #     'merch_outlier_mean_pred_fixed_mean',
# #        'merch_outlier_mean', 'merch_outlier_sum',
# #        'merch_outlier_max'
#         ]

random_state = 241
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(train[clmns], train['outlier'], test_size=test_size, random_state=random_state)


# In[59]:


train_ds = lgb.Dataset(x_train, y_train)#, categorical_feature=cat_features)
test_ds = lgb.Dataset(x_test, y_test)#, categorical_feature=cat_features)
num_leaves = 50
min_data_in_leaf = 50
params = {
        'objective' :'binary',
        'learning_rate': 0.05,
        'max_bin': 50,
        'max_depth': 5,
        'num_leaves' : num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
#         'feature_fraction': 0.64, 
#         'bagging_fraction': 0.8, 
#         'bagging_freq': 1,
#         'device': 'gpu',
        'lambda_l1': 5,
        'lambda_l2': 5,
#         'min_gain_to_split': .1,
        'boosting_type' : 'gbdt',
        'metric': 'binary_logloss'#l2_root, l2
    }
num_boost = 350
clf = lgb.train(params, train_ds, num_boost, valid_sets=[test_ds], verbose_eval=25, early_stopping_rounds=30)

pred = np.round(clf.predict(x_test))
print('TEST f1:', f1_score(pred, y_test), 'accuracy:', accuracy_score(y_test, pred), 'number of bad predicted samples:', (1-accuracy_score(y_test, pred))*y_test.shape[0], 'out of', y_test.shape[0])
print('number of outliers in validation set:', y_test.value_counts().values[1])
pred = np.round(clf.predict(x_train))
print('TRAIN f1:', f1_score(pred, y_train), 'accuracy:', accuracy_score(y_train, pred), 'number of bad predicted samples:', (1-accuracy_score(y_train, pred))*y_train.shape[0], 'out of', y_test.shape[0])
print('number of outliers in training set:', y_train.value_counts().values[1])

cond = train['outlier']==1
# threshold = .7
pred = np.round(clf.predict(train[cond][clmns]))
print('bad classified outliers in all TRAIN dataset:', (1-accuracy_score(train[cond]['outlier'], pred))*train[cond].shape[0])
      
cond = train['outlier']==0
pred = np.round(clf.predict(train[cond][clmns]))
print('bad classified good cards in all TRAIN dataset:', (1-accuracy_score(train[cond]['outlier'], pred))*train[cond].shape[0])


# In[60]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(), clmns)), columns=['Value','Feature']).sort_values(by='Value', ascending=False)
feature_imp.tail(34)
# clf.feature_importance(), clmns


# In[61]:


#distribution of predicted outliers in TEST.csv
pd.Series(np.round(clf.predict(test[clmns]))).value_counts()


# In[62]:


#distribution of predicted outliers in TEST.csv
pd.Series(np.round(clf.predict(test[clmns]))).value_counts()

