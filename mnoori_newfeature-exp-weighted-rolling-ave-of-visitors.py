#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob,re,os
import numpy as np
import pandas as pd
from sklearn import *
from xgboost import XGBRegressor
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[2]:


#Assiging a name to each data frame
data={
    'tra':pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_visit_data.csv'),
    'as':pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_store_info.csv'),
    'hs':pd.read_csv('../input/recruit-restaurant-visitor-forecasting/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_reserve.csv'),
    'hr': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/hpg_reserve.csv'),
    'id': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/store_id_relation.csv'),
    'tes': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/sample_submission.csv'),
    'hol':pd.read_csv('../input/recruit-restaurant-visitor-forecasting/date_info.csv').rename(columns={'calendar_date':'visitor_date'})
}


# In[3]:


plt.subplots(figsize=(12,6))
plt.subplot(1, 2, 1)
data['tra']['visitors'].hist()
plt.title('Histogram of visitors')
plt.grid(False)

plt.subplot(1, 2, 2)
plt.title('Histogram of log of visitors')
data['tra']['visitors'].map(pd.np.log1p).hist()
plt.grid(False)


# In[4]:


# Now let's add to the hpg_reserve the ids from id dataset
data['hr']=data['hr'].merge(data['id'],on=['hpg_store_id'],how='inner')

data['ar'].head()


# In[5]:


#let's tranfrom date to datetime objects. Please note, to_datetime also includes the actual time. Using .dt.date we only capture date.
for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime']).dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime']).dt.date
    
    #here, we are actually engineering a new feature that captures the difference between visit and reserve times
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    
    #let's group datasets by id and visit date, then get the sum and mean of reserve and reserve differnce, then rename the columns
    temp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    
    temp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    #now let's merge these two new temp dataframes.
    data[df]=temp1.merge(temp2,how='inner',on=['air_store_id','visit_date'])


# In[6]:


#let's take a look at hr and see what has happened to it.
data['hr'].head()


# In[7]:


data['tra']['visit_date']=pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow']=data['tra']['visit_date'].dt.dayofweek
data['tra']['year']=data['tra']['visit_date'].dt.year
data['tra']['month']=data['tra']['visit_date'].dt.month
data['tra']['visit_date']=data['tra']['visit_date'].dt.date


# In[8]:


data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date


# In[9]:


unique_stores=data['tes']['air_store_id'].unique()
print('The number of unique stores is:', unique_stores.shape[0])
print('total number of data records in test set is',data['tes'].shape[0])


# In[10]:


stores=pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)],
            axis=0,ignore_index=True).reset_index(drop=True)

stores.shape


# In[11]:


#loading the test set with first round of projections.
data['test_ewm']=pd.read_csv('../input/first-round-predictions/test_ewm.csv')

#defining a function that calculates the exponential weighted value of a series. alpha is the smoothing factor.
def calc_shifted_ewm(series, alpha, adjust=True):
    return series.shift().ewm(alpha=alpha, adjust=adjust).mean()

for df in ['tra','test_ewm']:
    data[df]['ewm'] = data[df].groupby(['air_store_id', 'dow'])                  .apply(lambda g: calc_shifted_ewm(g['visitors'], 0.1)).sort_index(level=['air_store_id','dow']).values


# In[12]:


#finding the mean of ewm
mean_ewm_train=data['tra'].groupby(['air_store_id','dow']).mean().reset_index()
mean_ewm_test=data['test_ewm'].groupby(['air_store_id','dow']).mean().reset_index()

#setting new index for new_ewm_train
mean_ewm_train['id_dow']=mean_ewm_train.apply(lambda x: '_'.join([str(x['air_store_id']),str(x['dow'])]),axis=1)
mean_ewm_train=mean_ewm_train.set_index('id_dow')

mean_ewm_test['id_dow']=mean_ewm_test.apply(lambda x: '_'.join([str(x['air_store_id']),str(x['dow'])]),axis=1)
mean_ewm_test=mean_ewm_test.set_index('id_dow')


# In[13]:


#setting new index for data['tra']
data['tra']['id_dow']=data['tra'].apply(lambda x: '_'.join([str(x['air_store_id']),str(x['dow'])]),axis=1)
data['tra']=data['tra'].set_index('id_dow')

#setting new index for data['test_ewm']
data['test_ewm']['id_dow']=data['test_ewm'].apply(lambda x: '_'.join([str(x['air_store_id']),str(x['dow'])]),axis=1)
data['test_ewm']=data['test_ewm'].set_index('id_dow')


# In[14]:


#filling na
data['tra']['ewm']=data['tra']['ewm'].fillna(mean_ewm_train['ewm'])
data['test_ewm']['ewm']=data['test_ewm']['ewm'].fillna(mean_ewm_test['ewm'])

#making sure there are no missing values.
data['test_ewm'].isnull().sum()


# In[15]:


#merging new ewm with test set.
data['tes']=data['tes'].merge(data['test_ewm'],on=['id'],how='left')
data['tes']=data['tes'][['id','visitors_x','visit_date','air_store_id_x','dow_x','year','month','ewm']]
data['tes']=data['tes'].rename(columns={'visitors_x':'visitors','air_store_id_x':'air_store_id','dow_x':'dow'})


# In[16]:


temp=data['tra'].groupby(['air_store_id','dow']).agg({'visitors':[np.min, np.mean, np.median, np.max, np.size]}).reset_index()

temp.head()


# In[17]:


temp.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors','max_visitors','count_observations']

stores=stores.merge(temp, on=['air_store_id','dow'],how='left')
stores.head()


# In[18]:


stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])

stores.head()


# In[19]:


stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))

#list of unique genres
stores['air_genre_name'].unique()
stores['air_genre_name'].unique().shape[0]


# In[20]:


stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))

#number of unique areas
stores['air_area_name'].unique().shape[0]


# In[21]:


lbl = preprocessing.LabelEncoder()
for i in range(10):
    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

stores['air_genre_name'].unique()


# In[22]:


data['hol'].head()


# In[23]:


data['hol']['visit_date']=pd.to_datetime(data['hol']['visitor_date'])
data['hol']['day_of_week']=lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date']=data['hol']['visit_date'].dt.date
data['hol']=data['hol'].drop('visitor_date',axis=1)

#merge the holiday flags to train and test sets.
train=data['tra'].merge(data['hol'],on=['visit_date'],how='left')
test=data['tes'].merge(data['hol'],on=['visit_date'],how='left')


# In[24]:


#merge stores
train=train.merge(stores,how='left',on=['air_store_id','dow'])
test=test.merge(stores,how='left',on=['air_store_id','dow'])


# In[25]:


for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])


# In[26]:


train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

#engineering new features

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

train.head()


# In[27]:


# engineeirng new features, Please refer to original codes mentioned in the introduction for more info.

train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']


# In[28]:


train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']


# In[29]:


lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])


# In[30]:


col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

#let's see how many features are we traning on
print('number of features are: ', len(col))


# In[31]:


# XGB starter template borrowed from @anokas: https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655

for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32)

for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.float32)


# In[32]:


#error metric
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5


# In[33]:


model1 = ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=3, n_estimators=200, subsample=0.8, 
                      max_depth =10)
model2 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
model3 = XGBRegressor(learning_rate=0.2, random_state=3, n_estimators=250, subsample=0.8, 
                      colsample_bytree=0.8, max_depth =10)

model1.fit(train[col], np.log1p(train['visitors'].values))
model2.fit(train[col], np.log1p(train['visitors'].values))
model3.fit(train[col], np.log1p(train['visitors'].values))

preds1 = model1.predict(train[col])
preds2 = model2.predict(train[col])
preds3 = model3.predict(train[col])

print('RMSLE GradientBoostingRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds1))
print('RMSLE KNeighborsRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds2))
print('RMSLE XGBRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds3))
preds1 = model1.predict(test[col])
preds2 = model2.predict(test[col])
preds3 = model3.predict(test[col])


# In[34]:


test['visitors'] = 0.3*preds1+0.3*preds2+0.4*preds3
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
sub1 = test[['id','visitors']].copy()
del train; del data;


# In[35]:


# from hklee
# https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st/code
dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):
    pd.read_csv(fn)for fn in glob.glob('../input/recruit-restaurant-visitor-forecasting/*.csv')}

for k, v in dfs.items(): locals()[k] = v

wkend_holidays = date_info.apply(
    (lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5  

visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)

wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0:'visitors'}, inplace=True) # cumbersome, should be better ways.

sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(visitors, on=[
    'air_store_id', 'day_of_week', 'holiday_flg'], how='left')

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), 
    how='left')['visitors_y'].values

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), 
    on='air_store_id', how='left')['visitors_y'].values

sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)
sub2 = sample_submission[['id', 'visitors']].copy()
sub_merge = pd.merge(sub1, sub2, on='id', how='inner')

sub_merge['visitors'] = 0.7*sub_merge['visitors_x'] + 0.3*sub_merge['visitors_y']* 1.2
sub_merge[['id', 'visitors']].to_csv('submission.csv', index=False)

