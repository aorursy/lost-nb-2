#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from numpy import NaN
pd.set_option('display.max_rows', 70)
pd.set_option('display.max_columns', 70)
pd.set_option('display.width', 100)


# In[2]:


prop16 = pd.read_csv("../input/zillow-prize-1/properties_2016.csv")
prop17 = pd.read_csv("../input/zillow-prize-1/properties_2017.csv")
smplsub = pd.read_csv("../input/zillow-prize-1/sample_submission.csv")
train16 = pd.read_csv("../input/zillow-prize-1/train_2016_v2.csv")
train17 = pd.read_csv("../input/zillow-prize-1/train_2017.csv")


# In[3]:


prop16.head()


# In[4]:


prop17.head()


# In[5]:


train16.head()


# In[6]:


train17.head()


# In[7]:


smplsub.head()


# In[8]:


#function to get all info in one go
def full_info(df):
    df_column=[]
    df_dtype=[]
    df_null=[]
    df_nullc=[]
    df_mean=[]
    df_median=[]
    df_std=[]
    df_min=[]
    df_max=[]
    df_uniq=[]
    for col in df.columns: 
        df_column.append( col)
        df_dtype.append( df[col].dtype)
        df_null.append( round(100 * df[col].isnull().sum(axis=0)/len(df[col]),2))
        df_nullc.append( df[col].isnull().sum(axis=0))
        df_uniq.append( df[col].nunique()) if df[col].dtype == 'object' else df_uniq.append( NaN)
        df_mean.append(  '{0:.2f}'.format(df[col].mean())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_mean.append( NaN)
        df_median.append( '{0:.2f}'.format(df[col].median())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_median.append( NaN)
        df_std.append( '{0:.2f}'.format(df[col].std())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_std.append( NaN)
        df_max.append( '{0:.2f}'.format(df[col].max())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_max.append( NaN)
        df_min.append( '{0:.2f}'.format(df[col].min())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_min.append( NaN)
    return pd.DataFrame(data = {'ColName': df_column, 'ColType': df_dtype, 'NullCnt': df_nullc, 'NullCntPrcntg': df_null,  'Min': df_min, 'Max': df_max, 'Mean': df_mean, 'Med': df_median, 'Std': df_std, 'UniqCnt': df_uniq})


# In[9]:


prop16_Info = full_info(prop16)
prop16_Info.sort_values(by=['NullCnt'], ascending=False, inplace=True, ignore_index=True)
prop16_Info


# In[10]:


full_info(train16)


# In[11]:


print('size of properties_2016.csv: ', prop16.shape)
print('size of train_2016_v2.csv: ', train16.shape)
print('size of properties_2017.csv: ', prop17.shape)
print('size of train_2017.csv: ', train17.shape)


# In[12]:


unique_props = len(train16['parcelid'].unique())
multiple_sales = len(train16) - unique_props
print('number of unique sales: ', unique_props)
print('Number of duplicate: ', multiple_sales)


# In[13]:


# lets visualize the Null Count percentage graphically
prop16_Info.plot.bar(x = 'ColName', y = 'NullCnt', figsize=(25, 6),rot=90, title='Missing (null) Feature Values')
plt.show()


# In[14]:


# interactive feature transfromation.
prop16['prop_age'] = 2018 - prop16['yearbuilt']  # property age
prop16['has_basement'] = prop16['basementsqft'].apply(lambda x: 0 if np.isnan(x) else 1).astype(float)
prop16['has_pool'] = prop16[['poolcnt','poolsizesum','pooltypeid10','pooltypeid2','pooltypeid7']].apply(lambda x: 1 if(np.all(pd.notnull(x[1]))) else 0, axis = 1)
prop16['has_patio_yard'] = prop16['yardbuildingsqft17'].apply(lambda x: 0 if np.isnan(x) else 1).astype(float)
prop16['has_starage_yard'] = prop16['yardbuildingsqft26'].apply(lambda x: 0 if np.isnan(x) else 1).astype(float)
prop16['has_garage'] = prop16['garagecarcnt'].apply(lambda x: 0 if np.isnan(x) else 1).astype(float)


# In[15]:


# some nan features actually make sense, lets fill them with 0
prop16.yardbuildingsqft17.fillna(0, inplace=True)
prop16.yardbuildingsqft26.fillna(0, inplace=True)
prop16.basementsqft.fillna(0, inplace=True)
prop16.poolcnt.fillna(0, inplace=True)
prop16.poolsizesum.fillna(0, inplace=True)
prop16.pooltypeid10.fillna(0, inplace=True)
prop16.pooltypeid2.fillna(0, inplace=True)
prop16.pooltypeid7.fillna(0, inplace=True)
prop16.garagecarcnt.fillna(0, inplace=True)


# In[16]:


# drop columns with data that has > 90% null
prop16_trim = prop16.drop(prop16_Info[(prop16_Info.NullCntPrcntg>=90)].ColName.values.tolist(),axis=1)
prop16_trim


# In[17]:


prop16_trim.select_dtypes(include=['object']).columns


# In[18]:


# these object dtype not categorical. can be ignored.
prop16_trim[['propertycountylandusecode', 'propertyzoningdesc']]


# In[19]:


# drop the object dtype columns
prop16_trim=prop16_trim.drop(['propertycountylandusecode', 'propertyzoningdesc'],axis=1)


# In[20]:


# lets fill rest of the NaNs with medians
prop16_median_imputed = prop16_trim.fillna(prop16_trim.median())
prop16_median_imputed


# In[21]:


# lets concatenated both property and train data
train16_merge = pd.merge(prop16_median_imputed, train16, on='parcelid', how='inner')
train16_merge


# In[22]:


train16_merge=train16_merge.drop(['parcelid'],axis=1)


# In[23]:


# lets check the correlation of the feature to target
from yellowbrick.target.feature_correlation import feature_correlation
X, y = train16_merge.drop(columns =[ 'logerror', 'transactiondate', 'fireplacecnt']), train16_merge['logerror']

features = np.array(train16_merge.drop(columns = [ 'logerror', 'transactiondate', 'fireplacecnt']).columns)
fig, ax = plt.subplots(figsize=(10,18))
visualizer = feature_correlation(X, y, labels=features, sort= True, color='gray', show=True, ax=ax)
plt.show()


# In[24]:


# import LightGBM Libraries
import lightgbm as lgb
import random


# In[25]:


#LightGBM accepts numphy array as input
x_train = train16_merge.drop(columns =[ 'logerror', 'transactiondate', 'fireplacecnt']).values.astype(np.float32) # np array
y_train = train16_merge['logerror'].values.astype(np.float32)  # np array
x_test = train16_merge.drop([ 'logerror', 'transactiondate', 'fireplacecnt'], axis=1).values.astype(np.float32)  # np array
train_columns = train16_merge.drop(columns = [ 'logerror', 'transactiondate', 'fireplacecnt']).columns 


# In[26]:


# manually added the features as the numbers in some features is not acceptable for lgb
d_train = lgb.Dataset(x_train, y_train, feature_name=['airconditioningtypeid', 'bathroomcnt', 'bedroomcnt',
        'buildingqualitytypeid', 'calculatedbathnbr',
       'calculatedfinishedsquarefeet', 'finishedsquarefeet', 'fips',
       'fullbathcnt', 'garagecarcnt', 'garagetotalsqft',
       'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
       'poolcnt', 'pooltypeid', 'propertylandusetypeid',
       'rawcensustractandblock', 'regionidcity', 'regionidcounty',
       'regionidneighborhood', 'regionidzip', 'roomcnt', 'threequarterbathnbr',
       'unitcnt', 'yearbuilt', 'numberofstories', 'structuretaxvaluedollarcnt',
       'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',
       'taxamount', 'censustractandblock', 'prop_age', 'has_basement',
       'has_pool', 'has_patio_yard', 'has_starage_yard','has_garage'])  # lightgbm data model


# In[27]:


# lgb hyper parameters
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.01  # shrinkage_rate 0.0021 grid search = 0.01
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'  # l1
params['sub_feature'] = 0.5  # feature_fraction
params['bagging_fraction'] = 0.85  # sub_row
params['num_leaves'] = 512  # num_leaf
params['min_data'] = 500  # min_data_in_leaf
params['min_hessian'] = 0.05  # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3
#params['n_estimators'] = 10  # grid search
params['colsample_bytree'] = 0.85
params['num_leaves'] = 22
params['subsample'] = 0.7

np.random.seed(0)
random.seed(0)


# In[28]:


# lgb train
clf = lgb.train(params, d_train, 430)


# In[29]:


# lgb predict
p_test = clf.predict(x_test)
pd.DataFrame(p_test).head()


# In[30]:


# lgb feature importance
lgb.plot_importance(clf, figsize=(20,20))
plt.show()


# In[31]:


# lgb tree plot
import os
os.environ["PATH"] += os.pathsep + '/opt/anaconda3/lib/python3.7/site-packages/sphinx/templates/graphviz'
lgb.plot_tree(clf, figsize=(100,40))
plt.show()


# In[32]:


# import the library
import xgboost as xgb


# In[33]:


#XGBoost instead accepts dataframe as input
x_train_xgb = train16_merge.drop(columns =[ 'logerror', 'transactiondate', 'fireplacecnt']) 
y_train_xgb = train16_merge['logerror']
x_test_xgb = train16_merge.drop([ 'logerror', 'transactiondate', 'fireplacecnt'], axis=1)
train_columns_xgb = train16_merge.drop(columns = [ 'logerror', 'transactiondate', 'fireplacecnt']).columns 


# In[34]:


# xgb hyperparameters

y_mean = np.mean(y_train_xgb)
xgb_params1 = {
    'eta' : 0.04,  # 0.037 grid search = .04
    'max_depth' : 6,  #5
    'subsample' : 0.80,
    'objective' : 'reg:linear',
    'eval_metric' : 'mae',
    'lambda' : 0.8,
    'alpha' : 0.4,
    'base_score' : y_mean,
    'silent' : 1,
    'min_child_weight': 5  # grid search
}


# In[35]:


dtrain1 = xgb.DMatrix(x_train_xgb, y_train_xgb, feature_names=train_columns_xgb)
dtest1 = xgb.DMatrix(x_test_xgb)


# In[36]:


# training
model1 = xgb.train(dict(xgb_params1, silent=1), dtrain1, num_boost_round=150)


# In[37]:


# xgb feature importance
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model1, max_num_features=50, height=0.8, ax=ax)
plt.show()


# In[38]:


fig, ax = plt.subplots(figsize=(100, 60))
xgb.plot_tree(model1, num_trees=4, ax=ax)
plt.show()


# In[39]:


# xgb2 hyperparameters
xgb_params2 = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:squarederror',
    'silent': 1,
    'seed' : 0
}
dtrain2 = xgb.DMatrix(x_train_xgb, y_train_xgb, feature_names=train_columns)
dtest2 = xgb.DMatrix(x_test_xgb)


# In[40]:


# training
model2 = xgb.train(dict(xgb_params2, silent=0), dtrain2, num_boost_round=50)


# In[41]:


# predict
xgb_pred2 = model2.predict(dtest2)
xgb_pred2


# In[42]:


# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model2, max_num_features=50, height=0.8, ax=ax)
plt.show()


# In[43]:


fig, ax = plt.subplots(figsize=(100, 60))
xgb.plot_tree(model2, num_trees=4, ax=ax)
plt.show()

