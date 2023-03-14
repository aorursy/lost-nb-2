#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


# In[3]:


get_ipython().run_cell_magic('time', '', "\n# Read data...\nroot = '../input/ashrae-energy-prediction'\n\ntrain_df = pd.read_csv(os.path.join(root, 'train.csv'))\nweather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))\ntest_df = pd.read_csv(os.path.join(root, 'test.csv'))\nweather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))\nbuilding_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))\nsample_submission = pd.read_csv(os.path.join(root, 'sample_submission.csv'))")


# In[4]:


# transfer to .feature format

train_df.to_feather('train.feather')
test_df.to_feather('test.feather')
weather_train_df.to_feather('weather_train.feather')
weather_test_df.to_feather('weather_test.feather')
building_meta_df.to_feather('building_metadata.feather')
sample_submission.to_feather('sample_submission.feather')


# In[5]:


# %%time

train_df = pd.read_feather('train.feather')
weather_train_df = pd.read_feather('weather_train.feather')
# test_df = pd.read_feather('test.feather')
# weather_test_df = pd.read_feather('weather_test.feather')
building_meta_df = pd.read_feather('building_metadata.feather')
sample_submission = pd.read_feather('sample_submission.feather')


# In[6]:


train_df = train_df.merge(building_meta_df, left_on = "building_id", right_on = "building_id", how = "left")
train_df = train_df.merge(weather_train_df, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
# del weather_train_df


# In[7]:


train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
train_df["hour"] = train_df["timestamp"].dt.hour
train_df["day"] = train_df["timestamp"].dt.day
train_df["weekend"] = train_df["timestamp"].dt.weekday
train_df["month"] = train_df["timestamp"].dt.month
train_df = train_df.drop("timestamp", axis = 1)


# In[8]:


# # plot the log of (1+ meter_reading)
# import plotly.express as px
# def plot_date_usage(train_df, meter=0, building_id=0):
#     train_temp_df = train_df[train_df['meter'] == meter]
#     train_temp_df = train_temp_df[train_temp_df['building_id'] == building_id]    
    
#     train_temp_df_meter = train_temp_df.groupby('date')['meter_reading_log1p'].sum()
#     train_temp_df_meter = train_temp_df_meter.to_frame().reset_index()
#     fig = px.line(train_temp_df_meter, x='date', y='meter_reading_log1p')
#     fig.show()
# plot_date_usage(train_df, meter=0, building_id=0)


# In[9]:


# building_meta_df[building_meta_df.site_id == 0]


# In[10]:


# train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['primary_use'] = le.fit(train_df['primary_use']).transform(train_df['primary_use'])
train_df['primary_use'].unique()


# In[12]:


# train_df.info()


# In[13]:


categoricals = ["building_id", "primary_use", "hour", "day", "weekend", "month", "meter"]
drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]
numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature"]
feat_cols = categoricals + numericals
train_df = train_df.drop(drop_cols + ["site_id", "floor_count"], axis = 1)


# In[14]:


train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])
del train_df["meter_reading"]
target = 'meter_reading_log1p'


# In[15]:


train_df, NAlist = reduce_mem_usage(train_df)


# In[16]:


# (train_y<=0).sum()


# In[17]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
target = 'meter_reading_log1p'
num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = False, random_state = 42)
error = 0
models = []
for i, (train_index, val_index) in enumerate(kf.split(train_df)):
    if i + 1 < num_folds:
        continue
    print(train_index.max(), val_index.min())
    train_X = train_df[feat_cols].iloc[train_index]
    val_X = train_df[feat_cols].iloc[val_index]
    train_y = train_df[target].iloc[train_index]
    val_y = train_df[target].iloc[val_index]
    
    
    lgb_train = lgb.Dataset(train_X, train_y > 0)
    lgb_eval = lgb.Dataset(val_X, val_y > 0)
    params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq' : 5
            }
    gbm_class = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=20,
               verbose_eval = 20)
    
    
    
    
    
    
    lgb_train = lgb.Dataset(train_X[train_y > 0], train_y[train_y > 0])
    lgb_eval = lgb.Dataset(val_X[val_y > 0] , val_y[val_y > 0])
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'learning_rate': 0.5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq' : 5
            }

    gbm_regress = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=20,
               verbose_eval = 20)
#     models.append(gbm)

    y_pred = (gbm_class.predict(val_X, num_iteration=gbm_class.best_iteration) > .5) *    (gbm_regress.predict(val_X, num_iteration=gbm_regress.best_iteration))
    error += np.sqrt(mean_squared_error(y_pred, (val_y)))/num_folds
    print(np.sqrt(mean_squared_error(y_pred, (val_y))))
    break
print(error)


# In[18]:


sorted(zip(gbm_regress.feature_importance(), gbm_regress.feature_name()),reverse = True)


# In[19]:


import gc
del train_df
del train_X, val_X, lgb_train, lgb_eval, train_y, val_y, y_pred, target
gc.collect()


# In[20]:


#preparing test data
test_df = pd.read_feather('test.feather')
test_df, NAlist = reduce_mem_usage(test_df)
test_df = test_df.merge(building_meta_df, left_on = "building_id", right_on = "building_id", how = "left")


# In[21]:


del building_meta_df


# In[22]:


gc.collect()


# In[23]:


weather_test_df = pd.read_feather('weather_test.feather')
weather_test_df = weather_test_df.drop(drop_cols, axis = 1)
test_df = test_df.merge(weather_test_df, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
del weather_test_df


# In[24]:


test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
test_df["hour"] = test_df["timestamp"].dt.hour.astype(np.uint8)
test_df["day"] = test_df["timestamp"].dt.day.astype(np.uint8)
test_df["weekend"] = test_df["timestamp"].dt.weekday.astype(np.uint8)
test_df["month"] = test_df["timestamp"].dt.month.astype(np.uint8)
test_df = test_df[feat_cols]


# In[25]:


test_df["primary_use"] = le.transform(test_df["primary_use"])


# In[26]:


from tqdm import tqdm
i=0
res=[]
step_size = 50000
for j in tqdm(range(int(np.ceil(test_df.shape[0]/50000)))):
    
    res.append(np.expm1((gbm_class.predict(test_df.iloc[i:i+step_size], num_iteration=gbm_class.best_iteration) > .5) *    (gbm_regress.predict(test_df.iloc[i:i+step_size], num_iteration=gbm_regress.best_iteration))))
    i+=step_size

    
del test_df
res = np.concatenate(res)
pd.DataFrame(res).describe()


# In[27]:


sample_submission["meter_reading"] = res
sample_submission.to_csv("submission.csv", index = False)

