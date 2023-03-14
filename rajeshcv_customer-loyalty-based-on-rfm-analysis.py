#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold
import warnings
import time
import sys
import datetime
from datetime import timedelta
import gc
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import mode
from functools import reduce
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)
import os
print(os.listdir("../input"))


# In[2]:


train = pd.read_csv("../input/elo-merchant-category-recommendation/train.csv")
test = pd.read_csv("../input/elo-merchant-category-recommendation/test.csv")
history =pd.read_csv("../input/elo-merchant-category-recommendation/historical_transactions.csv",parse_dates=['purchase_date'])
new =pd.read_csv("../input/elo-merchant-category-recommendation/new_merchant_transactions.csv",parse_dates=['purchase_date'])
cardreference = pd.read_csv("../input/feature-engineering-on-multiple-reference-dates/Cardreferencedate.csv",parse_dates=['reference_date'])


# In[3]:


history=history.loc[history.authorized_flag=="Y",]
history.purchase_amount += 0.75
new.purchase_amount += 0.75


# In[4]:


cardrfm = history.groupby('card_id').agg({'card_id': 'count','purchase_date': 'max','purchase_amount': 'sum'})
cardrfm.rename(columns={'card_id' : 'frequency','purchase_date': 'date_recency','purchase_amount': 'value'},inplace=True)
cardrfm = pd.merge(cardrfm,cardreference.iloc[:,0:2],on='card_id',how='left')
cardrfm['recency'] = cardrfm['reference_date'] - cardrfm['date_recency']
cardrfm.recency= cardrfm.recency/(24*np.timedelta64(1, 'h')) # to convert to day fractions
cardrfm.drop(columns=['date_recency','reference_date'],inplace=True)
cardrfm.head()


# In[5]:


print('Target value minimum',train.target.min())
print('Target value maximum',train.target.max())
print('Target value median',train.target.median())


# In[6]:


train.target.quantile(q=[0.011,0.05,0.25,0.5,0.75,0.95,0.989])


# In[7]:


quantiles = cardrfm.quantile(q=[0.011,0.05,0.25,0.5,0.75,0.95,0.989])
quantiles = quantiles.to_dict()
quantiles


# In[8]:


def RScore(x,p,d):
    if x <= d[p][0.011]:
        return 1
    elif x <= d[p][0.050]:
        return 2
    elif x <= d[p][0.25]: 
        return 3
    elif x <= d[p][0.5]:
        return 4
    elif x <= d[p][0.75]:
        return 5
    elif x <= d[p][0.95]:
        return 6
    elif x <= d[p][0.989]:
        return 7
    else:
        return 8
    
def FMScore(x,p,d):
    if x <= d[p][0.011]:
        return 8
    elif x <= d[p][0.050]:
        return 7
    elif x <= d[p][0.25]: 
        return 6
    elif x <= d[p][0.5]:
        return 5
    elif x <= d[p][0.75]:
        return 4
    elif x <= d[p][0.95]:
        return 3
    elif x <= d[p][0.989]:
        return 2
    else:
        return 1


# In[9]:


cardrfm['r_quantile'] = cardrfm['recency'].apply(RScore, args=('recency',quantiles))
cardrfm['f_quantile'] = cardrfm['frequency'].apply(FMScore, args=('frequency',quantiles))
cardrfm['v_quantile'] = cardrfm['value'].apply(FMScore, args=('value',quantiles))
cardrfm['RFMindex'] = cardrfm.r_quantile.map(str)+cardrfm.f_quantile.map(str)+cardrfm.v_quantile.map(str)                       
cardrfm['RFMScore'] = cardrfm.r_quantile+cardrfm.f_quantile+cardrfm.v_quantile 
cardrfm.head()


# In[10]:


cardrfm.RFMindex= cardrfm.RFMindex.astype(int)
RFMindex=pd.DataFrame(np.unique(np.sort(cardrfm.RFMindex)),columns=['RFMindex'])
RFMindex.index=RFMindex.index.set_names(['RFMIndex'])
RFMindex.reset_index(inplace=True)
cardrfm =pd.merge(cardrfm,RFMindex,on='RFMindex',how='left')
cardrfm.drop(columns="RFMindex",inplace=True) 
cardrfm.head()


# In[11]:


cardrfm_new = new.groupby('card_id').agg({'card_id': 'count','purchase_date': 'max','purchase_amount': 'sum'})
cardrfm_new.rename(columns={'card_id' : 'frequency_new','purchase_date': 'date_recency','purchase_amount': 'value_new'},inplace=True)
cardrfm_new = pd.merge(cardrfm_new,cardreference.iloc[:,0:2],on='card_id',how='left')
cardrfm_new['recency_new'] = cardrfm_new['reference_date'] - cardrfm_new['date_recency'] + datetime.timedelta(days=61)
cardrfm_new.recency_new= cardrfm_new.recency_new/(24*np.timedelta64(1, 'h')) # to convert to day fractions
cardrfm_new.drop(columns=['date_recency','reference_date'],inplace=True)
newquantiles = cardrfm_new.quantile(q=[0.011,0.05,0.25,0.5,0.75,0.95,0.989])
newquantiles = newquantiles.to_dict()
newquantiles


# In[12]:


cardrfm_new['rnew_quantile'] = cardrfm_new['recency_new'].apply(RScore, args=('recency_new',newquantiles))
cardrfm_new['fnew_quantile'] = cardrfm_new['frequency_new'].apply(FMScore, args=('frequency_new',newquantiles))
cardrfm_new['vnew_quantile'] = cardrfm_new['value_new'].apply(FMScore, args=('value_new',newquantiles))
cardrfm_new['RFMnewindex'] = cardrfm_new.rnew_quantile.map(str)+cardrfm_new.fnew_quantile.map(str)+cardrfm_new.vnew_quantile.map(str)                       
cardrfm_new['RFMnewScore'] = cardrfm_new.rnew_quantile+cardrfm_new.fnew_quantile+cardrfm_new.vnew_quantile 
cardrfm_new.head()


# In[13]:


cardrfm_new.RFMnewindex= cardrfm_new.RFMnewindex.astype(int)
RFMnewindex=pd.DataFrame(np.unique(np.sort(cardrfm_new.RFMnewindex)),columns=['RFMnewindex'])
RFMnewindex.index=RFMnewindex.index.set_names(['RFMnewIndex'])
RFMnewindex.reset_index(inplace=True)
cardrfm_new =pd.merge(cardrfm_new,RFMnewindex,on='RFMnewindex',how='left')
cardrfm_new.drop(columns="RFMnewindex",inplace=True) 
cardrfm_new.head()


# In[14]:


for df in [train,test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['weekday'] = df['first_active_month'].dt.weekday
    df['feature_comb'] = df.feature_1.map(str) + df.feature_2.map(str) + df.feature_3.map(str)
    df['feature_comb']= df['feature_comb'].astype(int)


# In[15]:


featureindex=pd.DataFrame(np.unique(np.sort(train['feature_comb'])),columns=['feature_comb'])
featureindex.index=featureindex.index.set_names(['feature_comb_index'])
featureindex.reset_index(inplace=True)
train =pd.merge(train,featureindex,on='feature_comb',how='left')
train.drop(columns="feature_comb",inplace=True) 
test =pd.merge(test,featureindex,on='feature_comb',how='left')
test.drop(columns="feature_comb",inplace=True)
test.head()


# In[16]:


train_df= pd.merge(train,cardrfm,on='card_id',how='left')
train_df= pd.merge(train_df,cardrfm_new,on='card_id',how='left')
train_df= pd.merge(train_df,cardreference,on='card_id',how='left')
train_df['frequency_new_hist'] =train_df.frequency_new/train_df.frequency
train_df['value_new_hist'] =train_df.value_new/train_df.value
train_df['recency_new_hist'] =train_df.recency_new/train_df.recency
train_df['elapsedtime']= (train_df['reference_date'] - train_df['first_active_month']).dt.days
test_df= pd.merge(test,cardrfm,on='card_id',how='left')
test_df= pd.merge(test_df,cardrfm_new,on='card_id',how='left')
test_df= pd.merge(test_df,cardreference,on='card_id',how='left')
test_df['frequency_new_hist'] =test_df.frequency_new/test_df.frequency
test_df['value_new_hist'] =test_df.value_new/test_df.value
test_df['recency_new_hist'] =test_df.recency_new/test_df.recency
test_df['elapsedtime']= (test_df['reference_date'] - test_df['first_active_month']).dt.days
train_df['first_active_month'] = pd.DatetimeIndex(train_df['first_active_month']).                                      astype(np.int64) * 1e-9
train_df['reference_date'] = pd.DatetimeIndex(train_df['reference_date']).                                      astype(np.int64) * 1e-9
test_df['first_active_month'] = pd.DatetimeIndex(test_df['first_active_month']).                                      astype(np.int64) * 1e-9
test_df['reference_date'] = pd.DatetimeIndex(test_df['reference_date']).                                    astype(np.int64) * 1e-9
test_df.head()


# In[17]:


train_df.to_csv("trainrfm.csv",index=False)
test_df.to_csv("testrfm.csv",index=False)


# In[18]:


target = train_df.target
train_df= train_df.drop(['card_id','target'],axis=1)
card_id = test_df['card_id']
test_df= test_df.drop(['card_id'],axis=1)


# In[19]:


features = [c for c in train_df.columns if c not in ['card_id','target']]
categorical_feats = ['feature_1','feature_2','feature_3','feature_comb_index','year', 'month','weekday','category_month_lag','r_quantile','f_quantile','v_quantile','RFMScore','RFMIndex','RFMIndex','rnew_quantile','fnew_quantile','vnew_quantile','RFMnewScore','RFMnewIndex','RFMIndex']


# In[20]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof1 = np.zeros(len(train_df))
predictions1 = np.zeros(len(test_df))
start = time.time()
feature_importance_df1 = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values,train_df['RFMScore'].values)):
     print("fold n°{}".format(fold_))
     trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
     val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

     num_round = 10000
#     params= param
     params ={
                 'task': 'train',
                 'boosting': 'goss',
                 'objective': 'regression',
                 'metric': 'rmse',
                 'learning_rate': 0.01,
                 'subsample': 0.9855232997390695,
                 'max_depth': 7,
                 'top_rate': 0.9064148448434349,
                 'num_leaves': 63,
                 'min_child_weight': 41.9612869171337,
                 'other_rate': 0.0721768246018207,
                 'reg_alpha': 9.677537745007898,
                 'colsample_bytree': 0.5665320670155495,
                 'min_split_gain': 9.820197773625843,
                 'reg_lambda': 8.2532317400459,
                 'min_data_in_leaf': 21,
                 'verbose': -1,
                 'seed':int(2**fold_),
                 'bagging_seed':int(2**fold_),
                 'drop_seed':int(2**fold_)}
     
    
     clf1 = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)
     oof1[val_idx] = clf1.predict(train_df.iloc[val_idx][features], num_iteration=clf1.best_iteration)
    
     fold_importance_df1 = pd.DataFrame()
     fold_importance_df1["feature"] = features
     fold_importance_df1["importance"] = clf1.feature_importance()
     fold_importance_df1["fold"] = fold_ + 1
     feature_importance_df1 = pd.concat([feature_importance_df1, fold_importance_df1], axis=0)
    
     predictions1 += clf1.predict(test_df[features], num_iteration=clf1.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(mean_squared_error(oof1, target)**0.5))


# In[21]:


cols = (feature_importance_df1[["feature", "importance"]]
         .groupby("feature")
         .mean()
         .sort_values(by="importance", ascending=False)[:1000].index)

best_features1 = feature_importance_df1.loc[feature_importance_df1.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
             y="feature",
             data=best_features1.sort_values(by="importance",
                                            ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances1.png')

