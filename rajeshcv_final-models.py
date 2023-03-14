#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
import time
import sys
import datetime
from datetime import timedelta
import sklearn
from sklearn.metrics import mean_squared_error,auc,roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from scipy.stats.mstats import mode
import gc
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)

import os
print(os.listdir("../input"))


# In[2]:


train = pd.read_csv("../input/elo-merchant-category-recommendation/train.csv")
test = pd.read_csv("../input/elo-merchant-category-recommendation/test.csv")
train_df = pd.read_csv("../input/fork-of-stacked-lgbm-reduced-models/trainconsol.csv")
test_df = pd.read_csv("../input/fork-of-stacked-lgbm-reduced-models/testconsol.csv")
train_rfm = pd.read_csv("../input/customer-loyalty-based-on-rfm-analysis/trainrfm.csv")
test_rfm = pd.read_csv("../input/customer-loyalty-based-on-rfm-analysis/testrfm.csv")
feat_imp = pd.read_csv("../input/feature-importance-selection/scores_df.csv")


# In[3]:


lessimp_feature =feat_imp.loc[feat_imp.gain_score <=0,'feature'].tolist()
lessimp_feature.extend(['card_id','target','outliers','RFMIndex','RFMnewIndex','month_diff_mean','month_diff_min','month_diff_max','month_diff_var','month_diffnew_mean','month_diffnew_var'])
# lessimp_feature = ['feature_1', 'year', 'weekday', 'cat2_2.0_sum', 'cat2_2.0_mean', 'cat2_3.0_sum',
#                     'cat2_3.0_mean', 'cat2_5.0_sum', 'cat2_5.0_mean', 'cat3_A_mean', 'cattrans_1.0_sum',
#                      'month_lag_max', 'month_diff_min', 'time_diff_min', 'city_idnew_nunique',
#                      'state_idnew_nunique', 'merchant_category_idnew_nunique', 'cat2_1.0new_mean',
#                      'cat2_2.0new_sum', 'cat2_2.0new_mean', 'cat2_3.0new_sum', 'cat2_3.0new_mean',
#                      'cat2_4.0new_sum', 'cat3_Anew_sum', 'cat3_Cnew_sum', 'cat3_Cnew_mean',
#                      'month_diffnew_min', 'month_diffnew_max', 'time_diffnew_min',
#                      'weekendnew_sum', 'dayofweeknew_mode', 'mon_change_ratio_card_std',
#                      'f_quantile', 'v_quantile', 'fnew_quantile', 'RFMnewScore', 'card_id',
#                      'target', 'outliers', 'RFMIndex', 'RFMnewIndex', 'month_diff_mean', 'month_diff_min',
#                      'month_diff_max', 'month_diff_var', 'month_diffnew_mean', 'month_diffnew_var']
lessimp_feature


# In[4]:


# lessimp_feature =['year','weekday','cat2_2.0_sum','cat2_2.0_mean', 'cat2_3.0_sum', 'cat2_3.0_mean',
#  'cat2_5.0_sum','cat2_5.0_mean', 'cat3_A_sum', 'cat3_A_mean', 'cattrans_1.0_sum', 'installments_min',
#  'month_lag_max', 'time_diff_min', 'city_idnew_nunique', 'cat2_1.0new_mean', 'cat2_2.0new_sum',
#  'cat2_2.0new_mean', 'cat2_3.0new_sum', 'cat2_3.0new_mean', 'cat2_4.0new_sum', 'cat2_5.0new_sum',
#  'cat2_5.0new_mean','cat3_Anew_sum','cat3_Cnew_sum','installmentsnew_min', 'installmentsnew_max',
#  'weekendnew_sum', 'weekendnew_mean', 'mon_change_ratio_card_std', 'merchant_id_rep_nunique',
#  'rep_card_month_freq_rep_max', 'merchant_id_repmer_nunique', 'v_quantile', 'fnew_quantile', 'RFMnewScore']


# In[5]:


train_rfm.drop(columns=['first_active_month', 'feature_1', 'feature_2', 'feature_3',
       'target', 'year', 'month', 'weekday',  'frequency',
       'value', 'recency',  'frequency_new', 'value_new', 'recency_new',
        'reference_date', 'category_month_lag','elapsedtime'],inplace= True )
test_rfm.drop(columns=['first_active_month', 'feature_1', 'feature_2', 'feature_3',
        'year', 'month', 'weekday',  'frequency',
       'value', 'recency',  'frequency_new', 'value_new', 'recency_new',
        'reference_date', 'category_month_lag','elapsedtime'],inplace= True )


# In[6]:


train_df.head()


# In[7]:


train_df= pd.merge(train_df,train_rfm,on='card_id',how='left')
test_df= pd.merge(test_df,test_rfm,on='card_id',how='left')


# In[8]:


# for df in [train_df,test_df]:
#     df['days_feature1'] = df['elapsedtime'] * df['feature_1']
#     df['days_feature2'] = df['elapsedtime'] * df['feature_2']
#     df['days_feature3'] = df['elapsedtime'] * df['feature_3']
#     df['days_feature1_ratio'] = df['feature_1'] / df['elapsedtime']
#     df['days_feature2_ratio'] = df['feature_2'] / df['elapsedtime']
#     df['days_feature3_ratio'] = df['feature_3'] / df['elapsedtime']


# In[9]:


target = train_df.target
train_df= train_df.drop(['card_id','target'],axis=1)
card_id = test_df['card_id']
test_df= test_df.drop(['card_id'],axis=1)


# In[10]:


train_df.head()


# In[11]:


features = [c for c in train_df.columns if c not in lessimp_feature ]
categorical_feats = ['feature_2','feature_1','feature_comb_index', 'month','category_month_lag','f_quantile','r_quantile','RFMScore','rnew_quantile','vnew_quantile' ]


# In[12]:


train_df[features].info()


# In[13]:


param = {
    'objective':'regression',
    'learning_rate': 0.01,
    'boosting': 'gbdt',
    'num_leaves': 31,
    'colsample_bytree': 0.5178284844630764,
    'subsample': 0.6701600397051216,
    'max_depth': 10,
    'reg_alpha': 2.5056142297682493,
    'reg_lambda': 2.4364948978876253,
    'min_split_gain': 4.947802532689812,
    'min_child_weight': 43.41313044141597,
    'min_data_in_leaf': 40,
      "metric": 'rmse',
    'bagging_seed':11,
      "verbosity": -1
}

param1= {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': 10,
         'learning_rate': 0.005,
         "min_child_samples": 100,
         "boosting": "gbdt",
         "metric": 'rmse',
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "lambda_l1": 0.1,
         "verbosity": -1
           }


# In[14]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values,train_df['feature_comb_index'].values)):
     print("fold n°{}".format(fold_))
     trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
     val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

     num_round = 10000
     params= param
     clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)
     oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
     fold_importance_df = pd.DataFrame()
     fold_importance_df["feature"] = features
     fold_importance_df["importance"] = clf.feature_importance()
     fold_importance_df["fold"] = fold_ + 1
     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
     predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))


# In[15]:


cols = (feature_importance_df[["feature", "importance"]]
         .groupby("feature")
         .mean()
         .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
             y="feature",
             data=best_features.sort_values(by="importance",
                                            ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[16]:


print(min(predictions))
print(max(predictions))


# In[17]:


sns.set(rc={'figure.figsize':(30,12)})
sns.scatterplot(x=target,y=oof)


# In[18]:


submit = pd.DataFrame({'card_id':card_id,'target':predictions})
submit.to_csv("featurecombLGBM1.csv",index=False)


# In[19]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof1 = np.zeros(len(train_df))
predictions1 = np.zeros(len(test_df))
start = time.time()
feature_importance_df1 = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values,train_df['category_month_lag'].values)):
     print("fold n°{}".format(fold_))
     trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
     val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

     num_round = 10000
     params= param
#      params ={
#                  'task': 'train',
#                  'boosting': 'goss',
#                  'objective': 'regression',
#                  'metric': 'rmse',
#                  'learning_rate': 0.01,
#                  'subsample': 0.9855232997390695,
#                  'max_depth': 7,
#                  'top_rate': 0.9064148448434349,
#                  'num_leaves': 63,
#                  'min_child_weight': 41.9612869171337,
#                  'other_rate': 0.0721768246018207,
#                  'reg_alpha': 9.677537745007898,
#                  'colsample_bytree': 0.5665320670155495,
#                  'min_split_gain': 9.820197773625843,
#                  'reg_lambda': 8.2532317400459,
#                  'min_data_in_leaf': 21,
#                  'verbose': -1,
#                  'seed':int(2**fold_),
#                  'bagging_seed':int(2**fold_),
#                  'drop_seed':int(2**fold_)}
     
    
     clf1 = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)
     oof1[val_idx] = clf1.predict(train_df.iloc[val_idx][features], num_iteration=clf1.best_iteration)
    
     fold_importance_df1 = pd.DataFrame()
     fold_importance_df1["feature"] = features
     fold_importance_df1["importance"] = clf1.feature_importance()
     fold_importance_df1["fold"] = fold_ + 1
     feature_importance_df1 = pd.concat([feature_importance_df1, fold_importance_df1], axis=0)
    
     predictions1 += clf1.predict(test_df[features], num_iteration=clf1.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(mean_squared_error(oof1, target)**0.5))


# In[20]:


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


# In[21]:


print(min(predictions1))
print(max(predictions1))


# In[22]:


submissioncatmonth = pd.read_csv('../input/elo-merchant-category-recommendation/sample_submission.csv')
submissioncatmonth['target'] = predictions1
submissioncatmonth.to_csv('LGBMcategorymonthlag1.csv', index=False)


# In[23]:


sns.set(rc={'figure.figsize':(30,12)})
sns.scatterplot(x=target,y=oof1)


# In[24]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof2 = np.zeros(len(train_df))
predictions2 = np.zeros(len(test_df))
start = time.time()
feature_importance_df2 = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values,train_df['RFMScore'].values)):
     print("fold n°{}".format(fold_))
     trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
     val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

     num_round = 10000
     params= param
#      params ={
#                  'task': 'train',
#                  'boosting': 'goss',
#                  'objective': 'regression',
#                  'metric': 'rmse',
#                  'learning_rate': 0.01,
#                  'subsample': 0.9855232997390695,
#                  'max_depth': 7,
#                  'top_rate': 0.9064148448434349,
#                  'num_leaves': 63,
#                  'min_child_weight': 41.9612869171337,
#                  'other_rate': 0.0721768246018207,
#                  'reg_alpha': 9.677537745007898,
#                  'colsample_bytree': 0.5665320670155495,
#                  'min_split_gain': 9.820197773625843,
#                  'reg_lambda': 8.2532317400459,
#                  'min_data_in_leaf': 21,
#                  'verbose': -1,
#                  'seed':int(2**fold_),
#                  'bagging_seed':int(2**fold_),
#                  'drop_seed':int(2**fold_)}
     
    
     clf2 = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)
     oof2[val_idx] = clf2.predict(train_df.iloc[val_idx][features], num_iteration=clf2.best_iteration)
    
     fold_importance_df2 = pd.DataFrame()
     fold_importance_df2["feature"] = features
     fold_importance_df2["importance"] = clf2.feature_importance()
     fold_importance_df2["fold"] = fold_ + 1
     feature_importance_df2 = pd.concat([feature_importance_df2, fold_importance_df2], axis=0)
    
     predictions2 += clf2.predict(test_df[features], num_iteration=clf2.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(mean_squared_error(oof2, target)**0.5))


# In[25]:


cols = (feature_importance_df2[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features2 = feature_importance_df2.loc[feature_importance_df2.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features2.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances2.png')


# In[26]:


print(min(predictions2))
print(max(predictions2))


# In[27]:


submissionrfm = pd.read_csv('../input/elo-merchant-category-recommendation/sample_submission.csv')
submissionrfm['target'] = predictions2
submissionrfm.to_csv('LGBMrfmscore1.csv', index=False)


# In[28]:


sns.set(rc={'figure.figsize':(30,12)})
sns.scatterplot(x=target,y=oof2)


# In[29]:


folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=15)
oof3 = np.zeros(len(train_df))
predictions3 = np.zeros(len(test_df))
start = time.time()
feature_importance_df3 = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values,target.values)):
     print("fold n°{}".format(fold_))
     trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
     val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

     num_round = 10000
     params= param
#      params ={
#                  'task': 'train',
#                  'boosting': 'goss',
#                  'objective': 'regression',
#                  'metric': 'rmse',
#                  'learning_rate': 0.01,
#                  'subsample': 0.9855232997390695,
#                  'max_depth': 7,
#                  'top_rate': 0.9064148448434349,
#                  'num_leaves': 63,
#                  'min_child_weight': 41.9612869171337,
#                  'other_rate': 0.0721768246018207,
#                  'reg_alpha': 9.677537745007898,
#                  'colsample_bytree': 0.5665320670155495,
#                  'min_split_gain': 9.820197773625843,
#                  'reg_lambda': 8.2532317400459,
#                  'min_data_in_leaf': 21,
#                  'verbose': -1,
#                  'seed':int(2**fold_),
#                  'bagging_seed':int(2**fold_),
#                  'drop_seed':int(2**fold_)}
     
    
     clf3 = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)
     oof3[val_idx] = clf3.predict(train_df.iloc[val_idx][features], num_iteration=clf3.best_iteration)
    
     fold_importance_df3 = pd.DataFrame()
     fold_importance_df3["feature"] = features
     fold_importance_df3["importance"] = clf3.feature_importance()
     fold_importance_df3["fold"] = fold_ + 1
     feature_importance_df3 = pd.concat([feature_importance_df3, fold_importance_df3], axis=0)
    
     predictions3 += clf3.predict(test_df[features], num_iteration=clf3.best_iteration) / (5 *  2)


print("CV score: {:<8.5f}".format(mean_squared_error(oof3, target)**0.5))


# In[30]:


cols = (feature_importance_df3[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features3 = feature_importance_df3.loc[feature_importance_df3.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features3.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances3.png')


# In[31]:


submissiontarget = pd.read_csv('../input/elo-merchant-category-recommendation/sample_submission.csv')
submissiontarget['target'] = predictions3
submissiontarget.to_csv('LGBMfinal1.csv', index=False)


# In[32]:


print(min(predictions3))
print(max(predictions3))


# In[33]:


sns.set(rc={'figure.figsize':(30,12)})
sns.scatterplot(x=target,y=oof3)


# In[34]:


oof.to_csv("oof.csv",index=False)
oof1.to_csv("oof1.csv",index=False)
oof2.to_csv("oof2.csv",index=False)
oof3.to_csv("oof3.csv",index=False)


# In[35]:


from sklearn.linear_model import BayesianRidge

train_stack = np.vstack([oof,oof1,oof2,oof3]).transpose()
test_stack = np.vstack([predictions, predictions1, predictions2,predictions3]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=1, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions4 = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
    
    clf4 = BayesianRidge()
    clf4.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf4.predict(val_data)
    predictions4 += clf4.predict(test_stack) / 5
    
np.sqrt(mean_squared_error(target.values, oof_stack))


# In[36]:


sample_submission = pd.read_csv('../input/elo-merchant-category-recommendation/sample_submission.csv')
sample_submission['target'] = predictions4
sample_submission.to_csv('LGBMstacksubmissionfinal1.csv', index=False)


# In[37]:


print(min(predictions4))
print(max(predictions4))


# In[38]:


sns.set(rc={'figure.figsize':(30,12)})
sns.scatterplot(x=target,y=oof_stack)

