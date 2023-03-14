#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss


# In[2]:


get_ipython().run_cell_magic('time', '', "df_train = pd.read_csv('../input/predicting-outliers-to-improve-your-score/train_clean.csv')\ndf_test = pd.read_csv('../input/predicting-outliers-to-improve-your-score/test_clean.csv')")


# In[3]:


df_train = df_train[df_train['outliers'] == 0]
target = df_train['target']
del df_train['target']
features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','outliers']]
categorical_feats = [c for c in features if 'feature_' in c]


# In[4]:


param = {'objective':'regression',
         'num_leaves': 31,
         'min_data_in_leaf': 25,
         'max_depth': 7,
         'learning_rate': 0.01,
         'lambda_l1':0.13,
         "boosting": "gbdt",
         "feature_fraction":0.85,
         'bagging_freq':8,
         "bagging_fraction": 0.9 ,
         "metric": 'rmse',
         "verbosity": -1,
         "random_state": 2333}


# In[5]:


get_ipython().run_cell_magic('time', '', 'folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)\noof = np.zeros(len(df_train))\npredictions = np.zeros(len(df_test))\nfeature_importance_df = pd.DataFrame()\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train[\'outliers\'].values)):\n    print("fold {}".format(fold_))\n    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)\n    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)\n\n    num_round = 10000\n    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval= 100, early_stopping_rounds = 200)\n    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n    \n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["Feature"] = features\n    fold_importance_df["importance"] = clf.feature_importance()\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    \n    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits\n\nprint("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))')


# In[6]:


model_without_outliers = pd.DataFrame({"card_id":df_test["card_id"].values})
model_without_outliers["target"] = predictions


# In[7]:


get_ipython().run_cell_magic('time', '', "df_train = pd.read_csv('../input/predicting-outliers-to-improve-your-score/train_clean.csv')\ndf_test = pd.read_csv('../input/predicting-outliers-to-improve-your-score/test_clean.csv')")


# In[8]:


target = df_train['outliers']
del df_train['outliers']
del df_train['target']


# In[9]:


features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]


# In[10]:


param = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 6,
         'learning_rate': 0.01,
         "boosting": "rf",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 2333}


# In[11]:


get_ipython().run_cell_magic('time', '', 'folds = KFold(n_splits=5, shuffle=True, random_state=15)\noof = np.zeros(len(df_train))\npredictions = np.zeros(len(df_test))\nfeature_importance_df = pd.DataFrame()\n\nstart = time.time()\n\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):\n    print("fold n°{}".format(fold_))\n    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)\n    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)\n\n    num_round = 10000\n    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)\n    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n    \n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["feature"] = features\n    fold_importance_df["importance"] = clf.feature_importance()\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    \n    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits\n\nprint("CV score: {:<8.5f}".format(log_loss(target, oof)))')


# In[12]:


### 'target' is the probability of whether an observation is an outlier
df_outlier_prob = pd.DataFrame({"card_id":df_test["card_id"].values})
df_outlier_prob["target"] = predictions
df_outlier_prob.head()


# In[13]:


# if the test set has the same ratio of outliers as training set, 
# then the numbuer of outliers in test is about: (1.06% outliers in training set)
123623*0.0106


# In[14]:


# In case missing some predictable outlier, we choose top 25000 with highest outliers likelyhood.
outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000)['card_id'])


# In[15]:


best_submission = pd.read_csv('../input/finaldata/submission_ashish.csv')


# In[16]:


most_likely_liers = best_submission.merge(outlier_id,how='right')
most_likely_liers.head()


# In[17]:


get_ipython().run_cell_magic('time', '', "for card_id in most_likely_liers['card_id']:\n    model_without_outliers.loc[model_without_outliers['card_id']==card_id,'target']\\\n    = most_likely_liers.loc[most_likely_liers['card_id']==card_id,'target'].values")


# In[18]:


model_without_outliers.to_csv("combining_submission.csv", index=False)

