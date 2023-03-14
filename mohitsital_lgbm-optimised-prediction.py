#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


train = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv',na_values="-1")
test= pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv', na_values="-1")
sub = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/sample_submission.csv')
print(train.shape)
print(test.shape)


# In[3]:


print(train.columns.difference(test.columns))


# In[4]:


print(train.isnull().sum().sum())
print(test.isnull().sum().sum())


# In[5]:


train['target'].value_counts()


# In[6]:


train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)
print('done')


# In[7]:


test_id = test['id']
test.drop('id',axis=1,inplace=True)
train.drop('id',axis=1,inplace=True)
print('done')


# In[8]:


feature_names = train.columns.tolist()[1:]
cat_features = [c for c in feature_names if ('cat' in c and 'count' not in c)]
num_features = [c for c in feature_names if ('cat' not in c and 'calc' not in c)]
print('done')


# In[9]:


cat_count_features = []
for c in cat_features+['new_ind']:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)
    
print('done')


# In[10]:


#train[cat_features] = train[cat_features].isnull().sum()


# In[11]:



#unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
#print(unwanted)
#train = train.drop(unwanted, axis=1)  
#test = test.drop(unwanted, axis=1)  
#print(train.shape)
#print(test.shape)


# In[12]:


target = 'target'
predictors=['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin',
 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14',
 'ps_ind_15', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_01_cat',
 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15',
 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'missing', 'ps_ind_02_cat_count', 'ps_ind_04_cat_count', 'ps_ind_05_cat_count',
 'ps_car_01_cat_count', 'ps_car_02_cat_count', 'ps_car_03_cat_count', 'ps_car_04_cat_count', 'ps_car_05_cat_count',
 'ps_car_06_cat_count', 'ps_car_07_cat_count', 'ps_car_08_cat_count', 'ps_car_09_cat_count', 'ps_car_10_cat_count',
 'ps_car_11_cat_count', 'new_ind_count']


# In[13]:


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True


# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import pickle
import os
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
gc.enable()


bayesian_tr_index, bayesian_val_index  = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(train,train.target.values))[0]

def LGB_bayesian(
     num_leaves,  # int
     min_data_in_leaf,  # int
     learning_rate,
     min_sum_hessian_in_leaf,    # int  
     feature_fraction,
     lambda_l1,
     lambda_l2,
     min_gain_to_split,
     max_depth):
    
     # LightGBM expects next three parameters need to be integer. So we make them integer
     num_leaves = int(round(num_leaves))
     min_data_in_leaf = int(round(min_data_in_leaf))
     max_depth = int(round(max_depth))

     assert type(num_leaves) == int
     assert type(min_data_in_leaf) == int
     assert type(max_depth) == int

     param = {
         'num_leaves': num_leaves,
         'max_bin': 63,
         'min_data_in_leaf': min_data_in_leaf,
         'learning_rate': learning_rate,
         'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
         'bagging_fraction': 1.0,
         'bagging_freq': 5,
         'feature_fraction': feature_fraction,
         'lambda_l1': lambda_l1,
         'lambda_l2': lambda_l2,
         'min_gain_to_split': min_gain_to_split,
         'max_depth': max_depth,
         'save_binary': True, 
         'seed': 1337,
         'feature_fraction_seed': 1337,
         'bagging_seed': 1337,
         'drop_seed': 1337,
         'data_random_seed': 1337,
         'objective': 'binary',
         'boosting_type': 'gbdt',
         'verbose': 1,
         'metric': 'auc',
         'is_unbalance': True,
         'boost_from_average': False,   

     }    
    
    
     xg_train = lgb.Dataset(train.iloc[bayesian_tr_index][predictors].values,
                            label=train.iloc[bayesian_tr_index][target].values,
                            feature_name=predictors,
                            free_raw_data = False
                            )
     xg_valid = lgb.Dataset(train.iloc[bayesian_val_index][predictors].values,
                            label=train.iloc[bayesian_val_index][target].values,
                            feature_name=predictors,
                            free_raw_data = False
                            )   

     num_round = 5000
     clf = lgb.train(param, xg_train, num_round, valid_sets = [xg_valid],feval=evalerror, verbose_eval=50,early_stopping_rounds = 50)
    
     predictions = clf.predict(train.iloc[bayesian_val_index][predictors].values, num_iteration=clf.best_iteration)   
    
     score = Gini(train.iloc[bayesian_val_index][target].values, predictions)
    
     return score


# In[15]:


# # Bounded region of parameter space
bounds_LGB = {
     'num_leaves': (2, 5), 
     'min_data_in_leaf': (1, 10),  
     'learning_rate': (0.03, 0.07),
     'min_sum_hessian_in_leaf': (0.1, 0.5),    
     'feature_fraction': (0.3, 0.7),
     'lambda_l1': (0, 1), 
     'lambda_l2': (0, 1), 
     'min_gain_to_split': (0.1, 1.0),
     'max_depth':(2,10),
 }

from bayes_opt import BayesianOptimization

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)

print(LGB_BO.space.keys)

init_points = 10
n_iter = 10

target = 'target'
#predictors = train.columns.values.tolist()[1:]

print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


# In[16]:


LGB_BO.max   


# In[17]:


param_lgb = {
    'max_bin': 63,
    'bagging_fraction': 1.0,
    'bagging_freq': 5,
    'feature_fraction': 0.557897278957675,
    'lambda_l1': 0.9962765743947803,
    'lambda_l2': 0.9167636072131916,
    'learning_rate': 0.05183596549633812,
    'max_depth': int(5.23946365804394),
    'min_data_in_leaf': int(9.214731014239497),
    'min_gain_to_split': 0.7987771215261199,
    'min_sum_hessian_in_leaf': 0.11869234058908429,
    'num_leaves': int(4.686698789103291),   
    'save_binary': True, 
    'seed': 1337,
    'feature_fraction_seed': 1337,
    'bagging_seed': 1337,
    'drop_seed': 1337,
    'data_random_seed': 1337,
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'verbose': 1,
    'metric': 'auc',
    'is_unbalance': True,
    'boost_from_average': False
    
}


# In[18]:


nfold = 5

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

oof = np.zeros(len(X))
predictions = np.zeros((len(test),nfold))

i = 1
for train_index, valid_index in skf.split(train, train.target.values):
    print("\nfold {}".format(i))

    xg_train = lgb.Dataset(train.iloc[train_index][predictors].values,
                           label=train.iloc[train_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )
    xg_valid = lgb.Dataset(train.iloc[valid_index][predictors].values,
                           label=train.iloc[valid_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )   

    
    clf = lgb.train(param_lgb, xg_train, 10000000, valid_sets = [xg_valid],feval=evalerror, verbose_eval=250, early_stopping_rounds = 100)
    oof[valid_index] = clf.predict(train.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) 
    
    predictions[:,i-1] += clf.predict(test[predictors].values, num_iteration=clf.best_iteration)
    i = i + 1

print("\n\nCV GINI: {:<0.8f}".format(Gini(train.target.values, oof)))


# In[19]:


lgb_bay = []

for i in range(len(predictions)):
    lgb_bay.append(predictions[i][-1])


# In[20]:


sub['target'] = lgb_bay
sub.to_csv('sub6.csv', index = False, header = True)

