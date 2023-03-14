#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# setup
# -------------------------------------------------------------
# system
import os
# -------------------------------------------------------------
# fundamental modules
import operator
import pandas as pd
import numpy as np
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# ds tools
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# models
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import mxnet as mx

# for gini
from numba import jit

# -------------------------------------------------------------
import gc
import logging
import sys
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

# =================================================================================
# custom objective function (similar to auc)
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_nn(y, pred):
    return gini(y, pred) / gini(y, y)

# -------------------------------------------------------------
# for sklearn api
@jit
def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_xgb_sklearn_api(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return [('gini', gini_score)]


# In[2]:


# read in files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# basic understanding of the dataset
print ("Dimension of train data {}".format(train.shape))
print ("Dimension of test data {}".format(test.shape))

====================================================================================
Work with features
credit to:
Heads or Tails's Steering Wheel of Fortune - Porto Seguro EDA notebook
Cam Nugent's deep neural network: insurance claims (~0.268)
# In[3]:


# list columns that have missing values
for col in train.columns:
    null_sum = (train[col] == -1).sum()
    if null_sum > 0:
        print('feature %s has %i missing entries' %(col, null_sum))


# In[4]:


# olivier's feature selection and 
feat_olivier_peatle = [
    "ps_car_13",    # xgb #1  lgb #1
    "ps_reg_03",    # xgb #2  lgb #2
    "ps_ind_03",    # xgb #3  lgb #3
    "ps_car_14",    # xgb #4  lgb #5
    "ps_ind_15",    # xgb #5  lgb #4
    "ps_reg_02",    # xgb #6  lgb #8
    "ps_ind_05_cat",# xgb #7  lgb #6
    "ps_ind_01",    # xgb #8  lgb #9
    "ps_car_11_cat",# xgb #9  lgb #10
    "ps_car_01_cat",# xgb #10 lgb #11
    "ps_reg_01",    # xgb #11 lgb #7
    "ps_car_12",    # xgb #12 lgb #15
    "ps_car_15",    # xgb #13 lgb #12
    "ps_car_06_cat",# xgb #14 lgb #13
    "ps_car_09_cat",# xgb #15 lgb #19
    "ps_ind_02_cat",# xgb #16 lgb #18
    
    "ps_car_07_cat",# xgb #18 lgb #16
    "ps_car_11",    # xgb #19 lgb #20
    "ps_car_03_cat",# xgb #20 lgb #17
    "ps_ind_04_cat",# xgb #21 lgb #24
    "ps_car_04_cat",# xgb #22 lgb #25

    
    'ps_car_05_cat',# xgb #27 lgb #27
    
    "ps_car_02_cat",# xgb #29 lgb #29
    "ps_car_08_cat",# xgb #30 lgb #30
    
    
    'ps_car_10_cat',# xgb #33 lgb #34
    
    

    'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', # these three sums to ps_ind_14, so ps_ind_14 removed
    'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', # these four sums to 1
    'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', #If one of ps_ind_16_bin, ps_ind_17_bin and ps_ind_18_bin is 1, then the others are 0, but they can all be 0
]


# In[5]:


# data preparation for training

# drop id and target column
train_y = train['target']
train_x = train.drop(['id', 'target'], axis = 1)

# prepare testing dataset
test_x = test.drop(['id'], axis = 1)#test[feat_olivier_peatle]


# In[6]:


merged_dat = pd.concat([train_x, test_x], axis = 0)
print(merged_dat.shape)


# In[7]:


# add feature "https://www.kaggle.com/headsortails/steering-wheel-of-fortune-porto-seguro-eda"
# Number of NAs per ID
merged_dat['nona_calc_'] = (merged_dat == -1).sum(axis = 1)
# Sum of binary features
clmn_fltr = [col for col in merged_dat.columns if 'ind' in col and 'bin' in col]
merged_dat['sumbin_calc_'] = merged_dat[clmn_fltr].sum(axis = 1)
# Difference measure for binary features
fltr_df = merged_dat[clmn_fltr]
diff_df = fltr_df.apply(lambda x: x - fltr_df.iloc[0, :], axis = 1).fillna(0)
merged_dat['diffbin_calc_'] = diff_df.sum(axis = 1)


# In[8]:


merged_dat = merged_dat[feat_olivier_peatle + ['nona_calc_', 'sumbin_calc_', 'diffbin_calc_']]


# In[9]:


#change data to float32
for c, dtype in zip(merged_dat.columns, merged_dat.dtypes): 
    if dtype == np.float64:     
        merged_dat[c] = merged_dat[c].astype(np.float32)


# In[10]:


#one hot encode the categoricals
cat_features = [col for col in merged_dat.columns if col.endswith('cat')]
# count ind (no cat nor bin) column as categoricals
cat_features = cat_features +                [col for col in merged_dat.columns if 'ind' in col and 'cat' not in col and 'bin' not in col]
for column in cat_features:
    temp = pd.get_dummies(pd.Series(merged_dat[column]))
    merged_dat = pd.concat([merged_dat, temp],axis=1)
    merged_dat.drop([column], axis=1, inplace = True)


# In[11]:


merged_dat.replace(-1, 0, inplace = True)


# In[12]:


# normailise the scale of the numericals
scaler = MinMaxScaler()
merged_dat = scaler.fit_transform(merged_dat)


# In[13]:


train_X = merged_dat[:train_x.shape[0]]
test_X = merged_dat[train_x.shape[0]:]


# In[14]:


# ------------------------------------------------
gc.collect()

# process data
X = train_X
y = train_y.values

# k fold training
k = 5
skf = StratifiedKFold(n_splits = k, random_state = 0, shuffle  = True)

# ------------------------------------------------
# setup neural mlp model
def get_mlp(n_in):
    """
    multi-layer perceptron
    """
    n_out = 1
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden = n_in)
    act1 = mx.symbol.Activation(data = fc1, name = 'relu1', act_type = "relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 3 * int((n_in - n_out) / 4 // 1 + n_out))
    act2 = mx.symbol.Activation(data = fc2, name = 'relu2', act_type = "relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 2 * int((n_in - n_out) / 4 // 1 + n_out))
    act3 = mx.symbol.Activation(data = fc3, name = 'relu3', act_type = "relu")
    fc4  = mx.symbol.FullyConnected(data = act3, name = 'fc4', num_hidden = int((n_in - n_out) / 4 // 1 + n_out))
    act4 = mx.symbol.Activation(data = fc4, name = 'relu4', act_type = "relu")
    fc5  = mx.symbol.FullyConnected(data = act4, name = 'fc5', num_hidden = n_out)
    mlp  = mx.symbol.LogisticRegressionOutput(data = fc5, name = 'softmax')
    return mlp

optimizer_params = {'learning_rate': 0.1, 
                    'momentum' : 0.9,
                    'wd' : 0.0001, 
                    'lr_scheduler': mx.lr_scheduler.FactorScheduler(step = 1000, factor = 0.5)}

gini_metric = mx.metric.create(gini_nn)

print('---training using multi layer perception---')
for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print('=====================================================')
    print('kfold learning: {}  of  {} : '.format(i + 1, k))
    
    # process data
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    # apply upsampling using olivier's approach: XGB classifier, upsampling LB 0.283
    upsampling = False
    if upsampling:
        X_train = np.append(X_train, X_train[np.where(y_train == 1)[0], :], axis = 0)
        y_train = np.append(y_train, y_train[np.where(y_train == 1)[0]], axis = 0)
        X_train, y_train = shuffle(X_train, y_train)
    
    # mxnet ndarray
    mx_train = mx.io.NDArrayIter(X_train, y_train, batch_size = 50000)
    mx_valid = mx.io.NDArrayIter(X_valid, y_valid, batch_size = 50000)
    #---------------------------------
    model_mlp = mx.mod.Module(symbol = get_mlp(X.shape[1]), context = mx.gpu())
    model_mlp.fit(mx_train,
                  num_epoch          = 400,
                  optimizer          = 'sgd',
                  eval_data          = mx_valid,
                  eval_metric        = gini_metric,
                  optimizer_params   = optimizer_params,
                  initializer        = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
                  epoch_end_callback = mx.callback.do_checkpoint('fold_' + str(i), 10),
                 )


# In[15]:


# collate results
fold_pred_collator = pd.DataFrame()

# setup prediction
best_epoch = [170, 150, 120, 150, 150]
for i in range(0, 5):
    print('generating prediction using fold # %i model.' %i)
    sym, arg_params, aux_params = mx.model.load_checkpoint('fold_' + str(i), best_epoch[i])
    mdl_pred = mx.mod.Module(symbol = sym, context = mx.gpu(), label_names = None)
    mdl_pred.bind(for_training = False, data_shapes = [('data', test_X.shape)], label_shapes = None)
    mdl_pred.set_params(arg_params, aux_params, allow_missing=True)
    mdl_pred.forward(Batch([mx.nd.array(test_X)]))
    fold_pred_collator['fold_' + str(i)] = mdl_pred.get_outputs()[0].asnumpy().squeeze()
    


# In[16]:


sbmtn_result_df = test['id'].to_frame()
sbmtn_result_df['target'] = fold_pred_collator.mean(axis = 1)


# In[17]:


sbmtn_result_df.to_csv('nn_submission.csv', index = False)

