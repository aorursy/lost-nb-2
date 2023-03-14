#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install gpyopt')


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/1-stats-k3"))

# Any results you write to the current directory are saved as output.


# In[3]:


import time
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
# import GPyOpt
from imblearn.over_sampling import ADASYN
from collections import Counter
import lightgbm as lgb
from datetime import datetime
from numba import jit

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import GPyOpt


# In[4]:


# Read in features from GitHub
train_data = pd.read_csv('../input/1-stats-k3/train_milos.csv')
test_data = pd.read_csv('../input/1-stats-k3/test_milos.csv')
sample_submission = pd.read_csv('../input/1-stats-k3/sample_submission.csv')

print('Training data shape: ', train_data.shape)
print('Testing data shape:  ', test_data.shape)


# In[5]:


train_data.describe()


# In[6]:


# Convert to numpy arrays
original_features  = train_data.drop(columns=['ID_code', 'target'])
testing_features = test_data.drop(columns=['ID_code'])

# Sklearn wants the labels as one-dimensional vectors
original_targets  = train_data.filter(items=['target'])


# In[7]:


## Inspiration from
#https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment
def augment(train, target, num_n=1, num_p=2):
    newtrain=[train]
    newtarget=[target]
    
    n=train[target==0]
    z=target[target==0]
    for i in range(num_n):
        random_state = np.random.permutation(len(n))
        newtrain.append(n.apply(lambda x:x.values.take(random_state)))
        newtarget.append(z.apply(lambda x:x.values.take(random_state)))
    
    for i in range(num_p):
        p=train[target>0]
        o=target[target>0]
        random_state = np.random.permutation(len(p))
        newtrain.append(p.apply(lambda x:x.values.take(random_state)))
        newtarget.append(o.apply(lambda x:x.values.take(random_state)))
    return pd.concat(newtrain), pd.concat(newtarget)


# In[8]:


# @jit
def augment_fast(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        mask = mask[:,0]
        x1 = x[mask,:].copy()
        ids = np.arange(x1.shape[0])
        for p in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,p] = x1[ids][:,p]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        mask = mask[:,0]
        x1 = x[mask, :].copy()
        ids = np.arange(x1.shape[0])
        for n in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,n] = x1[ids][:,n]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
#     print(y.shape)
#     print(ys.shape)
#     print(yn.shape)
    y = np.concatenate([y[:,0],ys,yn])
    return x,y


# In[9]:


training_features = original_features
training_targets = original_targets


# In[10]:


# done in the model loop:
simple_aug = False


# In[11]:


# training_features = original_features[original_features.columns[200:]]
# training_targets = original_targets
# testing_features = testing_features[testing_features.columns[200:]]


# In[12]:


print('Training feature shape: ', training_features.shape)
print('Training targe shape:  ', training_targets.shape)
print('Testing targe shape:  ', testing_features.shape)


# In[13]:


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.1, #0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': -1,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
}


# In[14]:


get_ipython().run_cell_magic('time', '', 'result=np.zeros(testing_features.shape[0])\n\nskf = StratifiedKFold(n_splits=5, random_state=None)\nfor counter,(train_index, valid_index) in enumerate(skf.split(training_features, training_targets),1):\n    print (counter)\n    \n    X_train, y_train = training_features.iloc[train_index], training_targets.iloc[train_index]\n    X_valid, y_valid = training_features.iloc[valid_index], training_targets.iloc[valid_index]\n    \n    if simple_aug:\n        X_train, y_train = augment_fast(X_train.values, y_train.values)\n        X_train = pd.DataFrame(X_train)\n        print("Augmenation done!")\n    \n    # LightGBM data\n    lgb_train = lgb.Dataset(X_train, label=y_train)\n    lgb_val = lgb.Dataset(X_valid, label=y_valid)\n    \n    #Training\n    model_lgb = lgb.train(param, lgb_train, 1000, valid_sets = [lgb_train, lgb_val], verbose_eval=20, early_stopping_rounds = 50)\n    result += model_lgb.predict(testing_features, num_iteration=model_lgb.best_iteration)')


# In[15]:


submission_lgb = sample_submission.copy()
submission_lgb['target'] = result/counter
filename="{:%Y-%m-%d_%H_%M}_lgb.csv".format(datetime.now())
submission_lgb.to_csv(filename, index=False)


# In[16]:


# scaler = StandardScaler()
scaler = MinMaxScaler()

scaler.fit(training_features)
training_features_scaled = pd.DataFrame(scaler.transform(training_features))
testing_features_scaled = pd.DataFrame(scaler.transform(testing_features))


# In[17]:


# Build neural network
def create_model():
    model = Sequential()
    model.add(Dense(2**12, activation='relu', input_dim=training_features.shape[1]))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2**8, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(2**6, activation='linear'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(2**4, activation='linear'))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Defining optimizer:
    my_opt = optimizers.Adam(lr=0.001, beta_1=0.995, beta_2=0.99999, epsilon=0.01, decay=1e-6, amsgrad=True)
    # my_opt = optimizers.SGD(lr=0.0001, momentum=0.5, decay=1e-6, nesterov=True)

    # Compile model
    model.compile(optimizer=my_opt,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    
    return model


# In[18]:


model_ann = create_model()

result_ann=np.zeros(testing_features.shape[0])

skf = StratifiedKFold(n_splits=2, random_state=None)
for counter,(train_index, valid_index) in enumerate(skf.split(training_features_scaled, training_targets),1):
    print (counter)
    
    X_train, y_train = training_features_scaled.iloc[train_index], training_targets.iloc[train_index]
    X_valid, y_valid = training_features_scaled.iloc[valid_index], training_targets.iloc[valid_index]
    
    if simple_aug:
        X_train, y_train = augment_fast(X_train.values, y_train.values)
        X_train = pd.DataFrame(X_train)
        print("Augmenation done!")
    
    #Training
    ### training
    es = EarlyStopping(monitor='val_binary_accuracy', mode='auto', verbose=0, patience=50)
    cp = ModelCheckpoint(filepath='/best_weights_'+str(counter)+'_.hdf5', verbose=1, save_best_only=True)
    model_ann.fit(X_train, y_train, batch_size=2**8, epochs=1000, validation_data=(X_valid, y_valid), callbacks=[es, cp], verbose=1)
    
    model_best = create_model()
    model_best.load_weights('/best_weights_'+str(counter)+'_.hdf5')
    result_ann += model_best.predict(testing_features_scaled)[:,0]


# In[19]:


submission_ann = sample_submission.copy()
submission_ann['target'] = result/counter
filename="{:%Y-%m-%d_%H_%M}_ann.csv".format(datetime.now())
submission_ann.to_csv(filename, index=False)


# In[20]:


get_ipython().run_cell_magic('time', '', 'result_gnb=np.zeros(testing_features.shape[0])\n\nskf = StratifiedKFold(n_splits=5, random_state=None)\nfor counter,(train_index, valid_index) in enumerate(skf.split(training_features, training_targets),1):\n    print (counter)\n    \n    X_train, y_train = training_features.iloc[train_index], training_targets.iloc[train_index]\n    X_valid, y_valid = training_features.iloc[valid_index], training_targets.iloc[valid_index]\n    \n    if simple_aug:\n        X_train, y_train = augment_fast(X_train.values, y_train.values)\n        X_train = pd.DataFrame(X_train)\n        print("Augmenation done!")\n    \n    skf_inner = StratifiedKFold(n_splits=4, random_state=None)\n\n    # Optimizing\n    # Score. Optimizer will try to find minimum, so we will add a "-" sign.\n    def ml_fun(parameters):\n        parameters = parameters[0]\n        score = -cross_val_score(GaussianNB(priors=None, var_smoothing = parameters[0]),\n            X_train, y_train, cv=skf_inner.split(X_train, y_train), scoring=\'roc_auc\', n_jobs=-1).mean()\n        score = np.array(score)\n        return score\n\n    # Bounds (NOTE: define continuous variables first, then discrete!)\n    bounds = [\n                {\'name\': \'var_smoothing\', \'type\': \'continuous\', \'domain\': (1e-11, 1e-7)}\n             ]\n\n    optimizer = GPyOpt.methods.BayesianOptimization(f=ml_fun, domain=bounds,\n                                                acquisition_type =\'MPI\',\n                                                exact_eval=True)\n\n    optimizer.run_optimization(1e2, 1e3, verbosity=True)\n    \n    #Training & validation\n    gnb = GaussianNB(priors=None, var_smoothing = optimizer.x_opt[0]).fit(X_train, y_train)\n    GaussNB_val_proba = gnb.predict_proba(X_valid)\n    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_valid, GaussNB_val_proba[:, 1])\n    print("GaussianNB validation {d}: {:.5f}".format(counter, auc(fpr_keras, tpr_keras)))\n    \n    result_gnb += gnb.predict_proba(testing_features)')


# In[21]:


submission_gnb = sample_submission.copy()
submission_gnb['target'] = result_gnb/counter
filename="{:%Y-%m-%d_%H_%M}_gbn.csv".format(datetime.now())
submission_gnb.to_csv(filename, index=False)


# In[22]:


get_ipython().run_cell_magic('time', '', 'result_LogR=np.zeros(testing_features.shape[0])\n\nskf = StratifiedKFold(n_splits=5, random_state=None)\nfor counter,(train_index, valid_index) in enumerate(skf.split(training_features, training_targets),1):\n    print (counter)\n    \n    X_train, y_train = training_features.iloc[train_index], training_targets.iloc[train_index]\n    X_valid, y_valid = training_features.iloc[valid_index], training_targets.iloc[valid_index]\n    \n    if simple_aug:\n        X_train, y_train = augment_fast(X_train.values, y_train.values)\n        X_train = pd.DataFrame(X_train)\n        print("Augmenation done!")\n    \n    skf_inner = StratifiedKFold(n_splits=4, random_state=None)\n\n    # Optimizing\n    # Score. Optimizer will try to find minimum, so we will add a "-" sign.\n    def ml_fun(parameters):\n        parameters = parameters[0]\n        score = -cross_val_score(LogisticRegression(penalty=\'l1\', dual=False, solver=\'saga\', multi_class=\'auto\',\n                           tol = parameters[0],\n                           C = parameters[1], \n                           max_iter = parameters[2],\n                           n_jobs=-1),\n            X_train, y_train, cv=skf_inner.split(X_train, y_train), scoring=\'roc_auc\', n_jobs=-1).mean()\n        score = np.array(score)\n        return score\n\n    # Bounds (NOTE: define continuous variables first, then discrete!)\n    bounds = [\n                {\'name\': \'var_smoothing\', \'type\': \'continuous\', \'domain\': (1e-11, 1e-7)}\n             ]\n\n    optimizer = GPyOpt.methods.BayesianOptimization(f=ml_fun, domain=bounds,\n                                                acquisition_type =\'MPI\',\n                                                exact_eval=True)\n\n    optimizer.run_optimization(1e2, 1e3, verbosity=True)\n    \n    #Training & validation\n    LogR = LogisticRegression(LogisticRegression(penalty=\'l1\', dual=False, solver=\'saga\', multi_class=\'auto\',\n                           tol = optimizer.x_opt[0],\n                           C = optimizer.x_opt[1], \n                           max_iter = optimizer.x_opt[2],\n                           n_jobs=-1).fit(X_train, y_train)\n    LogR_val_proba = LogR.predict_proba(X_valid)\n    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_valid, LogR_val_proba[:, 1])\n    print("GaussianNB validation {d}: {:.5f}".format(counter, auc(fpr_keras, tpr_keras)))\n    \n    result_LogR += LogR.predict_proba(testing_features)')


# In[23]:


submission_LogR = sample_submission.copy()
submission_LogR['target'] = result_LogR/counter
filename="{:%Y-%m-%d_%H_%M}_gbn.csv".format(datetime.now())
submission_LogR.to_csv(filename, index=False)

