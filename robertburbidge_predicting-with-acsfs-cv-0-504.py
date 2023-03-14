#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('pip install dscribe')


# In[3]:


from sklearn import metrics
import lightgbm
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from dscribe.descriptors import ACSF
from ase.io import read as ase_read


# In[4]:


# settings for cross-validation and LightGBM training
nfolds = 5
niters = 4000


# In[5]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# In[6]:


# Setting up the ACSF descriptor
# see the kernel referenced above as well as the Dscribe docs:
# https://github.com/SINGROUP/dscribe/blob/master/docs/tutorials/acsf.html
# chemists can add more functions and tune the parameters here
acsf = ACSF(
    species=['H', 'C', 'N', 'O', 'F'],
    rcut=3.0,
    g2_params=[(0.4, 0.2),(0.4, 0.5),(0.4, 1.0),(0.5, 2.0),(0.5, 3.0),(0.5, 4.0)],
)


# In[7]:


# calculate ACSFs
def calc_acsf(df):
    acsf_arr = np.zeros([df.shape[0], 70])
    for i in range(df.shape[0]):
        molecule_name = df['molecule_name'].iloc[i]
        atoms = ase_read('../input/structures/' + molecule_name + '.xyz')
        acsfi = acsf.create(atoms, positions = [df['atom_index_0'].iloc[i], df['atom_index_1'].iloc[i]], n_jobs=4)
        acsf_arr[i, :] = np.reshape(acsfi, [70])
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(acsf_arr)], axis=1)
    return df


# In[8]:


train = calc_acsf(train)
test = calc_acsf(test)


# In[9]:


# predictive vars for LightGBM
pred_vars = [v for v in train_sample.columns if v not in ['id', 'molecule_name', 'scalar_coupling_constant',
                                                   'atom_index_0', 'atom_index_1']]


# In[10]:


# encode type
cat_feats = ['type']
for f in cat_feats:
    lbl = LabelEncoder()
    lbl.fit(list(train[f].values) + list(test[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))


# In[11]:


# heuristic parameters for LightGBM
params = { 'objective': 'regression_l1',
           'learning_rate': 0.1,
           'num_leaves': 255,
           'min_data_in_leaf': 100,
           'max_depth': 10,
           'num_threads': -1,
           'bagging_fraction': 0.5,
           'bagging_freq': 1,
           'feature_fraction': 0.9,
           'lambda_l1': 10.0,
           'lambda_l2': 10.0,
           'max_bin': 255,
           'verbosity': -1
           }


# In[12]:


# evaluation metric for validation
# https://www.kaggle.com/abhishek/competition-metric
def metric(df, preds):
    df["prediction"] = preds
    maes = []
    for t in df.type.unique():
        y_true = df[df.type==t].scalar_coupling_constant.values
        y_pred = df[df.type==t].prediction.values
        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))
        maes.append(mae)
    return np.mean(maes)


# In[13]:


# KFold cross-validation by molecule, separate models for each type
kf = KFold(n_splits=nfolds)
molecule_names = train_sample['molecule_name'].unique()

preds = np.empty([train_sample.shape[0]])
preds_sub = np.zeros([sub.shape[0]])
types = train_sample['type'].unique()

for train_mol_idx, val_mol_idx in kf.split(molecule_names):
    #
    train_idx = pd.merge(train_sample[['molecule_name']].reset_index(),
                         pd.DataFrame(molecule_names[train_mol_idx]), how='right',
                         left_on='molecule_name', right_on=0)['index']
    val_idx = pd.merge(train_sample[['molecule_name']].reset_index(),
                         pd.DataFrame(molecule_names[val_mol_idx]), how='right',
                       left_on='molecule_name', right_on=0)['index']
    #
    for type in types:
        train_data = lightgbm.Dataset(train_sample.iloc[train_idx,:].loc[train_sample.iloc[train_idx,:]['type']==type][pred_vars],
                                      label=train_sample.iloc[train_idx,:].loc[train_sample.iloc[train_idx,:]['type']==type]['scalar_coupling_constant'],
                                      categorical_feature=cat_feats)
        val_data = lightgbm.Dataset(train_sample.iloc[val_idx,:].loc[train_sample.iloc[val_idx,:]['type']==type][pred_vars],
                                      label=train_sample.iloc[val_idx,:].loc[train_sample.iloc[val_idx,:]['type']==type]['scalar_coupling_constant'],
                                      categorical_feature=cat_feats)
        #
        # training
        model = lightgbm.train(params,
                               train_data,
                               valid_sets=[train_data, val_data], verbose_eval=int(niters/8),
                               num_boost_round=niters,
                               early_stopping_rounds=int(niters/40))
        #
        tmp_idx = val_idx.values[train_sample.iloc[val_idx, :]['type'] == type]
        preds[tmp_idx] =            model.predict(train_sample.iloc[val_idx,:].loc[train_sample.iloc[val_idx,:]['type']==type][pred_vars])
        #
        preds_sub[test['type']==type] = preds_sub[test['type']==type] +             model.predict(test.loc[test['type']==type,:][pred_vars])


# In[14]:


# validation performance
print(metric(pd.concat([train_sample[pred_vars], train_sample['scalar_coupling_constant']], axis=1), preds))


# In[15]:


# submission
sub['scalar_coupling_constant'] = preds_sub
sub.to_csv('submission_acsf01.csv', index=False)

