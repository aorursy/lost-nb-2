#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Minimalistic version of the catbost model using several "magical" features. The idea can be developed further.


# In[2]:


import numpy as np
import gc, os
import pandas as pd
import numpy as np
import catboost as ctb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from scipy.stats import gmean
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


# In[3]:


VERSION = 6
SEED = 42


# In[4]:


test_df_src = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv') 
train_df_src = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
with open('../input/data-separation/synthetic_samples_indexes.npy', 'rb') as f:
    test_synth = np.load(f)
    test_synth = list(test_synth)
with open('../input/data-separation/public_LB.npy', 'rb') as f:
    test_pub = np.load(f)  
    test_pub = list(test_pub.reshape(-1)[0])
with open('../input/data-separation/private_LB.npy', 'rb') as f:
    test_priv = np.load(f)  
    test_priv = list(test_priv.reshape(-1)[0])


# In[5]:


var_cols = [c for c in list(train_df_src.columns) if 'var_' in c]


# In[6]:


all_df_real = pd.concat([train_df_src, test_df_src.loc[test_pub + test_priv]],
                        axis=0, copy=False, sort=False).set_index('ID_code').astype('float32')
all_df_synth = pd.concat([train_df_src, test_df_src.loc[test_synth]],
                        axis=0, copy=False, sort=False).set_index('ID_code').astype('float32')


# In[7]:


def get_all_freq(df, columns, frqs_series=None):
    freq_df = pd.DataFrame(index=df.index)
    f_s = frqs_series if frqs_series else {}
    for col in tqdm(columns):        
        if not frqs_series:
            f_s[col] = df[f'{col}'].value_counts()
        freq_df[f'{col}_freq_N'] = df[f'{col}'].map(f_s[col])        
        freq_df[f'{col}_freq_1'] = (freq_df[f'{col}_freq_N'] > 1).astype('category')
        freq_df[f'{col}_mul_freq'] = df[col]*freq_df[f'{col}_freq_N']
        freq_df[f'{col}_div_freq'] = df[col]/freq_df[f'{col}_freq_N']          
    return freq_df, f_s


# In[8]:


all_df_real_uflag, f_s = get_all_freq(all_df_real, var_cols)
all_df_synth_uflag, f_s = get_all_freq(all_df_synth, var_cols, f_s)

all_df_real = pd.concat([all_df_real, all_df_real_uflag],
                        axis=1, copy=False, sort=False)
all_df_synth = pd.concat([all_df_synth, all_df_synth_uflag],
                         axis=1, copy=False, sort=False)

train_df = all_df_real[all_df_real.index.str.contains('train') | all_df_real.index.str.contains('pred')].copy()
train_df = train_df.loc[train_df_src['ID_code']]


# In[9]:


del all_df_real_uflag, all_df_synth_uflag, train_df_src
gc.collect()


# In[10]:


test_df_real = all_df_real[all_df_real.index.str.contains('test')]
test_df_synth = all_df_synth[all_df_synth.index.str.contains('test')]
test_df = pd.concat([test_df_real, test_df_synth], axis=0, copy=True, sort=False)
test_df = test_df.loc[test_df_src['ID_code']]


# In[11]:


del all_df_real, all_df_synth, test_df_src
gc.collect()


# In[12]:


features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']


# In[13]:


def add_freq(df):
    df_res = pd.DataFrame(index=df.index)
    for c in tqdm(var_cols):
        df_res[f'{c}_freq_mean'] = df.groupby(f'{c}_freq_1')[c].transform(np.mean).astype('float16')
        df_res[f'{c}_freq_std'] = df.groupby(f'{c}_freq_1')[c].transform(np.std).astype('float16')
        df_res[f'{c}_freq_var1'] = (df_res[f'{c}_freq_mean'] - 2*df_res[f'{c}_freq_std']).astype('float16') 
        df_res[f'{c}_freq_var2'] = (df_res[f'{c}_freq_mean'] + 2*df_res[f'{c}_freq_std']).astype('float16') 
        df_res[f'{c}_freq_mean_unq'] = df[c]
        idx = df[df[f'{c}_freq_1']==False].index
        df_res.loc[idx, [f'{c}_freq_mean_unq']] = df_res.loc[idx][f'{c}_freq_mean_unq'].mean()
    return df_res

    
def add_features(df):
    df_count = add_freq(df)
    colums = [f'{x}_freq_N' for x in var_cols] +              [f'{x}_freq_1' for x in var_cols] +              [f'{x}_mul_freq' for x in var_cols] +              [f'{x}_div_freq' for x in var_cols] +              ['target']
    res = pd.concat([df[colums], df_count], axis=1, copy=False, sort=False)
    return res


# In[14]:


param = {
    'random_seed': SEED,
    'gpu_ram_part': 0.95,
    'iterations': 200000,
    'learning_rate': 0.04,
    'l2_leaf_reg': 5,
    'depth': 1,
    'thread_count': 4,
    'custom_metric': ['Logloss', 'AUC:hints=skip_train~false'],
    'od_type': 'Iter',
    'od_wait': 500,
    'task_type': 'GPU',
    'eval_metric': 'AUC',
    'use_best_model': True
}


# In[15]:


get_ipython().run_cell_magic('time', '', '\nnum_folds = 5\nfeatures = [c for c in train_df.columns if c not in [\'ID_code\', \'target\']]\nprint(\'Training the Model:\')\nval_list = []\npredictions = []\nfor i in range(4):\n    param[\'depth\'] = i%2 + 1\n    clf = ctb.CatBoostClassifier(**param)\n    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED*i)\n    oof = np.zeros(len(train_df))\n    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):\n        gc.collect()\n        print("Fold idx:{}".format(fold_ + 1))\n        trn_data = add_features(train_df.iloc[trn_idx])\n        y_trn = trn_data[\'target\']\n        features_ext = [c for c in trn_data.columns if c not in [\'ID_code\', \'target\']]\n        trn_data = trn_data[features_ext]\n        val_data  = add_features(train_df.iloc[val_idx])\n        y_val = val_data[\'target\']\n        val_data = val_data[features_ext]         \n        \n        clf.fit(X=trn_data, y=y_trn, eval_set=[(val_data, y_val)], verbose=1000, early_stopping_rounds = 1001) \n        del trn_data, val_data\n        gc.collect() \n        \n        train_data = add_features(train_df.iloc[val_idx])[features_ext]        \n        oof[val_idx] = clf.predict_proba(train_data)[:,1]\n        del train_data\n        gc.collect()\n        test_data = add_features(test_df)[features_ext]\n        predictions.append(clf.predict_proba(test_data)[:,1])  \n        del test_data\n    del clf    \n    val_list.append(oof)\n    \noof = gmean(val_list, 0)\npredictions_gmean = gmean(predictions, 0)')


# In[16]:


sub = pd.DataFrame({"ID_code": test_df.index.values})
sub["target"] = predictions_gmean
sub.to_csv('submission_cb_{}_seed_v{}.csv'.format(SEED, VERSION), index=False)


# In[17]:


roc = roc_auc_score(target, oof)
print("CV score: {:<8.5f}".format(roc))


# In[18]:


oof_train = pd.DataFrame()
oof_train[f'oof_cb_{num_folds}_{SEED}_{VERSION}'] = oof
oof_test = pd.DataFrame()
oof_test[f'oof_cb_{num_folds}_{SEED}_{VERSION}'] = predictions_gmean
oof_train.to_csv('train_cb_roc_{}_seed_{}_v{}.csv'.format(
    roc, SEED, VERSION), index=False)
oof_test.to_csv('test_cb_roc_{}_seed_{}_v{}.csv'.format(
    roc, SEED, VERSION), index=False)

