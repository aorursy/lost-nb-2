#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
pd.options.display.precision = 15

import lightgbm as lgb
import time
import datetime


import warnings
warnings.filterwarnings("ignore")
import gc


# In[2]:


def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()

def train_lgb_regression_group(X, X_test, y, params, folds, groups,
                               eval_metric='mae', 
                               columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - Group Kfolds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                        'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                        'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                        'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    
    result_dict = {}
    
    # out-of-fold predictions on train data
    oof = np.zeros(len(X))
    
    # averaged predictions on train data
    prediction = np.zeros(len(X_test))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,groups=groups)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
        model.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_valid, y_valid)], 
                  eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                verbose=verbose, early_stopping_rounds=early_stopping_rounds)

        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred    
        
        if plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits
    
    print('CV mean score: {0:.6f}, std: {1:.6f}.\n'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    
    if plot_feature_importance:
        feature_importance["importance"] /= folds.n_splits
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:50].index

        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

        plt.figure(figsize=(16, 12));
        sns.barplot(x="importance", y="feature", 
                    data=best_features.sort_values(by="importance", ascending=False));
        plt.title('LGB Features (avg over folds)');

        result_dict['feature_importance'] = feature_importance
        
    return result_dict


# In[3]:


giba_columns = ['inv_dist0', 'inv_dist1', 'inv_distP', 'inv_dist0R', 'inv_dist1R', 'inv_distPR',
 'inv_dist0E', 'inv_dist1E', 'inv_distPE', 'linkM0', 'linkM1',
 'min_molecule_atom_0_dist_xyz',
 'mean_molecule_atom_0_dist_xyz',
 'max_molecule_atom_0_dist_xyz',
 'sd_molecule_atom_0_dist_xyz',
 'min_molecule_atom_1_dist_xyz',
 'mean_molecule_atom_1_dist_xyz',
 'max_molecule_atom_1_dist_xyz',
 'sd_molecule_atom_1_dist_xyz',
 'coulomb_C.x', 'coulomb_F.x', 'coulomb_H.x', 'coulomb_N.x', 'coulomb_O.x',
 'yukawa_C.x', 'yukawa_F.x', 'yukawa_H.x', 'yukawa_N.x', 'yukawa_O.x',
 'vander_C.x', 'vander_F.x', 'vander_H.x', 'vander_N.x', 'vander_O.x',
 'coulomb_C.y', 'coulomb_F.y', 'coulomb_H.y', 'coulomb_N.y', 'coulomb_O.y',
 'yukawa_C.y', 'yukawa_F.y', 'yukawa_H.y', 'yukawa_N.y', 'yukawa_O.y',
 'vander_C.y', 'vander_F.y', 'vander_H.y', 'vander_N.y', 'vander_O.y',
 'distC0', 'distH0', 'distN0', 'distC1', 'distH1', 'distN1',
 'adH1', 'adH2', 'adH3', 'adH4', 'adC1', 'adC2', 'adC3', 'adC4',
 'adN1', 'adN2', 'adN3', 'adN4',
 'NC', 'NH', 'NN', 'NF', 'NO']

qm9_columns = [
'rc_A', 'rc_B', 'rc_C', 
'mu', 'alpha', 
'homo','lumo', 'gap', 
'zpve', 'Cv', 
'freqs_min', 'freqs_max', 'freqs_mean',
'mulliken_min', 'mulliken_max', 
'mulliken_atom_0', 'mulliken_atom_1'
]

label_columns = ['molecule_name',
'atom_index_0', 'atom_index_1',
'structure_atom_0','structure_atom_1','type']

index_columns = ['type','molecule_name','id']

diff_columns = ['Cv',
 'alpha', 'freqs_max', 'freqs_mean', 'freqs_min',
 'gap', 'homo', 'linkM0',
 'lumo', 'mu', 'mulliken_atom_0', 'mulliken_max', 'mulliken_min',
 'rc_A', 'rc_B', 'rc_C', 'sd_molecule_atom_1_dist_xyz', 'zpve']


# In[4]:


get_ipython().run_cell_magic('time', '', 'file_folder = \'../input/champs-scalar-coupling\'\n\nprint("Load Giba\'s features...")\ntrain = pd.read_csv(\'../input/giba-molecular-features/train_giba.csv/train_giba.csv\',\n                   usecols=index_columns+giba_columns+[\'scalar_coupling_constant\'])\ny = train[\'scalar_coupling_constant\']\ntrain = reduce_mem_usage(train,verbose=True)\n\ntest = pd.read_csv(\'../input/giba-molecular-features/test_giba.csv/test_giba.csv\',\n                  usecols=index_columns+giba_columns)\ntest = reduce_mem_usage(test,verbose=True)\n\nprint("Load QM9 features...")\ndata_qm9 = pd.read_pickle(\'../input/quantum-machine-9-qm9/data.covs.pickle\')\ndata_qm9 = data_qm9.drop(columns = [\'type\', \'linear\', \'atom_index_0\', \'atom_index_1\', \n            \'scalar_coupling_constant\', \'U\', \'G\', \'H\', \n            \'mulliken_mean\', \'r2\', \'U0\'], axis=1)\ndata_qm9 = reduce_mem_usage(data_qm9,verbose=False)\n\ntrain = pd.merge(train, data_qm9, how=\'left\', on=[\'molecule_name\',\'id\'])\ntest = pd.merge(test, data_qm9, how=\'left\', on=[\'molecule_name\',\'id\'])\n\ndel data_qm9\ngc.collect()\n\nprint("Encoding label features...\\n")\nfor f in label_columns: \n    # \'type\' has to be the last one\n    # since the this label encoder is used later\n    if f in train.columns:\n        lbl = LabelEncoder()\n        lbl.fit(list(train[f].values) + list(test[f].values))\n        train[f] = lbl.transform(list(train[f].values))\n        test[f] = lbl.transform(list(test[f].values))\n        \ntrain = train[index_columns+giba_columns+qm9_columns]\ntest = test[index_columns+giba_columns+qm9_columns]')


# In[5]:


coef = [0.25, 0.5, 0.2, 0.05]

def custom_objective(y_true, y_pred):
    
    # fair
    c = 0.5
    residual = y_pred - y_true
    grad = c * residual /(np.abs(residual) + c)
    hess = c ** 2 / (np.abs(residual) + c) ** 2
    
    # huber
    h = 1.2  #h is delta in the Huber's formula
    scale = 1 + (residual / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad_huber = residual / scale_sqrt
    hess_huber = 1 / scale / scale_sqrt

    # rmse grad and hess
    grad_rmse = residual
    hess_rmse = 1.0

    # mae grad and hess
    grad_mae = np.array(residual)
    grad_mae[grad_mae > 0] = 1.
    grad_mae[grad_mae <= 0] = -1.
    hess_mae = 1.0

    return coef[0] * grad + coef[1] * grad_huber + coef[2] * grad_rmse + coef[3] * grad_mae,            coef[0] * hess + coef[1] * hess_huber + coef[2] * hess_rmse + coef[3] * hess_mae

params = {
'num_leaves': 400,
'objective': custom_objective,
'max_depth': 9,
'learning_rate': 0.1,
'boosting_type': 'gbdt',
'metric': 'mae',
'verbosity': -1,
'lambda_l1': 2,
'lambda_l2': 0.2,
'feature_fraction': 0.6,
}


# In[6]:


X_short = pd.DataFrame({'ind': list(train.index), 
                        'type': train['type'].values,
                        'oof': [0] * len(train), 
                        'target': y.values})
X_short_test = pd.DataFrame({'ind': list(test.index), 
                             'type': test['type'].values, 
                             'prediction': [0] * len(test)})


# In[7]:


get_ipython().run_cell_magic('time', '', "CV_score = 0\nfolds = GroupKFold(n_splits=3)\n####Iters####  [1JHC, 1JHN, 2JHC, 2JHH, 2JHN, 3JHC, 3JHH, 3JHN]\nn_estimators = [6000, 2500, 3500, 3000, 3000, 5000, 3000, 3000]\n\nfor t in train['type'].unique():\n    type_ = lbl.inverse_transform([t])[0]\n    print(f'Training of type {t}: {type_}.\\n')\n    X_t = train.loc[train['type'] == t]\n    X_test_t = test.loc[test['type'] == t]\n    y_t = X_short.loc[X_short['type'] == t, 'target']\n    \n    scaler = StandardScaler()\n    X_t[diff_columns] = scaler.fit_transform(X_t[diff_columns].fillna(-999))\n    X_test_t[diff_columns] = scaler.transform(X_test_t[diff_columns].fillna(-999))\n    \n    molecules_id = X_t.molecule_name\n    \n    result_dict_lgb = train_lgb_regression_group(X=X_t.drop(columns=['molecule_name','id']), \n                                          X_test=X_test_t.drop(columns=['molecule_name','id']), \n                                          y=y_t, params=params, \n                                          folds=folds, groups=molecules_id,\n                                          eval_metric='group_mae', \n                                          plot_feature_importance=False,\n                                          verbose=3000, early_stopping_rounds=200, \n                                          n_estimators=n_estimators[t])\n    X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb['oof']\n    X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict_lgb['prediction']\n    CV_score += np.array(result_dict_lgb['scores']).mean()/8 # total 8 types")


# In[8]:


sub = pd.read_csv(f'{file_folder}/sample_submission.csv')
sub['scalar_coupling_constant'] = X_short_test['prediction']
today = str(datetime.date.today())
sub.to_csv(f'LGB_{today}_{CV_score:.4f}.csv', index=False)

