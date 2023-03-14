#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error


# In[ ]:


get_ipython().run_cell_magic('time', '', "data = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "def to_hist_func(row):\n    return np.bincount(row, minlength=30)\n\nfeatures = [f for f in data.columns if f not in ['target', 'ID']]\n\nhist_data = np.apply_along_axis(\n    func1d=to_hist_func, \n    axis=1, \n    arr=(np.log1p(data[features])).astype(int)) ")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'hist_test = np.apply_along_axis(\n    func1d=to_hist_func, \n    axis=1, \n    arr=(np.log1p(test[features])).astype(int)) ')


# In[ ]:


folds = KFold(n_splits=5, shuffle=True, random_state=1)
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

for n_fold, (trn_, val_) in enumerate(folds.split(hist_data)):
    reg = ExtraTreesRegressor(
        n_estimators=1000, 
        max_features=.8,                       
        max_depth=12, 
        min_samples_leaf=10, 
        random_state=3, 
        n_jobs=-1
    )
    # Fit Extra Trees
    reg.fit(hist_data[trn_], np.log1p(data['target'].iloc[trn_]))
    # Get OOF predictions
    oof_preds[val_] = reg.predict(hist_data[val_])
    # Update TEST predictions
    sub_preds += reg.predict(hist_test) / folds.n_splits
    # Display fold's score
    print('Fold %d scores : TRN %.4f TST %.4f'
          % (n_fold + 1,
             mean_squared_error(np.log1p(data['target'].iloc[trn_]),
                                reg.predict(hist_data[trn_])) ** .5,
             mean_squared_error(np.log1p(data['target'].iloc[val_]),
                                reg.predict(hist_data[val_])) ** .5))
          
print('Full OOF score : %.4f' % (mean_squared_error(np.log1p(data['target']), oof_preds) ** .5))


# In[ ]:


test['target'] = np.expm1(sub_preds)
test[['ID', 'target']].to_csv('histogram_predictions.csv', index=False)


# In[ ]:




