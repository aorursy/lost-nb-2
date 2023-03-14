#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pydatatable/0.7.0.dev490/x86_64-centos7/datatable-0.7.0.dev490-cp36-cp36m-linux_x86_64.whl')


# In[3]:


from sklearn.metrics import log_loss, roc_auc_score
from datetime import datetime
import datatable as dt
from datatable.models import Ftrl


# In[4]:


get_ipython().run_cell_magic('time', '', "train = dt.fread('../input/train.csv')")


# In[5]:


get_ipython().run_cell_magic('time', '', "test = dt.fread('../input/test.csv')")


# In[6]:


train.head()


# In[7]:


train.shape


# In[8]:


test.head()


# In[9]:


test.shape


# In[10]:


train.nunique()


# In[11]:


test.nunique()


# In[12]:


train[:, 'EngineVersion'].nunique1()


# In[13]:


train_unique = dt.unique(train[:, 'EngineVersion']).to_list()[0]
len(train_unique)


# In[14]:


test_unique = dt.unique(test[:, 'EngineVersion']).to_list()[0]
len(test_unique)


# In[15]:


intersection = list(set(train_unique) & set(test_unique))
len(intersection)


# In[16]:


train.names


# In[17]:


train.ltypes


# In[18]:


'''%%time
for name in test.names:
    if test[:, name].ltypes[0] == dt.ltype.str:
        train.replace(None, '-1')
        test.replace(None, '-1')
    elif test[:, name].ltypes[0] == dt.ltype.int:
        train.replace(None, -1)
        test.replace(None, -1)
    elif test[:, name].ltypes[0] == dt.ltype.bool:
        train.replace(None, 0)
        test.replace(None, 0)
    elif test[:, name].ltypes[0] == dt.ltype.real:
        train.replace(None, -1.0)
        test.replace(None, -1.0)'''


# In[19]:


get_ipython().run_cell_magic('time', '', "for f in train.names:\n    if f not in ['MachineIdentifier', 'HasDetections']:\n        if train[:, f].ltypes[0] == dt.ltype.str:\n            print('factorizing %s' % f)\n            col_f = pd.concat([train[:, f].to_pandas(), test[:, f].to_pandas()], ignore_index=True)\n            encoding = col_f.groupby(f).size()\n            encoding = encoding/len(col_f)\n            column = col_f[f].map(encoding).values.flatten()\n            del col_f, encoding\n            gc.collect()\n            train[:, f] = dt.Frame(column[:8921483])\n            test[:, f] = dt.Frame(column[8921483:])\n            del column\n            gc.collect()")


# In[20]:


train[:, f]


# In[21]:


train.head()


# In[22]:


test.head()


# In[23]:


features = [f for f in train.names if f not in ['HasDetections']]
ftrl = Ftrl(nepochs=2, interactions=True)


# In[24]:


get_ipython().run_cell_magic('time', '', "print('Start Fitting on   ', train.shape, ' @ ', datetime.now())\nftrl.fit(train[:, features], train[:, 'HasDetections'])\nprint('Fitted complete on ', train.shape, ' @ ', datetime.now())  \nprint('Current loss : %.6f' \n          % log_loss(np.array(train[:, 'HasDetections'])[:, 0],  \n                             np.array(ftrl.predict(train[:, features]))))")


# In[25]:


print('Current AUC : %.6f' 
          % roc_auc_score(np.array(train[:, 'HasDetections'])[:, 0],  
                             np.array(ftrl.predict(train[:, features]))))


# In[26]:


preds1 = np.array(ftrl.predict(test[:, features]))
preds1 = preds1.flatten()


# In[27]:


ftrl = Ftrl(nepochs=20, interactions=False)
ftrl.fit(train[:, features], train[:, 'HasDetections'])
preds2 = np.array(ftrl.predict(test[:, features]))
preds2 = preds2.flatten()


# In[28]:


np.save('preds1', preds1)
np.save('preds2', preds2)


# In[29]:


sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[30]:


sample_submission['HasDetections'] = 0.6*preds1+0.4*preds2


# In[31]:


sample_submission.to_csv('datatable_ftrl_submission.csv', index=False)

