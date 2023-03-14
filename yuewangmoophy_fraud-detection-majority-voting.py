#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from shutil import copyfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
copyfile(src = "../input/158333a1/a1_wrangle.py", dst = "../working/a1_wrangle.py")
import a1_wrangle as a1w


# In[3]:


X_train, y_train, X_test, submission = a1w.quick_wrangle()


# In[4]:


y_train.head()


# In[5]:


X_train.head()


# In[6]:


X_test.head()


# In[7]:


rf = RandomForestClassifier(n_estimators=900, max_depth=2, random_state=0, n_jobs=4)
ab = AdaBoostClassifier(n_estimators=500, random_state=0)
xg = xgb.XGBClassifier(n_estimators=500, n_jobs=4, max_depth=11, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, missing=-999)

ec = VotingClassifier(estimators=[('rf', rf), ('ab', ab), ('xg', xg)], voting='hard')


# In[8]:


ec.fit(X_train, y_train)


# In[9]:


submission['isFraud'] = ec.predict(X_test)


# In[10]:


submission.to_csv('majority_voting_3_in_1.csv')

