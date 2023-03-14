#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from shutil import copyfile
from sklearn.ensemble import RandomForestClassifier


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


for n in range(100, 1501, 200):
    clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0, n_jobs=4)
    clf.fit(X_train, y_train)
    submission['isFraud'] = clf.predict_proba(X_test)[:,1]
    submission.to_csv('RandomForest' + str(n) + '.csv')

