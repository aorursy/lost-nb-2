#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv("../input/datasetsmodifiedscancer/trainM.csv")
teste = pd.read_csv('../input/datasetsmodifiedscancer/testM.csv')
data.head(5)


# In[3]:


xgb = XGBClassifier(base_score=0.0025, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.65, gamma=2, learning_rate=0.3, max_delta_step=1,
       max_depth=4, min_child_weight=2, missing=None, n_estimators=280,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


# In[4]:


def correlate(data,t1,t2):
    numerical_column = ['int64','float64'] #select only numerical features to find correlation
    plt.figure(figsize=(t1,t2))
    sns.heatmap(
        data.select_dtypes(include=numerical_column).corr(),
        cmap=plt.cm.RdBu,
        vmax=1.0,
        linewidths=0.1,
        linecolor='white',
        square=True,
        annot=True
    )
    
   


# In[5]:


correlate(data,25,25)


# In[6]:


train=pd.DataFrame((data[['diagnosis','radius_mean','perimeter_mean','area_mean','concave points_mean','radius_worst','perimeter_worst','area_worst','concave points_worst']]))
test=pd.DataFrame((teste[['radius_mean','perimeter_mean','area_mean','concave points_mean','radius_worst','perimeter_worst','area_worst','concave points_worst']]))


# In[7]:


correlate(train,10,10)


# In[8]:


X=np.array(train.drop('diagnosis',1))
y=np.array(train.diagnosis)
xteste=np.array(test)
xtrei,xtest,ytrei,ytest=train_test_split(X,y,test_size=0.30)


# In[9]:


def score(model,x,y):
    prob=model.predict_proba(x)
    prob = prob[:, 1]
    auc = roc_auc_score(y, prob)
    print('AUC: {}\nROC_AUC: {}\n {}'.format(model.score(x,y),auc,prob[:10]))


# In[10]:



xgb.fit(X,y)
score(xgb,X,y)


# In[11]:


prob=xgb.predict_proba(xteste)


# In[12]:


submission = pd.DataFrame({
    "Id": teste.id, 
    "Expected": prob[:,1]
})
submission.head()


# In[13]:


submission.to_csv('sampleSubmission.csv', index=False)

