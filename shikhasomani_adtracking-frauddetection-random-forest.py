#!/usr/bin/env python
# coding: utf-8

# In[46]:


# This is a simple data exploration notebook. 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


print(os.listdir('../input'))


# In[47]:


train= pd.read_csv('../input/train_sample.csv')
train.head()


# In[48]:


train.info()


# In[49]:


cols=train.columns
cols_time=['click_time', 'attributed_time']
cols_categorical=[col for col in cols if col not in cols_time]

for col in cols_categorical:
    train[col]=train[col].astype('category')
for col in cols_time:
    train[col]=pd.to_datetime(train[col])


# In[50]:


train.describe()


# In[51]:


train['conversion_time']=pd.to_timedelta(train['attributed_time']-train['click_time']).astype('timedelta64[s]')
print(train['conversion_time'].quantile(0.9)/3600)
train.describe()


# In[11]:


ctimes=train['conversion_time'].dropna()


# In[12]:


sns.distplot(ctimes)


# In[13]:


sns.distplot(np.log10(ctimes))


# In[14]:


# min_time=np.nanmin(train['click_time'])
# max_time=np.nanmax(train['attributed_time'])
# print(min_time)
# print(max_time)


# In[15]:


# max_engagement_window_size=str(int(np.ceil(np.nanmax(ctimes)/3600)))+'H'
# max_engagement_window_size


# In[16]:


sns.pairplot(train[train['is_attributed']==1])


# In[17]:


col_list=[col for col in train.columns if col!='conversion_time']
# print(col_list)

sns.pairplot(train.loc[train['is_attributed']!=1,col_list])


# In[24]:


from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


# In[53]:


train['is_attributed'].value_counts()/train.shape[0]


# In[54]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve


# In[55]:


train['chour']=train['click_time'].dt.hour
train['cminute']=train['click_time'].dt.minute
train['cday']=train['click_time'].dt.day


# In[56]:


del train['click_time']
del train['attributed_time']


# In[ ]:


Check how much of the data still has missing values


# In[58]:


train.isnull().any()


# In[59]:


del train['conversion_time']


# In[60]:


X=train
Y=train['is_attributed']
del X['is_attributed']
X.head()


# In[61]:


Xtrain, Xtest, Ytrain, Ytest=train_test_split(X,Y,test_size=0.2, random_state=32)


# In[62]:


param_grid={'n_estimators':np.arange(10,100,20), 
           'max_depth':[None, 5, 10],
           'max_features':['auto','sqrt']}
rfgrid=GridSearchCV(RandomForestClassifier(),param_grid,cv=StratifiedKFold(5))


# In[63]:


rfgrid.fit(Xtrain, Ytrain)


# In[68]:


rfgrid.score(Xtest,Ytest)

