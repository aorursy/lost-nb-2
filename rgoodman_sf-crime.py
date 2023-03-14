#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt


# In[2]:


matplotlib inline


# In[3]:


train = pd.read_csv("/kaggle/input/sf-crime/train.csv")
test = pd.read_csv("/kaggle/input/sf-crime/test.csv")


# In[4]:


train.head()


# In[5]:


train.dtypes


# In[6]:


test.head()


# In[7]:


np.setdiff1d(test.columns,train.columns)


# In[8]:


np.setdiff1d(train.columns,test.columns).tolist()


# In[9]:


train.isnull().sum()


# In[10]:


train.nunique()


# In[11]:


train['Resolution'].value_counts().plot.barh();


# In[12]:


train[train['Resolution']!='NONE']['Resolution'].value_counts().plot.barh();


# In[13]:


train['PdDistrict'].value_counts().plot.barh();


# In[14]:


train['DayOfWeek'].value_counts().plot.barh();


# In[15]:


train['Category'].value_counts().plot.barh(figsize = (5,18));

dropping "Descript" and "Resolution" from train data since those features aren't in the test data.
# In[16]:


train_feats = train.drop(labels = ['Descript','Resolution'],axis = 1)
train_feats.head()


# In[17]:


train_dummies = pd.get_dummies(train_feats[['PdDistrict','DayOfWeek']])
train_feats_dummies = pd.merge(train_feats.drop(['PdDistrict','DayOfWeek'],1),train_dummies,left_index = True, right_index = True)
train_feats_dummies.columns


# In[18]:


test_dummies = pd.get_dummies(test[['PdDistrict','DayOfWeek']])
test_feats_dummies = pd.merge(test.drop(['PdDistrict','DayOfWeek'],1),test_dummies,left_index = True, right_index = True)
test_feats_dummies = test_feats_dummies.drop(['Dates','Address'],1)
test_feats_dummies.columns


# In[19]:


X = train_feats_dummies.drop('Category',1)
y = train_feats_dummies[['Category']]


# In[20]:


X = X.drop(['Dates','Address'],1)
X.head(2)


# In[21]:


y.head(2)


# In[22]:


pd.get_dummies(y,prefix = None,prefix_sep = '')


# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)


# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# In[25]:


#clf = GaussianNB().fit(X_train, y_train['Category']) #worked (maybe)
#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train['Category']) #worked
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr').fit(X_train, y_train['Category']) #worked


# In[26]:


clf.coef_.shape


# In[27]:


clf.classes_


# In[28]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[29]:


y_pred = clf.predict(X_test)

print(np.unique(y_pred,return_counts = True))
#print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# In[30]:


X_train.head(2)


# In[31]:


test_feats_dummies.head(2)


# In[32]:


test_predictions = clf.predict(test_feats_dummies.drop('Id',1))
test_predictions


# In[33]:


test_probability_predictions = clf.predict_proba(test_feats_dummies.drop('Id',1))


# In[34]:


test_probability_predictions.shape


# In[35]:


clf.classes_


# In[36]:


submission = pd.merge(test['Id'],pd.DataFrame(data = test_probability_predictions,columns = clf.classes_),          left_index = True, right_index = True)


# In[37]:


submission.to_csv('submission.csv', index = False)

