#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing #for labelling data
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


test.head()


# In[ ]:


le = preprocessing.LabelEncoder() #we will be using label encoder for labelling classes


# In[ ]:


le.fit(train.species)
Labels=le.transform(train.species) # will get labels
Classes=le.classes_
test_ids = test.id                       


# In[ ]:


train.head() # need to remove id and species column 


# In[ ]:


train=train.drop(['id','species'],axis=1)
test=test.drop(['id'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, Labels, test_size=0.30, random_state=42)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


Predicted=clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, Predicted)
acc


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[ ]:


clf1=LinearDiscriminantAnalysis()


# In[ ]:


clf1.fit(X_train,y_train)


# In[ ]:


Predictedlda=clf.predict(X_test)
accLda= accuracy_score(y_test, Predictedlda)
accLda


# In[ ]:


Finalpredictions= clf1.predict_proba(test)


# In[ ]:


Finalpredictions.shape


# In[ ]:


Submit = pd.DataFrame(Finalpredictions, columns=Classes)
Submit.insert(0, 'id', test_ids)
Submit.reset_index()


#submission.to_csv('submission.csv', index = False)
Submit.to_csv('leaf_classification.csv', index = False)
Submit.tail()


# In[ ]:




