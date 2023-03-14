#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[3]:


import pandas as pd
test = pd.read_csv("../input/data-science-london-scikit-learn/test.csv")
train = pd.read_csv("../input/data-science-london-scikit-learn/train.csv")
trainLabels = pd.read_csv("../input/data-science-london-scikit-learn/trainLabels.csv")


# In[4]:


seed = 2020
np.random.seed(seed)


# In[5]:


#Test with a 20 split
X, y = train, np.ravel(trainLabels)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)


# In[6]:


train.head()


# In[7]:


trainLabels.head()


# In[8]:


train.shape


# In[9]:


trainLabels.shape


# In[10]:


train.describe()


# In[11]:


def model(model):
   
   scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
   print('Cross validation score - ', scores.mean()*100)
   
   model.fit(X_train, y_train)
   y_pred = model.predict(X_valid)

   accuracy = accuracy_score(y_valid, y_pred) 
   print('Validation accuracy - ',accuracy*100)
   
   #Return trained model
   return model


# In[12]:


knn = KNeighborsClassifier()
knn = model(knn)


# In[13]:


dt = DecisionTreeClassifier(random_state=seed)
dt = model(dt)


# In[14]:


rf = RandomForestClassifier(n_estimators=8, random_state=seed)
rf = model(rf)


# In[15]:


def perform_grid_search(model, param_grid, cv = 10, scoring='accuracy'):
    
    grid_search_model = GridSearchCV(estimator=model, param_grid=param_grid, cv = cv,scoring=scoring,n_jobs=-1, iid=False)
    grid_search_model.fit(X_train, y_train)


    best_model = grid_search_model.best_estimator_
    print('Best Accuracy :',grid_search_model.best_score_ * 100)
    print('Best Parmas',grid_search_model.best_params_)
    
    y_pred = best_model.predict(X_valid)
    print('Validation Accuracy',accuracy_score(y_valid, y_pred)*100)
    
    return best_model


# In[16]:


knn = KNeighborsClassifier()
n_neighbors = [3,4,5,6,7,8,9,10]
param_grid_knn = dict(n_neighbors=n_neighbors)
knn_best = perform_grid_search(knn, param_grid_knn)


# In[17]:


pred  = knn_best.predict(test)
best_pred = pd.DataFrame(pred)

best_pred.index += 1

best_pred.columns = ['Solution']
best_pred['Id'] = np.arange(1, best_pred.shape[0]+1)
best_pred = best_pred[['Id', 'Solution']]

print(best_pred)

