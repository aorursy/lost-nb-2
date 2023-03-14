#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',message='DeprecationWarning')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


train=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip').copy()
test=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip').copy()


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.drop(['id'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)
sns.pairplot(train,hue='type')


# In[ ]:


sns.boxplot(x='type',y='bone_length',data=train)



# In[ ]:


sns.boxplot(x='type',y='hair_length',data=train)


# In[ ]:


sns.boxplot(x='type',y='has_soul',data=train)


# In[ ]:


sns.boxplot(x='type',y='rotting_flesh',data=train)


# In[ ]:


palette ={"clear":"moccasin","green":"green","black":"black", "white":"grey","blue":"blue",'blood':'red'}
sns.countplot(x='type',hue='color',data=train,palette=palette)


# In[ ]:


train.corr()


# In[ ]:


sns.heatmap(train.corr(),annot=True,vmin=-1)


# In[ ]:


X=train.drop(['type'],axis=1)
X=pd.get_dummies(X)
X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, train['type'] ,random_state=0)


# In[ ]:


clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[ ]:


sns.barplot(y=X_test.columns,x=clf.feature_importances_)


# In[ ]:


from sklearn import tree
tree.plot_tree(clf) 


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


accuracy_score=metrics.accuracy_score(y_test, y_pred)


# In[ ]:


print(accuracy_score)


# In[ ]:


params = {'max_leaf_nodes': list(range(1, 16)), 'min_samples_split': np.linspace(.1, 1,10, endpoint=True),"max_features":[1,4,6],'max_depth':np.linspace(1, 16, 16, endpoint=True)}
accuracy=metrics.make_scorer(metrics.accuracy_score)
clf1=GridSearchCV(clf,params,scoring=accuracy,n_jobs=-1)
clf1.fit(X_train,y_train)
print('best score :',clf1.best_score_)
print('params :',clf1.best_params_)


# In[ ]:


clf.get_params()
clf_best=clf1.best_estimator_


# In[ ]:


clf_best.fit(X_train,y_train)


# In[ ]:


y_pred_clf=clf_best.predict(X_test)
print('accuracy of best estimator for gridsearch:',metrics.accuracy_score(y_test,y_pred_clf))


# In[ ]:


rf=RandomForestClassifier(max_depth=12,max_features=6,max_leaf_nodes=10,min_samples_split=0.1)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


y_pred1=rf.predict(X_test)


# In[ ]:


accuracy_scorerf=metrics.accuracy_score(y_test, y_pred1)


# In[ ]:


print('accuracy Score:',accuracy_scorerf)
print('\n',metrics.classification_report(y_test,y_pred1))


# In[ ]:


rf.get_params()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 200, num = 200)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 20, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = np.linspace(.1, 1,10, endpoint=True)
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,8,10]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)


# In[ ]:


rf_random.fit(X_train,y_train)


# In[ ]:


rf_random.best_params_


# In[ ]:


rf_random.best_score_


# In[ ]:


rf_best=rf_random.best_estimator_


# In[ ]:


rf_best.fit(X_train,y_train)


# In[ ]:


y_pred_rf_best=rf_best.predict(X_test)


# In[ ]:


print(metrics.accuracy_score(y_test,y_pred_rf_best))


# In[ ]:


params1={'max_depth':[10,6,12,16] ,'max_features':['sqrt','auto'], 'max_leaf_nodes':[10,11,12,9],
                       'min_samples_leaf':[1,2,3,4], 'min_samples_split':[0.1,0.2],
                       'n_estimators':[94,85,173]}


# In[ ]:


rf=RandomForestClassifier()
rf_gsv=GridSearchCV(rf,param_grid=params1,cv=5,n_jobs=-1,scoring=accuracy)


# In[ ]:


rf_gsv.fit(X_train,y_train)
print('best score for rf GSV',rf_gsv.best_score_)


# In[ ]:


y_pred_rf_gsv=rf_gsv.predict(X_test)


# In[ ]:


print('acuuracy with test data:',metrics.accuracy_score(y_test,y_pred_rf_gsv))


# In[ ]:


test_=pd.get_dummies(test)


# In[ ]:


pre=rf_gsv.predict(test_)
test_f=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip').copy()


# In[ ]:


pre


# In[ ]:


df=pd.DataFrame({'id':test_f['id'],'type':pre},columns=['id','type'])
csv=df[['id','type']].to_csv('submission.csv',index=False)


# In[ ]:




