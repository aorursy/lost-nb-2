#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for data visulization
import matplotlib.pyplot as plt
import seaborn as sns

# for modeling estimators
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbm
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

#for data processing
from sklearn.model_selection import train_test_split

#for tuning parameters
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
from eli5.sklearn import PermutationImportance


# Misc.
import os
import time
import gc


# In[2]:


# Read in data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

ids=test['Id']


# In[3]:


train.head()


# In[4]:


train.shape, test.shape


# In[5]:


train.info() 


# In[6]:


train.plot(figsize = (12,10))


# In[7]:


sns.countplot("Target", data=train)


# In[8]:


sns.countplot(x="r4t3",hue="Target",data=train)


# In[9]:


sns.countplot(x="hhsize",hue="Target",data=train)


# In[10]:


from pandas.plotting import scatter_matrix
scatter_matrix(train.select_dtypes('float'), alpha=0.2, figsize=(26, 20), diagonal='kde')


# In[11]:


from collections import OrderedDict

plt.figure(figsize = (20, 16))
plt.style.use('fivethirtyeight')

# Color mapping
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})

# Iterate through the float columns
for i, col in enumerate(train.select_dtypes('float')):
    ax = plt.subplot(4, 2, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        # Plot each poverty level as a separate line
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top = 2)


# In[12]:


train.select_dtypes('object').head()


# In[13]:


yes_no_map = {'no':0,'yes':1}
train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)
train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)
train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)


# In[14]:


yes_no_map = {'no':0,'yes':1}
test['dependency'] = test['dependency'].replace(yes_no_map).astype(np.float32)
test['edjefe'] = test['edjefe'].replace(yes_no_map).astype(np.float32)
test['edjefa'] = test['edjefa'].replace(yes_no_map).astype(np.float32)


# In[15]:


train[["dependency","edjefe","edjefa"]].describe()


# In[16]:


train[["dependency","edjefe","edjefa"]].hist()


# In[17]:


plt.figure(figsize = (16, 12))

# Iterate through the float columns
for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):
    ax = plt.subplot(3, 1, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        # Plot each poverty level as a separate line
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top = 2)


# In[18]:


# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[19]:


train['v18q1'] = train['v18q1'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)
train['v2a1'] = train['v2a1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)

train['rez_esc'] = train['rez_esc'].fillna(0)
test['rez_esc'] = test['rez_esc'].fillna(0)
train['SQBmeaned'] = train['SQBmeaned'].fillna(0)
test['SQBmeaned'] = test['SQBmeaned'].fillna(0)
train['meaneduc'] = train['meaneduc'].fillna(0)
test['meaneduc'] = test['meaneduc'].fillna(0)


# In[20]:


#Checking for missing values again to confirm that no missing values present
# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[21]:


#Checking for missing values again to confirm that no missing values present
# Number of missing in each column
missing = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[22]:


train.drop(['Id','idhogar'], inplace = True, axis =1)

test.drop(['Id','idhogar'], inplace = True, axis =1)


# In[23]:


train.shape, test.shape


# In[24]:


y = train.iloc[:,140]
y.unique()


# In[25]:


X = train.iloc[:,1:141]
X.shape


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)


# In[27]:


modelgbm=gbm()


# In[28]:


start = time.time()
modelgbm = modelgbm.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[29]:


classes = modelgbm.predict(X_test)

classes


# In[30]:


(classes == y_test).sum()/y_test.size 


# In[31]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    gbm(
               # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 2                # Number of cross-validation folds
)


# In[32]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[33]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[34]:


modelgbmTuned=gbm(
               max_depth=31,
               max_features=29,
               min_weight_fraction_leaf=0.02067,
               n_estimators=489)


# In[35]:


start = time.time()
modelgbmTuned = modelgbmTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[36]:


ygbm=modelgbmTuned.predict(X_test)
ygbmtest=modelgbmTuned.predict(test)


# In[37]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[38]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[39]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[40]:


modelrf = rf()


# In[41]:


start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[42]:


classes = modelrf.predict(X_test)


# In[43]:


(classes == y_test).sum()/y_test.size 


# In[44]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    rf(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[45]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[46]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[47]:


modelrfTuned=rf(criterion="gini",
               max_depth=88,
               max_features=41,
               min_weight_fraction_leaf=0.1,
               n_estimators=285)


# In[48]:


start = time.time()
modelrfTuned = modelrfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[49]:


yrf=modelrfTuned.predict(X_test)
yrf


# In[50]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[51]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[52]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[53]:


modelneigh = KNeighborsClassifier(n_neighbors=7)


# In[54]:


start = time.time()
modelneigh = modelneigh.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[55]:


classes = modelneigh.predict(X_test)

classes


# In[56]:


(classes == y_test).sum()/y_test.size 


# In[57]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    KNeighborsClassifier(
       n_neighbors=7         # No need to tune this parameter value
      ),
    {"metric": ["euclidean", "cityblock"]},
    n_iter=32,            # How many points to sample
    cv = 2            # Number of cross-validation folds
   )


# In[58]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[59]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[60]:


modelneighTuned = KNeighborsClassifier(n_neighbors=7,
               metric="cityblock")


# In[61]:


start = time.time()
modelneighTuned = modelneighTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[62]:


yneigh=modelneighTuned.predict(X_test)


# In[63]:


yneightest=modelneighTuned.predict(test)


# In[64]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[65]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[66]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[67]:


modeletf = ExtraTreesClassifier()


# In[68]:


start = time.time()
modeletf = modeletf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[69]:


classes = modeletf.predict(X_test)

classes


# In[70]:


(classes == y_test).sum()/y_test.size


# In[71]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    ExtraTreesClassifier( ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {   'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    n_iter=32,            # How many points to sample
    cv = 2            # Number of cross-validation folds
)


# In[72]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[73]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[74]:


modeletfTuned=ExtraTreesClassifier(criterion="entropy",
               max_depth=100,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=100)


# In[75]:


start = time.time()
modeletfTuned = modeletfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[76]:


yetf=modeletfTuned.predict(X_test)
yetftest=modeletfTuned.predict(test)


# In[77]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[78]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[79]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[80]:


modelxgb=XGBClassifier()


# In[81]:


start = time.time()
modelxgb = modelxgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[82]:


classes = modelxgb.predict(X_test)

classes


# In[83]:


(classes == y_test).sum()/y_test.size 


# In[84]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    XGBClassifier(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 300),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[85]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[86]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[87]:


modelxgbTuned=XGBClassifier(criterion="gini",
               max_depth=85,
               max_features=47,
               min_weight_fraction_leaf=0.035997,
               n_estimators=178)


# In[88]:


start = time.time()
modelxgbTuned = modelxgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[89]:


#yxgb=modelxgbTuned.predict(X_test)
#yxgbtest=modelxgbTuned.predict(test)


# In[90]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[91]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[92]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[93]:


modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)


# In[94]:


start = time.time()
modellgb = modellgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[95]:


classes = modellgb.predict(X_test)

classes


# In[96]:


(classes == y_test).sum()/y_test.size 


# In[97]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    lgb.LGBMClassifier(
       n_jobs = 2         # No need to tune this parameter value
      ),

    # 2.12 Specify estimator parameters that you would like to change/tune
    {
        'n_estimators': (100, 500),           # Specify integer-values parameters like this
        'criterion': ['gini', 'entropy'],     # Specify categorical parameters as here
        'max_depth': (4, 100),                # integer valued parameter
        'max_features' : (10,64),             # integer-valued parameter
        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   # Float-valued parameter
    },

    # 2.13
    n_iter=32,            # How many points to sample
    cv = 3                # Number of cross-validation folds
)


# In[98]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[99]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[100]:


modellgbTuned = lgb.LGBMClassifier(criterion="entropy",
               max_depth=35,
               max_features=14,
               min_weight_fraction_leaf=0.18611,
               n_estimators=148)


# In[101]:


start = time.time()
modellgbTuned = modellgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[102]:


ylgb=modellgbTuned.predict(X_test)
ylgbtest=modellgbTuned.predict(test)


# In[103]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[104]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[105]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[106]:


NewTrain = pd.DataFrame()
#NewTrain['yrf'] = yrf.tolist()
NewTrain['yetf'] = yetf.tolist()
NewTrain['yneigh'] = yneigh.tolist()
NewTrain['ygbm'] = ygbm.tolist()
#NewTrain['yxgb'] = yxgb.tolist()
NewTrain['ylgb'] = ylgb.tolist()

NewTrain.head(5), NewTrain.shape


# In[107]:


NewTest = pd.DataFrame()
#NewTest['yrf'] = yrftest.tolist()
NewTest['yetf'] = yetftest.tolist()
NewTest['yneigh'] = yneightest.tolist()
NewTest['ygbm'] = ygbmtest.tolist()
#NewTest['yxgb'] = yxgbtest.tolist()
NewTest['ylgb'] = ylgbtest.tolist()
NewTest.head(5), NewTest.shape


# In[108]:


NewModel=rf(criterion="entropy",
               max_depth=87,
               max_features=4,
               min_weight_fraction_leaf=0.0,
               n_estimators=600)


# In[109]:


start = time.time()
NewModel = NewModel.fit(NewTrain, y_test)
end = time.time()
(end-start)/60


# In[110]:


ypredict=NewModel.predict(NewTest)
ypredict


# In[111]:



#submit=pd.DataFrame({'Id': ids, 'Target': ylgbtest})
submit=pd.DataFrame({'Id': ids, 'Target': ypredict})
submit.head(5)


# In[112]:


submit.to_csv('submit.csv', index=False)


# In[113]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = ypredict
sub.drop(sub.columns[[1]], axis=1, inplace=True)
sub.to_csv('submission.csv',index=False)

