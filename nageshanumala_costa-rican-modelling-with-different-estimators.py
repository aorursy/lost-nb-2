#!/usr/bin/env python
# coding: utf-8

# In[1]:


# essential libraries
import numpy as np 
import pandas as pd
# for data visulization
import matplotlib.pyplot as plt
import seaborn as sns


#for data processing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.compose import ColumnTransformer as ct
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# for modeling estimators
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbm
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

# for measuring performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#for tuning parameters
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
from eli5.sklearn import PermutationImportance

# Misc.
import os
import time
import gc
import random
from scipy.stats import uniform
import warnings


# In[2]:


pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


ids=test['Id']


# In[4]:


train.head()


# In[5]:


train.shape, test.shape


# In[6]:


train.info()   


# In[7]:


test


# In[8]:


sns.countplot("Target", data=train)


# In[9]:


sns.countplot(x="r4t3",hue="Target",data=train)


# In[10]:


sns.countplot(x="v18q",hue="Target",data=train)


# In[11]:


sns.countplot(x="v18q1",hue="Target",data=train)


# In[12]:


sns.countplot(x="tamhog",hue="Target",data=train)


# In[13]:


sns.countplot(x="hhsize",hue="Target",data=train)


# In[14]:


sns.countplot(x="abastaguano",hue="Target",data=train)


# In[15]:


sns.countplot(x="noelec",hue="Target",data=train)


# In[16]:


train.select_dtypes('object').head()


# In[17]:




yes_no_map = {'no':0,'yes':1}
train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)
train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)
train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)
    
    


# In[18]:


yes_no_map = {'no':0,'yes':1}
test['dependency'] = test['dependency'].replace(yes_no_map).astype(np.float32)
test['edjefe'] = test['edjefe'].replace(yes_no_map).astype(np.float32)
test['edjefa'] = test['edjefa'].replace(yes_no_map).astype(np.float32)


# In[19]:


train[["dependency","edjefe","edjefa"]].describe()


# In[20]:


# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[21]:


train['v18q1'] = train['v18q1'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)


# In[22]:


train['v2a1'] = train['v2a1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)


# In[23]:


train['rez_esc'] = train['rez_esc'].fillna(0)
test['rez_esc'] = test['rez_esc'].fillna(0)
train['SQBmeaned'] = train['SQBmeaned'].fillna(0)
test['SQBmeaned'] = test['SQBmeaned'].fillna(0)
train['meaneduc'] = train['meaneduc'].fillna(0)
test['meaneduc'] = test['meaneduc'].fillna(0)


# In[24]:


#Checking for missing values again to confirm that no missing values present
# Number of missing in each column
missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[25]:


#Checking for missing values again to confirm that no missing values present
# Number of missing in each column
missing = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(train)

missing.sort_values('percent', ascending = False).head(10)


# In[26]:


train.drop(['Id','idhogar'], inplace = True, axis =1)

test.drop(['Id','idhogar'], inplace = True, axis =1)


# In[27]:


train.shape


# In[28]:


test.shape


# In[29]:


y = train.iloc[:,140]
y.unique()


# In[30]:


X = train.iloc[:,1:141]
X.shape


# In[31]:


my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
scale = ss()
X = scale.fit_transform(X)
#pca = PCA(0.95)
#X = pca.fit_transform(X)


# In[32]:


X.shape


# In[33]:


#subjecting the same to test data
my_imputer = SimpleImputer()
test = my_imputer.fit_transform(test)
scale = ss()
test = scale.fit_transform(test)
#pca = PCA(0.95)
#test = pca.fit_transform(test)


# In[34]:


X.shape, y.shape,test.shape


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)


# In[36]:



modelrf = rf()


# In[37]:


start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[38]:


classes = modelrf.predict(X_test)


# In[39]:


(classes == y_test).sum()/y_test.size 


# In[40]:


f1 = f1_score(y_test, classes, average='macro')
f1


# In[41]:


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


# In[42]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[43]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[44]:


modelrfTuned=rf(criterion="entropy",
               max_depth=77,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[45]:


start = time.time()
modelrfTuned = modelrfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[46]:


yrf=modelrfTuned.predict(X_test)


# In[47]:


yrf


# In[48]:


yrftest=modelrfTuned.predict(test)


# In[49]:


yrftest


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


modeletf = ExtraTreesClassifier()


# In[54]:


start = time.time()
modeletf = modeletf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[55]:


classes = modeletf.predict(X_test)

classes


# In[56]:


(classes == y_test).sum()/y_test.size


# In[57]:


f1 = f1_score(y_test, classes, average='macro')
f1


# In[58]:


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


# In[59]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[60]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[61]:


modeletfTuned=ExtraTreesClassifier(criterion="entropy",
               max_depth=100,
               max_features=64,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[62]:


start = time.time()
modeletfTuned = modeletfTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[63]:


yetf=modeletfTuned.predict(X_test)


# In[64]:


yetftest=modeletfTuned.predict(test)


# In[65]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[66]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[67]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[68]:


modelneigh = KNeighborsClassifier(n_neighbors=4)


# In[69]:


start = time.time()
modelneigh = modelneigh.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[70]:


classes = modelneigh.predict(X_test)

classes


# In[71]:


(classes == y_test).sum()/y_test.size 


# In[72]:


f1 = f1_score(y_test, classes, average='macro')
f1


# In[73]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    KNeighborsClassifier(
       n_neighbors=4         # No need to tune this parameter value
      ),
    {"metric": ["euclidean", "cityblock"]},
    n_iter=32,            # How many points to sample
    cv = 2            # Number of cross-validation folds
   )


# In[74]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[75]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[76]:


modelneighTuned = KNeighborsClassifier(n_neighbors=4,
               metric="cityblock")


# In[77]:


start = time.time()
modelneighTuned = modelneighTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[78]:


yneigh=modelneighTuned.predict(X_test)


# In[79]:


yneightest=modelneighTuned.predict(test)


# In[80]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[81]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[82]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[83]:


modelgbm=gbm()


# In[84]:


start = time.time()
modelgbm = modelgbm.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[85]:


classes = modelgbm.predict(X_test)

classes


# In[86]:


(classes == y_test).sum()/y_test.size 


# In[87]:


f1 = f1_score(y_test, classes, average='macro')
f1


# In[88]:


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


# In[89]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[90]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[91]:


modelgbmTuned=gbm(
               max_depth=84,
               max_features=11,
               min_weight_fraction_leaf=0.04840,
               n_estimators=489)


# In[92]:


start = time.time()
modelgbmTuned = modelgbmTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[93]:


ygbm=modelgbmTuned.predict(X_test)


# In[94]:


ygbmtest=modelgbmTuned.predict(test)


# In[95]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[96]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[97]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[98]:


modelxgb=XGBClassifier()


# In[99]:


start = time.time()
modelxgb = modelxgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[100]:


classes = modelxgb.predict(X_test)

classes


# In[101]:


(classes == y_test).sum()/y_test.size 


# In[102]:


f1 = f1_score(y_test, classes, average='macro')
f1


# In[103]:


bayes_cv_tuner = BayesSearchCV(
    #  Place your estimator here with those parameter values
    #      that you DO NOT WANT TO TUNE
    XGBClassifier(
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


# In[104]:


# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[105]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[106]:


modelxgbTuned=XGBClassifier(criterion="gini",
               max_depth=4,
               max_features=15,
               min_weight_fraction_leaf=0.05997,
               n_estimators=499)


# In[107]:


start = time.time()
modelxgbTuned = modelxgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[108]:


yxgb=modelxgbTuned.predict(X_test)


# In[109]:


yxgbtest=modelxgbTuned.predict(test)


# In[110]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[111]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[112]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[113]:


modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)


# In[114]:


start = time.time()
modellgb = modellgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[115]:


classes = modellgb.predict(X_test)

classes


# In[116]:


(classes == y_test).sum()/y_test.size 


# In[117]:


f1 = f1_score(y_test, classes, average='macro')
f1


# In[118]:


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


# In[119]:



# Start optimization
bayes_cv_tuner.fit(X_train, y_train)


# In[120]:


#  Get list of best-parameters
bayes_cv_tuner.best_params_


# In[121]:


modellgbTuned = lgb.LGBMClassifier(criterion="gini",
               max_depth=5,
               max_features=53,
               min_weight_fraction_leaf=0.01674,
               n_estimators=499)


# In[122]:


start = time.time()
modellgbTuned = modellgbTuned.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[123]:


ylgb=modellgbTuned.predict(X_test)


# In[124]:


ylgbtest=modellgbTuned.predict(test)


# In[125]:


#  Get what average accuracy was acheived during cross-validation
bayes_cv_tuner.best_score_


# In[126]:


#  What accuracy is available on test-data
bayes_cv_tuner.score(X_test, y_test)


# In[127]:


#  And what all sets of parameters were tried?
bayes_cv_tuner.cv_results_['params']


# In[128]:


NewTrain = pd.DataFrame()
NewTrain['yrf'] = yrf.tolist()
NewTrain['yetf'] = yetf.tolist()
NewTrain['yneigh'] = yneigh.tolist()
NewTrain['ygbm'] = ygbm.tolist()
NewTrain['yxgb'] = yxgb.tolist()
NewTrain['ylgb'] = ylgb.tolist()

NewTrain.head(5), NewTrain.shape


# In[129]:


NewTest = pd.DataFrame()
NewTest['yrf'] = yrftest.tolist()
NewTest['yetf'] = yetftest.tolist()
NewTest['yneigh'] = yneightest.tolist()
NewTest['ygbm'] = ygbmtest.tolist()
NewTest['yxgb'] = yxgbtest.tolist()
NewTest['ylgb'] = ylgbtest.tolist()
NewTest.head(5), NewTest.shape


# In[130]:


NewModel=rf(criterion="entropy",
               max_depth=77,
               max_features=6,
               min_weight_fraction_leaf=0.0,
               n_estimators=500)


# In[131]:


start = time.time()
NewModel = NewModel.fit(NewTrain, y_test)
end = time.time()
(end-start)/60


# In[132]:


ypredict=NewModel.predict(NewTest)


# In[133]:


ylgbtest


# In[134]:


submit=pd.DataFrame({'Id': ids, 'Target': ylgbtest})
submit.head(5)


# In[135]:


submit.to_csv('submit.csv', index=False)

