#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Utilities from kaggle kernels
# Instead of data = pd.read_csv("../input/train_V2.csv")
# We use : data = read_fast("../input/train_V2.csv")
import random
import time

def reduce_mem_usage_func(df):
    """ Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def get_sampled_data(filename, sample_size):
    n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
    skip = sorted(random.sample(range(1,n+1),n-sample_size)) #the 0-indexed header will not be included in the skip list
    df = pd.read_csv(filename, skiprows=skip)
    return df


def read_fast(filename, sample=True, sample_size=500000, reduce_mem_usage=True):
    start_time = time.time()
    df = get_sampled_data(filename, sample_size) if sample else pd.read_csv(filename)
    new_df = reduce_mem_usage_func(df) if reduce_mem_usage else df
    elapsed_time = int(time.time() - start_time)
    print('Time to get data frame: {:02d}:{:02d}:{:02d}'.format(
               elapsed_time // 3600,
               (elapsed_time % 3600 // 60),
               elapsed_time % 60))
    return new_df


# In[ ]:


train = read_fast("../input/train.csv")
train.head()


# In[ ]:


test = read_fast("../input/test.csv", sample = False)
test.head()


# In[ ]:


# check index of dataframe
train.columns


# In[ ]:


fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(train.corr(), cmap ='RdBu')


# In[ ]:


train = train.dropna(thresh=0.70*len(train), axis=1)

test = test.dropna(thresh=0.70*len(test), axis=1)
train = train.drop(['SMode'], axis = 1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ( 'ProductName', 'EngineVersion', 'AppVersion')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))
    lbl.fit(list(test[c].values)) 
    test[c] = lbl.transform(list(test[c].values))


# In[ ]:


train = train.select_dtypes(include=[np.number])


# In[ ]:


train = train.fillna(train.mean())


# In[ ]:


y = train["HasDetections"]

X = train.drop(labels = ["HasDetections"],axis = 1)


# In[ ]:


#from lightgbm import LGBMClassifier
#from sklearn.model_selection import train_test_split
#
#def identify_zero_importance_features(X, y, iterations = 1):
#    """
#    Identify zero importance features in a training dataset based on the 
#    feature importances from a gradient boosting model. 
#    
#    Parameters
#    --------
#    train : dataframe
#        Training features
#        
#    train_labels : np.array
#        Labels for training data
#        
#    iterations : integer, default = 2
#        Number of cross validation splits to use for determining feature importances
#    """
#    
#    # Initialize an empty array to hold feature importances
#    feature_importances = np.zeros(X.shape[1])
#
#    # Create the model with several hyperparameters
#    model = LGBMClassifier(objective = 'binary',num_leaves=60,
#                        learning_rate=0.01,
#                        n_estimators=700,
#                     max_bin=55, boosting = 'gbdt',
#                              bagging_fraction=0.8,
#                              bagging_freq=1, 
#                              feature_fraction=0.8,
#                              feature_fraction_seed=9, 
#                              bagging_seed=11,metric = 'auc',
#                              min_data_in_leaf=60, 
#                              min_sum_hessian_in_leaf=2)
#    
#    # Fit the model multiple times to avoid overfitting
#    for i in range(iterations):
#
#        # Split into training and validation set
#        train_features, valid_features, train_y, valid_y = train_test_split(X, y, 
#                                                                            test_size = 0.25, 
#                                                                            random_state = i)
#
#        # Train using early stopping
#        model.fit(train_features, train_y, early_stopping_rounds=100, 
#                  eval_set = [(valid_features, valid_y)])
#
#        # Record the feature importances
#        feature_importances += model.feature_importances_ / iterations
#    
#    feature_importances = pd.DataFrame({'feature': list(X.columns), 
#                            'importance': feature_importances}).sort_values('importance', 
#                                                                            ascending = False)
#    
#    # Find the features with zero importance
#   zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
#    print('\nThere are %d features with 0.0 importance' % len(zero_features))
#    
#    return zero_features, feature_importances

#zero_features, feature_importances = identify_zero_importance_features(X, y, iterations = 1)
#print('zero_features:',zero_features)
#print('feature_importances : ', feature_importances)


# In[ ]:


#feature_importances.describe()


# In[ ]:


#pp =np.percentile(feature_importances['importance'], 20) 
#print(pp)


# In[ ]:


#to_drop = feature_importances[feature_importances['importance'] <= pp]['feature']
#X = X.drop(columns = to_drop)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)             


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
   # Cross validate model with Kfold stratified cross val
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)


# In[ ]:


# RFC Parameters tunning 
#from sklearn.ensemble import RandomForestClassifier

#RFC = RandomForestClassifier()



## Search grid for optimal parameters
#rf_param_grid = {"max_depth":  [n for n in range(9, 12)],  
 #             "max_features": [1, 3, 10],
  #            "min_samples_split": [n for n in range(4, 9)],
   #           "min_samples_leaf": [n for n in range(2, 5)],
    #          "bootstrap": [False],
     #         "n_estimators" :[n for n in range(10, 20, 10)],
      #        "criterion": ["gini"]}


#gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = -1)

#gsRFC.fit(X_train,y_train)

#RFC_best = gsRFC.best_estimator_

# Best score
#gsRFC.best_score_


# In[ ]:


#lgbm 
import lightgbm as lgb
lbm = lgb.LGBMClassifier()


## Search grid for optimal parameters
lbm_param_grid = model_params = {
        
        "objective": ["binary"],
        "boosting_type": ["gbdt"], 
        "learning_rate":[ 0.05],
        "max_depth": [8],
        "num_leaves": [120],
        "n_estimators": [1000],
        "bagging_fraction": [0.7],
        "feature_fraction": [0.7],
        "bagging_freq": [5],
        "bagging_seed": [2018],
        'min_child_samples':[ 80], 
        'min_child_weight': [100.0], 
        'min_split_gain': [0.1], 
        'reg_alpha': [0.005], 
        'reg_lambda': [0.1], 
        'subsample_for_bin': [25000], 
        'min_data_per_group': [100], 
        'max_cat_to_onehot': [4], 
        'cat_l2':[ 25.0], 
        'cat_smooth':[ 2.0], 
        'max_cat_threshold':[ 32], 
        "random_state": [1],
        "silent": [True],
        "metric": ["auc"],
    }


gsExtC = GridSearchCV(lbm,param_grid = lbm_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_


# In[ ]:


# RFC Parameters tunning 
#from sklearn.ensemble import RandomForestClassifier
#
#RFC = RandomForestClassifier()
#
#
#
### Search grid for optimal parameters
#rf_param_grid = {"max_depth": [None],
#              "max_features": [1, 3, 10],
#              "min_samples_split": [2, 3, 10],
#              "min_samples_leaf": [1, 3, 10],
#              "bootstrap": [False],
#              "n_estimators" :[100,300],
#              "criterion": ["gini"]}
#
#
#gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
#
#gsRFC.fit(X_train,y_train)
#
#RFC_best = gsRFC.best_estimator_
#
## Best score
#gsRFC.best_score_


# In[ ]:


# Adaboost
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#
#DTC = DecisionTreeClassifier()
#
#adaDTC = AdaBoostClassifier(DTC, random_state=7)
#
#ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
#              "base_estimator__splitter" :   ["best", "random"],
#              "algorithm" : ["SAMME","SAMME.R"],
#              "n_estimators" :[30],
#              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
#
#gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
#
#gsadaDTC.fit(X_train,y_train)
#
#ada_best = gsadaDTC.best_estimator_
#
#gsadaDTC.best_score_


# In[ ]:


#Gradient boosting tunning
#from sklearn.ensemble import GradientBoostingClassifier
#
#GBC = GradientBoostingClassifier()
#gb_param_grid = {'loss' : ["deviance"],
#              'n_estimators' : [n for n in range(10, 60, 10)],
#              'learning_rate': [0.1, 0.05, 0.01],
#              'max_depth':  [n for n in range(9, 14)],  
#              'min_samples_leaf': [n for n in range(2, 5)],
#              'max_features': [0.3, 0.1] 
#              }
#
#gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
#
#gsGBC.fit(X_train,y_train)
#
#GBC_best = gsGBC.best_estimator_
#
## Best score
#gsGBC.best_score_


# In[ ]:


#from sklearn.ensemble import VotingClassifier
#
#votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
#('gbc',GBC_best)], voting='soft', n_jobs=4)
#
#votingC = votingC.fit(X_train, y_train)
#


# In[ ]:


test_id = test['MachineIdentifier']


# In[ ]:


feats = test.drop(['MachineIdentifier'], axis=1)


# In[ ]:


#test = test.drop(['MachineIdentifier'], axis = 1)
feats = feats.select_dtypes(include=[np.number])

feats = feats[X_train.columns]


# In[ ]:


feats = feats.fillna(feats.mean())


# In[ ]:


predictions = gsExtC.predict(feats)


# In[ ]:


submission = pd.DataFrame()
submission['MachineIdentifier'] = test_id
submission['HasDetections'] = predictions 
submission.to_csv('submission1.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:




