#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import os.path

import numpy as np
import pandas as pd

from datetime import timedelta 
from datetime import datetime

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from category_encoders import TargetEncoder

import shap

from catboost import CatBoostClassifier, Pool

from hyperopt import fmin, hp, tpe

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[2]:


# import data
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv", index_col='id')
test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv", index_col='id')
sample = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")


# In[3]:


## I'm not going to spend much time worrying about the preparing the data at this time
## I'll reuse some code from other excellent notebooks to save time


# In[4]:


# Source: https://www.kaggle.com/vikassingh1996/don-t-underestimate-the-power-of-a-logistic-reg

'''Variable Description'''
def description(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['PercMissing'] = df.isnull().sum().values / df.isnull().count().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.iloc[0].values
    summary['Second Value'] = df.iloc[1].values
    summary['Third Value'] = df.iloc[2].values
    return summary
print('**Variable Description of  train Data:**')
description(train)


# In[5]:


# Source: https://www.kaggle.com/vikassingh1996/don-t-underestimate-the-power-of-a-logistic-reg

## To start, let's just replace all null values with the mode of that column
def replace_nan(data):
    for column in data.columns:
        if data[column].isna().sum() > 0:
            data[column] = data[column].fillna(data[column].mode()[0])


replace_nan(train)
replace_nan(test)


# In[6]:


target = train.pop('target')
target.shape


# In[7]:


## Source: https://www.kaggle.com/carlodnt/catboost-shap-fastai

# bin_3
train['bin_3'] = train['bin_3'].apply(lambda x: 0 if x == 'F' else 1)
test['bin_3'] = test['bin_3'].apply(lambda x: 0 if x == 'F' else 1)

# bin_4
train['bin_4'] = train['bin_4'].apply(lambda x: 0 if x == 'N' else 1)
test['bin_4'] = test['bin_4'].apply(lambda x: 0 if x == 'N' else 1)

# ord_1
train.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)
test.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)

# ord_2
train.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)
test.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

# ord_3
train.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)
test.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)

# ord_4
train.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 
                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                                  22, 23, 24, 25], inplace = True)
test.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 
                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                                  22, 23, 24, 25], inplace = True)

high_card = ['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9','ord_5']
for col in high_card:
    enc_nom = (train.groupby(col).size()) / len(train)
    train[f'{col}'] = train[col].apply(lambda x: hash(str(x)) % 5000)
    test[f'{col}'] = test[col].apply(lambda x: hash(str(x)) % 5000)


# In[8]:


train.shape, test.shape


# In[9]:


# create a training and validation set
X_train, X_validation, y_train, y_validation = train_test_split(train, target, train_size=0.8, random_state=42)

X_test = test.copy()


# In[10]:


X_train.shape, X_validation.shape, y_train.shape, y_validation.shape, X_test.shape


# In[11]:


categorical_features_indices = np.where(train.dtypes != np.float)[0]
categorical_features_indices


# In[12]:


## Source: https://www.kaggle.com/lucamassaron/catboost-in-action-with-dnn

# Initializing a CatBoostClassifier with best parameters
best_params = {'bagging_temperature': 0.8,
               'depth': 5,
               'iterations': 500,
               'l2_leaf_reg': 30,
               'learning_rate': 0.05,
               'random_strength': 0.8}


# In[13]:


model = CatBoostClassifier(
       **best_params,
       loss_function='Logloss',
       eval_metric='AUC',         
#         task_type="GPU",
       nan_mode='Min',
       verbose=False
   )


# In[14]:


model.fit(
        X_train, y_train,
        verbose_eval=100, 
        early_stopping_rounds=50,
        cat_features=categorical_features_indices,
        eval_set=(X_validation, y_validation),
        use_best_model=False,
        plot=True
);


# In[15]:


shap.initjs()


# In[16]:


explainer = shap.TreeExplainer(model)


# In[17]:


shap_values = explainer.shap_values(Pool(X_train, y_train, cat_features=categorical_features_indices))


# In[18]:


shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])


# In[19]:


shap.force_plot(explainer.expected_value, shap_values[4,:], X_train.iloc[4,:])


# In[20]:


# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values[0:50,:], X_train.iloc[0:50,:])


# In[21]:


# feature importance plot
shap.summary_plot(shap_values, X_train, plot_type="bar")


# In[22]:


# summarize the effects of all the features
shap.summary_plot(shap_values, X_train)


# In[ ]:




