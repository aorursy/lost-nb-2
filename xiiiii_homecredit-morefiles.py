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

# matplotlib
import matplotlib.pyplot as plt


# In[ ]:


# Training data
app_train = pd.read_csv('../input/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()


# In[ ]:


# Testing data
app_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()


# In[ ]:


# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# In[ ]:


train_labels = app_train['TARGET']

# Align the training and testing data. Keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join='inner', axis=1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# In[ ]:


app_train['DAYS_EMPLOYED'].describe()


# In[ ]:


app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
plt.xlabel('Days Empployment')


# In[ ]:


# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
app_test['DAYS_EMPLOYED'].replace({356243: np.nan}, inplace=True)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Drop the target from the training data
train = app_train.drop(columns=['TARGET'])

# Copy the testing data
test = app_test.copy()

# Median imputation of missing values
imputer = SimpleImputer(strategy='median')
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)


# In[ ]:


from sklearn.model_selection import train_test_split

# split the training set
X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.15, random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression

# Make the mode with the specified regularization parameter
log_reg = LogisticRegression()

# Train on the training data
log_reg.fit(X_train, y_train)


# In[ ]:


test_log_reg_pred = log_reg.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, test_log_reg_pred)


# In[ ]:


from sklearn.linear_model import LogisticRegression

# Make the mode with the specified regularization parameter
log_reg = LogisticRegression(C=1)

# Train on the training data
log_reg.fit(train, train_labels)


# In[ ]:


# Make predictions. Select 2nd column only for the probabiliy of not paying a loan
log_reg_pred = log_reg.predict_proba(test)[:, 1]


# In[ ]:


# Submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()


# In[ ]:


# Save the submission to a csv file
submit.to_csv('log_reg_baseline.csv', index=False)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
random_forest.fit(X_train, y_train)
test_random_forest_pred = random_forest.predict_proba(X_test)[:, 1]


# In[ ]:


roc_auc_score(y_test, test_random_forest_pred)


# In[ ]:


predictions = random_forest.predict_proba(test)[:, 1]


# In[ ]:


# Make a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline.csv', index=False)


# In[ ]:


from sklearn import tree.DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[ ]:


predictions = clf.predict_proba(test)[:, 1]


# In[ ]:


# Make a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('dt_baseline.csv', index=False)

