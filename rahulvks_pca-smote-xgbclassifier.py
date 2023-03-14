#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# StandardScaler
from sklearn.preprocessing import StandardScaler
# used for feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# to handle imbalanced data set
from imblearn.over_sampling import SMOTE
from collections import Counter
# performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_cell_magic('time', '', "# Load data\ntrain = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')\ntest = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')\n\nprint(train.shape)\nprint(test.shape)")


# In[3]:


train.head()


# In[4]:


# Change boolean value to int so as to encode
train['bin_3'] = train['bin_3'].apply(lambda x: 1 if x=='T' else 0)
train['bin_4'] = train['bin_4'].apply(lambda x:1 if x =='Y' else 0)
test['bin_3'] = test['bin_3'].apply(lambda x:1 if x=='T' else 0)
test['bin_4'] = test['bin_4'].apply(lambda x:1 if x == 'Y' else 0)


# In[5]:


def replace_nan(data):
    for column in data.columns:
        if data[column].isna().sum() > 0:
            data[column] = data[column].fillna(data[column].mode()[0])


replace_nan(train)
replace_nan(test)


# In[6]:


features = []

for col in train.columns[:-1]:
    rd = LabelEncoder()
    rd.fit_transform( train[col].append( test[col] ) )
    train[col] = rd.transform( train[col] )
    test [col] = rd.transform( test [col] )
    features.append(col)


# In[7]:


print("Train Sample")
train.head()


# In[8]:


print("Test Sample")
test.head()


# In[9]:


## Train
X_data = train.iloc[:,0:24]
y_data = train.iloc[:,-1]


# In[10]:


standard_scaler = preprocessing.StandardScaler()
X_standard_scaled_df = standard_scaler.fit_transform(X_data)


# In[11]:


X_standard_scaled_df = pd.DataFrame(data=X_standard_scaled_df[:,:], columns=['id','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'day', 'month',])  


# In[12]:


# Make an instance of the Model
pca = PCA(10)
pca_selected = pca.fit_transform(X_standard_scaled_df)
pca_selected_df = pd.DataFrame(data=pca_selected[:,:])


# In[13]:


print('After PCA' , pca_selected_df.head())


# In[14]:


ready_data = pca_selected_df.join(y_data)


# In[15]:


print('After Target' , ready_data.head())


# In[16]:


data_class_0 = ready_data[ready_data['target']==0]
print(data_class_0.shape)
data_class_1 = ready_data[ready_data['target']==1]
print(data_class_1.shape)


# In[17]:


X_0 = data_class_0.iloc[:,0:-1]  #independent columns
y_0 = data_class_0.iloc[:,-1]    #target column i.e Class

X_1 = data_class_1.iloc[:,0:-1]  #independent columns
y_1 = data_class_1.iloc[:,-1]    #target column i.e Class

# def train_gen():
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, test_size=0.33, random_state=42)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.33, random_state=42)


# In[18]:


X_train = pd.concat([X_train_0, X_train_1])
y_train = pd.concat([y_train_0, y_train_1])
X_test = pd.concat([X_test_0 , X_test_1])
y_test = pd.concat([y_test_0 , y_test_1])

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[19]:


print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print('Resampled dataset shape using smote %s' % Counter(y_res))


# In[20]:


from xgboost import XGBClassifier


# In[21]:


XGB_classifier = XGBClassifier(max_depth=20,n_estimators=2020,colsample_bytree=0.20,learning_rate=0.020,objective='binary:logistic', n_jobs=-1)
XGB_classifier.fit(X_train,y_train, eval_metric = 'aucpr')
#aucpr: Area under the PR curve


# In[22]:


XGB_classifier_predict_smote = XGB_classifier.predict(X_test)
print(XGB_classifier.score(X_train,y_train))
print(np.sqrt(mean_squared_error(XGB_classifier_predict_smote,y_test)))


# In[23]:


accuracy_score(y_test,XGB_classifier_predict_smote)


# In[24]:


print(classification_report(y_test,XGB_classifier_predict_smote))


# In[25]:


test.head(1)


# In[26]:


standard_scaler = preprocessing.StandardScaler()
test_standard_scaled_df = standard_scaler.fit_transform(test)


# In[27]:


test_standard_scaled_df = pd.DataFrame(data=test_standard_scaled_df[:,:], columns=['id','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'day', 'month',]) 


# In[28]:


pca_selected = pca.fit_transform(test_standard_scaled_df)
pca_selected_df = pd.DataFrame(data=pca_selected[:,:])


# In[29]:


XGB_classifier.fit(X_train,y_train, eval_metric = 'aucpr')
test = XGB_classifier.predict(pca_selected_df)


# In[30]:


test_score = XGB_classifier.predict(pca_selected_df)


# In[31]:


sample_submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")
sample_submission.shape


# In[32]:


sample_submission['target'] = test_score


# In[33]:


sample_submission.to_csv('submission_xgboost_v1.csv', index=False)

