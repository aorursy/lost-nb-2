#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[2]:


#Load the training data
train_csv_file = '../input/train.csv'
train_df = pd.read_csv(train_csv_file, nrows=100)


# In[3]:


#Better understand at the data
#train_df.describe()


# In[4]:


#Peak at the data
#train_df.head()


# In[5]:


#Pre-Processing Remove Outliers
#def pre_process_outliers(process_df):
   # for col in train.columns:
    #    if len(train[col].unique()) < 1000:
            #drop the col
            


# In[6]:


#Pre-Processing Scaling
def pre_process_scale(process_df):
    scaler=StandardScaler()
    return pd.DataFrame(scaler.fit_transform(process_df))
        


# In[7]:


#Pre-processing PCA
def pre_process_pca(process_df):
    pca=PCA(n_components=5) 
    pca_scaled_df = pca.fit_transform(process_df) 
    #let's check the shape of X_pca array
    ex_variance=np.var(pca_scaled_df,axis=0)
    ex_variance_ratio = ex_variance/np.sum(ex_variance)
    print( pca.explained_variance_ratio_.cumsum())
    return pca_scaled_df


# In[8]:


#Model data
#Try a quick and dirty ann until I can take a better look at things more closely
scaled_df = pre_process_scale(train_df.iloc[0:, 1:257])
pca_scaled_df = pre_process_pca(scaled_df)
y = train_df.iloc[0:, 257:258].values.ravel()
#y = np.ravel(train_df.iloc[:, 257:258])
kf = KFold(n_splits=100)

clf = MLPClassifier(hidden_layer_sizes=(130, 60 10, 4), max_iter=20000, alpha=1e-5,solver='sgd', random_state=21,tol=0.000000001)

#Try a few alternativ approaches to the dataset
model_df = scaled_df
#model_df = pca_scaled_df

clf.fit(model_df,y)
for train_indices, test_indices in kf.split(model_df):
    clf.fit(model_df.loc[train_indices], y[train_indices])  
y_pred = clf.predict(model_df, y)
measure_performance(y, y_pred[0:])


# In[ ]:





# In[9]:


#Measure accuracy score
def measure_performance(y_test, y_prediction):
    print(accuracy_score(y_test, y_prediction))


# In[10]:


#cm = confusion_matrix(y_test, y_prediction)
#cm


# In[11]:


#Now time to test the model
#Load the test data
test_csv_file = '../input/test.csv'
test_df = pd.read_csv(test_csv_file)

scaled_df = pre_process_scale(test_df.iloc[:, 1:])
model_df = scaled_df
y_test_pred = classifier.predict(model_df)


# In[12]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = y_test_pred
sub.to_csv('submission.csv',index=False)

