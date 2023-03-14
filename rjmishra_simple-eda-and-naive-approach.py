#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set()


# In[2]:


print("Files in the folder:")
print(os.listdir("../input"))


# In[3]:


train = pd.read_csv('../input/X_train.csv')
test = pd.read_csv('../input/X_test.csv')


# In[4]:


train.head()


# In[5]:


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,5,figsize=(16,8))

    for feature in features:
        i += 1
        plt.subplot(2,5,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();

def plot_feature_class_distribution(classes,tt, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(5,2,figsize=(16,24))

    for feature in features:
        i += 1
        plt.subplot(5,2,i)
        for clas in classes:
            ttc = tt[tt['surface']==clas]
            sns.kdeplot(ttc[feature], bw=0.5,label=clas)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();
    


# In[6]:


features = train.columns.values[3:]
plot_feature_distribution(train, test, 'train', 'test', features)


# In[7]:


labels = pd.read_csv('../input/y_train.csv')


# In[8]:


classes = (labels['surface'].value_counts()).index
aux = train.merge(labels, on='series_id', how='inner')
plot_feature_class_distribution(classes, aux, features)


# In[9]:


# first drop columns such as measurement and row_id in training data
train_d = train.drop(['row_id', 'measurement_number'], axis=1)
test_d = test.drop(['row_id','measurement_number'], axis=1)
train_f = train_d.groupby('series_id').agg(['min', 'max', 'mean', 'median', 'var'])
test_f = test_d.groupby('series_id').agg(['min', 'max', 'mean', 'median', 'var'])


# In[10]:


# lets see what we got
train_f.head()


# In[11]:


test_f.head()


# In[12]:


# I am going to write to temporary files and then use them for model building
# add surface to the training points
#aux = train_f.merge(labels, on='series_id', how='inner')
#aux.to_csv('training.csv', index=False, header=None)
#test_f.to_csv(input/testing.csv', index=False, header=None)
train_f['surface'] = labels['surface']


# In[13]:


train_f.head()


# In[14]:


train_f.to_csv('training.csv', index=False, header=None)
test_f.to_csv('testing.csv', index=False, header=None)


# In[15]:


# load the traing data 
data = pd.read_csv('training.csv', header=None)


# In[16]:


test_data = pd.read_csv('testing.csv', header=None)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# lets use GBM and AdaBoost, see how these work out
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[18]:


from sklearn.tree import DecisionTreeClassifier


# In[19]:


# First lets see AdaBoost  
model_ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7), random_state=41)


# In[20]:


train, valid = train_test_split(data, test_size=0.2)


# In[21]:


model_ab.fit(train.loc[:, 0:49], train[50])


# In[22]:


predict_train = model_ab.predict(train.loc[:, 0:49])
print(classification_report(train[50], predict_train))
confusion_matrix(train[50], predict_train)


# In[23]:


predict_valid = model_ab.predict(valid.loc[:, 0:49])
print(classification_report(valid[50], predict_valid))
confusion_matrix(valid[50], predict_valid)


# In[24]:


test_predict = model_ab.predict(test_data)


# In[25]:


# read sample submission file
submit = pd.read_csv('../input/sample_submission.csv')


# In[26]:


submit['surface'] = test_predict


# In[27]:


submit.head()


# In[28]:


submit.to_csv('naive_submission.csv', index=False)


# In[29]:


more naive_submission.csv


# In[30]:




