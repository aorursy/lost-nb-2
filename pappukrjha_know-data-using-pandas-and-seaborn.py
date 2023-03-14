#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

import random
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import os
print(os.listdir("../input"))

import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.


# In[2]:


# *** Read Data ***
df = pd.read_csv('../input/train.csv')
dfTest  = pd.read_csv('../input/test.csv')
dfSub   = pd.read_csv('../input/sample_submission.csv')


# In[3]:


# *** Data View ***
print(df.head())
print(dfSub.head())


# In[4]:


# *** First Data Impression ***
print('Train Data Info')
print(df.info())
print('\nTest Data Info \n')
print(dfTest.info())
print('\nSubmission Data Info \n')
print(dfSub.info())


# In[5]:


# *** Check Missing Values ***
print('Missing Values in Training Data')
print(df.isnull().sum())
print('Missing Values in Test Data')
print(df.isnull().sum())


# In[6]:


# *** Check Unique Values ***
print('Unique Values in Training Data')
print(df.nunique())
print('Unique Values in Test Data')
print(dfTest.nunique())


# In[7]:


# *** Explore Trainin Data ***
# scalar_coupling_constant
df['scalar_coupling_constant'].describe(percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


# In[8]:


# Plot the Distribution
sns.boxplot(df['scalar_coupling_constant'])


# In[9]:


# Check the dot plot to make the distribution even more clear
sns.stripplot(x = df['scalar_coupling_constant'])


# In[10]:


sns.distplot(df['scalar_coupling_constant'])


# In[11]:


# Distribution of scalar_coupling_constant by molecule_name
df.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index().head()


# In[12]:


df.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index()['scalar_coupling_constant'].         describe(percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


# In[13]:


sns.distplot(df.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index()['scalar_coupling_constant'])


# In[14]:


# Lets Explore the Atom Index Combination in Detail
(df['atom_index_0'].map(str) + '--' + df['atom_index_1'].map(str)).nunique()


# In[15]:


(df['atom_index_0'].map(str) + '--' + df['atom_index_1'].map(str)).value_counts().nlargest(10) 


# In[16]:


(df['atom_index_0'].map(str) + '--' + df['atom_index_1'].map(str)).value_counts().nsmallest(10) 


# In[17]:


df[df['atom_index_0'].map(str) == df['atom_index_1'].map(str)].shape


# In[18]:


# *** Distribution of Index 0 and 1
df['atom_index_0'].value_counts()


# In[19]:


df['atom_index_0'].value_counts(normalize = True)


# In[20]:


df['atom_index_1'].value_counts()


# In[21]:


df['atom_index_1'].value_counts(normalize = True)


# In[22]:


countDist = (df['atom_index_0'].map(str) + '--' + df['atom_index_1'].map(str)).value_counts().reset_index().              rename(columns = {0 : 'combinationCount'})
sns.distplot(countDist['combinationCount'])


# In[23]:


df['molecule_name'].value_counts().nlargest(10)


# In[24]:


df['molecule_name'].value_counts().reset_index()['molecule_name'].                          describe(percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


# In[25]:


ax = sns.boxplot(x=df['molecule_name'].value_counts().reset_index()['molecule_name'])


# In[26]:


# *** Simple Linear Regression Model ***
df['randomNum'] = df['id'].map(lambda x : random.uniform(0, 1))
dfTrain = df.query('randomNum <= 0.7')
dfValid  = df.query('randomNum > 0.7')
del dfTrain['randomNum']
del dfValid['randomNum']
del df['randomNum'] 


# In[27]:


molecule_name_risk = dfTrain.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index()
molecule_name_risk.rename(columns = {'scalar_coupling_constant' : 'molecule_name_risk'}, inplace = True)

dfTrain = pd.merge(dfTrain, molecule_name_risk, on = 'molecule_name')

index_0_risk = dfTrain.groupby('atom_index_0')['scalar_coupling_constant'].mean().reset_index()
index_0_risk.rename(columns = {'scalar_coupling_constant' : 'index_0_risk'}, inplace = True)

dfTrain = pd.merge(dfTrain, index_0_risk, on = 'atom_index_0')

index_1_risk = dfTrain.groupby('atom_index_1')['scalar_coupling_constant'].mean().reset_index()
index_1_risk.rename(columns = {'scalar_coupling_constant' : 'index_1_risk'}, inplace = True)

dfTrain = pd.merge(dfTrain, index_1_risk, on = 'atom_index_1')

type_risk = dfTrain.groupby('type')['scalar_coupling_constant'].mean().reset_index()
type_risk.rename(columns = {'scalar_coupling_constant' : 'type_risk'}, inplace = True)

dfTrain = pd.merge(dfTrain, type_risk, on = 'type')


# In[28]:


dfTrain.drop(['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], axis = 1, inplace = True)


# In[29]:


dfValid = pd.merge(dfValid, molecule_name_risk, on = 'molecule_name')
dfValid = pd.merge(dfValid, index_0_risk, on = 'atom_index_0')
dfValid = pd.merge(dfValid, index_1_risk, on = 'atom_index_1')
dfValid = pd.merge(dfValid, type_risk, on = 'type')


# In[30]:


dfValid.isnull().sum()


# In[31]:


print(dfValid.head())


# In[32]:


dfValid.drop(['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], axis = 1, inplace = True)


# In[33]:


molecule_name_risk.head()


# In[34]:


print(dfTrain.head())


# In[35]:


lmReg = LinearRegression()
lmReg.fit(dfTrain.drop(['id', 'scalar_coupling_constant'], axis = 1), dfTrain['scalar_coupling_constant'])
dfValid['scalar_coupling_constant_pred'] = lmReg.predict(dfValid.drop(['id', 'scalar_coupling_constant'], axis = 1))


# In[36]:


dfValid['predDiff'] = dfValid['scalar_coupling_constant'] - dfValid['scalar_coupling_constant_pred'] 


# In[37]:


dfValid['predDiff'].describe(percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


# In[38]:


dfValid['scalar_coupling_constant'].describe(percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


# In[39]:


# The coefficients
print('Coefficients: \n', lmReg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(dfValid['scalar_coupling_constant'], dfValid['scalar_coupling_constant_pred'] ))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(dfValid['scalar_coupling_constant'], dfValid['scalar_coupling_constant_pred']))


# In[40]:


# Plot outputs
sns.boxplot(x="variable", y="value", data=pd.melt(dfValid[['scalar_coupling_constant', 'scalar_coupling_constant_pred']]))


# In[41]:


ax = sns.lineplot(data = dfValid[['scalar_coupling_constant', 'scalar_coupling_constant_pred']])


# In[42]:


#dfTest = pd.merge(dfTest, molecule_name_risk, on = 'molecule_name')
dfTest = pd.merge(dfTest, index_0_risk, on = 'atom_index_0')
dfTest = pd.merge(dfTest, index_1_risk, on = 'atom_index_1')
dfTest = pd.merge(dfTest, type_risk, on = 'type')
dfTest['molecule_name_risk'] = 0


# In[43]:


dfTest['scalar_coupling_constant_pred'] = lmReg.predict(dfTest.drop(['id', 'molecule_name', 'atom_index_0', 'atom_index_1',                                                                     'type'], axis = 1))


# In[44]:


dfSub['scalar_coupling_constant'] = dfTest['scalar_coupling_constant_pred']


# In[45]:


dfSub.to_csv('sample_submission.csv', index = False)


# In[46]:


molecule_name_risk = df.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index()
molecule_name_risk.rename(columns = {'scalar_coupling_constant' : 'molecule_name_risk'}, inplace = True)

df = pd.merge(df, molecule_name_risk, on = 'molecule_name')

index_0_risk = df.groupby('atom_index_0')['scalar_coupling_constant'].mean().reset_index()
index_0_risk.rename(columns = {'scalar_coupling_constant' : 'index_0_risk'}, inplace = True)

df = pd.merge(df, index_0_risk, on = 'atom_index_0')

index_1_risk = df.groupby('atom_index_1')['scalar_coupling_constant'].mean().reset_index()
index_1_risk.rename(columns = {'scalar_coupling_constant' : 'index_1_risk'}, inplace = True)

df = pd.merge(df, index_1_risk, on = 'atom_index_1')

type_risk = df.groupby('type')['scalar_coupling_constant'].mean().reset_index()
type_risk.rename(columns = {'scalar_coupling_constant' : 'type_risk'}, inplace = True)

df = pd.merge(df, type_risk, on = 'type')

df.drop(['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], axis = 1, inplace = True)


# In[47]:


df.head()


# In[48]:


dfTest  = pd.read_csv('../input/test.csv')
dfTest = pd.merge(dfTest, index_0_risk, on = 'atom_index_0')
dfTest = pd.merge(dfTest, index_1_risk, on = 'atom_index_1')
dfTest = pd.merge(dfTest, type_risk, on = 'type')
dfTest['molecule_name_risk'] = 0


# In[49]:


dfTest.head()


# In[50]:


dfTest.drop(['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type'], axis = 1).head()


# In[51]:


df.head()


# In[52]:


lmReg = LinearRegression()
lmReg.fit(df.drop(['id', 'scalar_coupling_constant'], axis = 1), df['scalar_coupling_constant'])
dfTest['scalar_coupling_constant_pred'] = lmReg.predict(dfTest.drop(['id', 'molecule_name',                                                         'atom_index_0', 'atom_index_1', 'type'], axis = 1))


# In[53]:


dfSub['scalar_coupling_constant'] = dfTest['scalar_coupling_constant_pred']
dfSub.to_csv('sample_submission.csv', index = False)


# In[54]:


dfSub.head()


# In[ ]:




