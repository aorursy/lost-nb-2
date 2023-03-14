#!/usr/bin/env python
# coding: utf-8

# In[34]:


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


# In[35]:


pip install kaggle


# In[36]:


train = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv.zip')
test=pd.read_csv("../input/mercedes-benz-greener-manufacturing/test.csv.zip")


# In[37]:


print("Train shape :", train.shape)
print("Test shape :" ,test.shape)


# In[38]:


train.head()


# In[39]:


import matplotlib.pyplot as plt


# In[40]:


plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train.y.values))
plt.xlabel('value')
plt.ylabel('y')
plt.grid()
plt.show()


# In[41]:


train["y"].max()


# In[42]:


train["y"].min()


# In[43]:


train["y"].mean()


# In[44]:


print(train.columns)


# In[45]:


train.describe()


# In[46]:


import seaborn as sns


# In[47]:


train['X0'].unique()


# In[48]:


train.groupby('X0')['ID'].nunique()


# In[49]:


col_sort_order = np.sort(train['X0'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X0', y='y', data=train,order=col_sort_order)
plt.xlabel('X0', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X0")
plt.grid()
plt.show()


# In[50]:


col_sort_order = np.sort(train['X0'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X0', y='y', data=train, order=col_sort_order)
plt.xlabel('X0', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X0")
plt.show()


# In[51]:


train['X1'].unique()


# In[52]:


train.groupby('X1')['ID'].nunique()


# In[53]:


col_sort_order = np.sort(train['X1'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X1', y='y', data=train,order=col_sort_order)
plt.xlabel('X1', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X1")
plt.grid()
plt.show()


# In[54]:


col_sort_order = np.sort(train['X1'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X1', y='y', data=train, order=col_sort_order)
plt.xlabel('X1', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X1")
plt.show()


# In[55]:


train['X2'].unique()


# In[56]:


train.groupby('X2')['ID'].nunique()


# In[57]:


col_sort_order = np.sort(train['X2'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X2', y='y', data=train,order=col_sort_order)
plt.xlabel('X2', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X2")
plt.grid()
plt.show()


# In[58]:


col_sort_order = np.sort(train['X2'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X2', y='y', data=train, order=col_sort_order)
plt.xlabel('X2', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X2")
plt.show()


# In[59]:


train['X3'].unique()


# In[60]:


train.groupby('X3')['ID'].nunique()


# In[61]:


col_sort_order = np.sort(train['X3'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X3', y='y', data=train,order=col_sort_order)
plt.xlabel('X3', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X3")
plt.grid()
plt.show()


# In[62]:


col_sort_order = np.sort(train['X3'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X3', y='y', data=train, order=col_sort_order)
plt.xlabel('X3', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X3")
plt.show()


# In[63]:


train['X4'].unique()


# In[64]:


train.groupby('X4')['ID'].nunique()


# In[65]:


col_sort_order = np.sort(train['X4'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X4', y='y', data=train,order=col_sort_order)
plt.xlabel('X4', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X4")
plt.grid()
plt.show()


# In[66]:


col_sort_order = np.sort(train['X4'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X4', y='y', data=train, order=col_sort_order)
plt.xlabel('X4', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X4")
plt.show()


# In[67]:


train['X5'].unique()


# In[68]:


train.groupby('X5')['ID'].nunique()


# In[69]:


col_sort_order = np.sort(train['X5'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X5', y='y', data=train,order=col_sort_order)
plt.xlabel('X5', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X5")
plt.grid()
plt.show()


# In[70]:


col_sort_order = np.sort(train['X5'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X5', y='y', data=train, order=col_sort_order)
plt.xlabel('X5', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X5")
plt.show()


# In[71]:


train['X6'].unique()


# In[72]:


train.groupby('X6')['ID'].nunique()


# In[73]:


col_sort_order = np.sort(train['X6'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X6', y='y', data=train,order=col_sort_order)
plt.xlabel('X6', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X6")
plt.grid()
plt.show()


# In[74]:


col_sort_order = np.sort(train['X6'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X6', y='y', data=train, order=col_sort_order)
plt.xlabel('X6', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X6")
plt.show()


# In[75]:


train['X8'].unique()


# In[76]:


train.groupby('X8')['ID'].nunique()


# In[77]:


col_sort_order = np.sort(train['X8'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.stripplot(x='X8', y='y', data=train,order=col_sort_order)
plt.xlabel('X8', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title("Distb. of y variable with X8")
plt.grid()
plt.show()


# In[78]:


col_sort_order = np.sort(train['X8'].unique()).tolist()
plt.figure(figsize=(14,6))
sns.boxplot(x='X8', y='y', data=train, order=col_sort_order)
plt.xlabel('X8', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.title("Distb. of y variable with X8")
plt.show()


# In[79]:


get_ipython().system(' pip install pycaret')


# In[80]:


from pycaret.regression import *


# In[81]:


train.info()


# In[82]:


reg = setup(data = train, 
             target = 'y',
             categorical_features = ['X0','X1','X2','X3','X4','X5','X6','X8'],
              normalize = True,
             numeric_imputation = 'mean',
             pca=True,
             silent = True)


# In[83]:


compare_models()


# In[84]:


br=create_model('br')


# In[85]:


br


# In[86]:


tuned_dt = tune_model(br)


# In[87]:


plot_model(br)


# In[88]:


tuned_br = tune_model(br, optimize = 'R2')


# In[89]:


tuned_br


# In[90]:


final_br= finalize_model(tuned_br)


# In[91]:


save_model(final_br, 'br_saved1126082020')


# In[ ]:





# In[92]:


#sample= pd.read_csv('../input/sample/sample.csv')


# In[93]:


#predictions = predict_model(tuned_br, data = test)
#sample['y'] = predictions['Label']


# In[94]:


#sample.head()


# In[ ]:




