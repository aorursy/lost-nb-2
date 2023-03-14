#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[3]:


train_df.head()


# In[37]:


data = pd.concat([train_df,test_df])
data.head()
data.shape


# In[38]:


data = data.drop(data.select_dtypes(include=['O']).columns,axis = 1)
data.head()


# In[40]:


data1 = data.drop(['y'], axis=1)
print(data1.head())


# In[41]:


from sklearn.cluster import KMeans
kmeans= KMeans()
kmeans.fit(data1)


# In[ ]:


labels=kmeans.predict(data1)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(labels)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys)
plt.show()


# In[ ]:


labels=kmeans.predict(data1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


trail = list(data.drop(['y'],axis=1).columns)
len(trail)


# In[8]:


print(train_df.shape[0])


# In[11]:


train = (data[:train_df.shape[0]])
test = data[train_df.shape[0]:]


# In[ ]:





# In[ ]:


def cv_model(model):
    return cross_val_score(model,train,test,cv=)

