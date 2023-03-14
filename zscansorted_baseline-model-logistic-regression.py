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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[3]:


df_train = pd.read_csv('../input/train.csv')
df_train.shape


# In[4]:


X = df_train[['{}'.format(col) for col in range(1, 250)]]
y = df_train['target']


# In[5]:


np.random.seed(10)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[6]:


mdl = LogisticRegression()
mdl.fit(X_train, y_train)


# In[7]:


from sklearn.metrics import accuracy_score, f1_score,                        recall_score, precision_score

def show_all_metrics(y_true, y_pred):
    print('accuracy: {}'.format(accuracy_score(y_true, y_pred)))
    print('recall: {}'.format(recall_score(y_true, y_pred)))
    print('precision: {}'.format(precision_score(y_true, y_pred)))
    print('f1: {}'.format(f1_score(y_true, y_pred)))
    
    


# In[8]:


show_all_metrics(y_train, mdl.predict(X_train))


# In[9]:


show_all_metrics(y_test, mdl.predict(X_test))


# In[10]:


df_test = pd.read_csv('../input/test.csv')


# In[11]:


df_test.shape


# In[12]:


X_submission = df_test[['{}'.format(col) for col in range(1, 250)]]
y_submission = mdl.predict(X_submission)


# In[13]:


df_submission = pd.DataFrame(dict(
    id=df_test.id,
    target=y_submission
))


# In[14]:


df_submission.head()


# In[15]:


import os

os.mkdir('../output')


# In[16]:


df_submission.to_csv('../output/submission.csv', index=False)


# In[17]:


from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# In[18]:


create_download_link(df_submission)


# In[19]:




