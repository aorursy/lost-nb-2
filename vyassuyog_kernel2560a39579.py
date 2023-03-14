#!/usr/bin/env python
# coding: utf-8

# In[72]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


my_cmp=pltc.LinearSegmentedColormap.from_list("",["red","green","blue"])


# In[4]:


train_data=pd.read_csv("../input/2el1730-machinelearning/train.csv")
train_data.fillna(0)
train_data


# In[5]:


train_desc=train_data.describe()
train_desc.columns


# In[6]:


for i in train_desc.columns:
    plt.scatter(train_data[i],train_data["label"])
    plt.xlabel(i)
    plt.show()


# In[7]:


train_data.loc[:,"website"]=train_data.loc[:,"org"]  + train_data.loc[:,"tld"]
train_data


# In[8]:


org_freq=train_data.groupby("website").size()/len(train_data)
train_data.loc[:,"website_freq"]=train_data.loc[:,"website"].map(org_freq)
train_data


# In[9]:


mail_freq=train_data.groupby("mail_type").size()/len(train_data)
train_data.loc[:,"mail_type_freq"]=train_data.loc[:,"mail_type"].map(mail_freq)
train_data


# In[10]:


train_data.drop(["Id","date","org","tld","mail_type"],axis=1,inplace=True)
train_data
train_data.drop(["website"],axis=1,inplace=True)

train_data.fillna(0,inplace=True)


# In[11]:


X=train_data.drop(["label"],axis=1)
X


# In[12]:


Y=train_data["label"]
Y


# In[13]:


X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=0)
print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)


# In[14]:


std=StandardScaler()
X_train_std=std.fit_transform(X_train)
X_val_std=std.transform(X_val)


# In[15]:


X_train_std=pd.DataFrame(X_train_std,columns=X_train.columns)
X_train_std.fillna(0,inplace=True)
X_train_std


# In[16]:


X_val_std=pd.DataFrame(X_val_std,columns=X_val.columns)
X_val_std.fillna(0,inplace=True)
X_val_std


# In[17]:


for i in range(10):
    knn=KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_train_std,Y_train)
    Y_pred=knn.predict(X_val_std)
    acc=accuracy_score(Y_val,Y_pred)
    print(acc)


# In[18]:


for i in range(10):
    knn=KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_train_std,Y_train)
    Y_pred=knn.predict(X_train_std)
    acc=accuracy_score(Y_train,Y_pred)
    print(acc)


# In[19]:


X_train.fillna(0,inplace=True)
X_val.fillna(0,inplace=True)


# In[20]:


log=LogisticRegression()

log.fit(X_train_std,Y_train)
Y_pred=log.predict(X_val_std)
acc=accuracy_score(Y_val,Y_pred)
print(acc)


# In[21]:


log=LogisticRegression()

log.fit(X_train,Y_train)
Y_pred=log.predict(X_val)
acc=accuracy_score(Y_val,Y_pred)
print(acc)


# In[22]:


nb=GaussianNB()
nb.fit(X_train_std,Y_train)
Y_pred=nb.predict(X_val_std)
acc=accuracy_score(Y_val,Y_pred)
print(acc)


# In[23]:


trainl_data=pd.read_csv("../input/2el1730-machinelearning/train.csv")


# In[24]:


trainl_data


# In[25]:


trainl_data.loc[:,"website"]=trainl_data.loc[:,"org"]  + trainl_data.loc[:,"tld"]
trainl_data


# In[26]:


label=preprocessing.LabelEncoder()
trainl_data["website"]=label.fit_transform(list(trainl_data["website"]))
trainl_data["website"]


# In[27]:


label=preprocessing.LabelEncoder()
trainl_data["mail_type"]=label.fit_transform(list(trainl_data["mail_type"]))
trainl_data["mail_type"]


# In[28]:


trainl_data.drop(["Id","date","org","tld"],axis=1,inplace=True)
trainl_data


# In[29]:


X_l=trainl_data.drop(["label"],axis=1)
X_l


# In[30]:


Y_l=trainl_data["label"]
Y_l


# In[31]:


X_trainl,X_vall,Y_trainl,Y_vall=train_test_split(X_l,Y_l,test_size=0.2,stratify=Y,random_state=0)
print(X_trainl.shape,X_vall.shape,Y_trainl.shape,Y_vall.shape)


# In[32]:


std=StandardScaler()
X_trainl_std=std.fit_transform(X_trainl)
X_vall_std=std.transform(X_vall)


# In[33]:


X_trainl_std=pd.DataFrame(X_trainl_std,columns=X_trainl.columns)
X_trainl_std.fillna(0,inplace=True)
X_trainl_std


# In[34]:


X_vall_std=pd.DataFrame(X_vall_std,columns=X_vall.columns)
X_vall_std.fillna(0,inplace=True)
X_vall_std


# In[78]:


for i in range(10):
    knn=KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_trainl_std,Y_trainl)
    Y_pred=knn.predict(X_vall_std)
    acc=accuracy_score(Y_vall,Y_pred)
    print(acc)


# In[79]:


for i in range(10):
    knn=KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_trainl_std,Y_trainl)
    Y_pred=knn.predict(X_trainl_std)
    acc=accuracy_score(Y_trainl,Y_pred)
    print(acc)


# In[46]:


X_train.(kind="box",subplots=True,layout=(3,4))


# In[50]:


corr=X_train.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(corr,vmax=1,vmin=-1)
fig.colorbar(cax)
ticks=np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticklabels(X_train.columns)
ax.set_yticklabels(X_train.columns)
plt.show()


# In[ ]:


test_data=pd.read_csv("../input/2el1730-machinelearning/test.csv")
test_data.fillna(0)
test_data


# In[55]:


test_data.loc[:,"website"]=test_data.loc[:,"org"]  + test_data.loc[:,"tld"]
test_data


# In[57]:


label=preprocessing.LabelEncoder()
test_data["website"]=label.fit_transform(list(test_data["website"]))
test_data["website"]


# In[58]:


label=preprocessing.LabelEncoder()
test_data["mail_type"]=label.fit_transform(list(test_data["mail_type"]))
test_data["mail_type"]


# In[60]:


test_data.drop(["Id","date","org","tld"],axis=1,inplace=True)
test_data


# In[61]:


std=StandardScaler()
test_data_std=std.fit_transform(test_data)
test_data_std


# In[88]:


knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_trainl_std,Y_trainl)
Y_pred=knn.predict(test_data_std)
Y_pred


# In[89]:


Y_pred=pd.DataFrame(Y_pred)


# In[98]:


Submit=pd.concat((test_data,Y_pred),axis=1)
Submit
Submit.rename(columns={0: 'Y_pred'},inplace=True)


# In[99]:


Submit


# In[ ]:




