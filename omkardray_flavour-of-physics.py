#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


import numpy as np
import pandas as pd


# In[5]:


train = pd.read_csv('/kaggle/input/flavours-of-physics/training.csv.zip')
train.head()


# In[6]:


train.signal.nunique()


# In[7]:


train.signal.value_counts()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


sns.distplot(train['signal'])


# In[10]:


train.info()


# In[11]:


train = train.drop(columns=['min_ANNmuon'])


# In[12]:


train.columns


# In[13]:


test = pd.read_csv('/kaggle/input/flavours-of-physics/test.csv.zip')


# In[15]:


test.head()


# In[16]:


test.columns


# In[17]:


train= train.drop(['production','mass'],axis=1)


# In[18]:


train.head()


# In[19]:


y = train['signal']


# In[20]:


type(y)


# In[21]:


train.skew(axis = 0, skipna = True) 


# In[22]:


train['DOCAone'] = np.tanh(train.DOCAone)


# In[23]:


train.skew(axis = 0, skipna = True) 


# In[24]:


train.corr()


# In[25]:


sns.heatmap(train.corr(),vmin=0, vmax=1)


# In[26]:


for n in range(2):
  fig = plt.figure(figsize=(35,20))

  for i in range(1,26):
    ax = fig.add_subplot(5, 5, i)
    col = train.columns[i + 25*n]
    ax.set_title(col)

    plt.hist([train[train['signal'] == 1][col], train[train['signal'] == 0][col]], bins=50, histtype='stepfilled', color=['r', 'b'], alpha=0.4, label=['signal', 'background'])
    
    if (i == 5): ax.legend()
        
  fig.tight_layout(pad=1, w_pad=1, h_pad=1)
  fig.savefig('hist'+str(n+1)+'.png', dpi=150)


# In[27]:


train.columns


# In[29]:


X= train.drop(columns=['signal'])


# In[30]:


X.head()


# In[31]:


X= X.drop(columns=['id'])


# In[32]:


X.head()


# In[33]:


X.var()


# In[34]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled= pd.DataFrame(sc.fit_transform(X), columns=X.columns)
X_scaled.var()


# In[35]:


x=X_scaled.values


# In[36]:


type(x)


# In[37]:


X.head()


# In[38]:


X_scaled.head()


# In[39]:


X_scaled.shape


# In[40]:


np.shape(x)


# In[41]:


x


# In[42]:


y.shape


# In[43]:


y=y.values


# In[44]:


y=y.reshape(-1,1)


# In[45]:


np.shape(y)


# In[46]:


y


# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[48]:


X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.score(X_test,y_test)


# In[49]:


pip install tpot


# In[53]:


from tpot import TPOTClassifier

parameters = {'criterion': ['entropy', 'gini'],
               'max_depth': [2],
               'max_features': ['auto'],
               'min_samples_leaf': [4, 12, 16],
               'min_samples_split': [5, 10,15],
               'n_estimators': [10]}
               
tpot_classifier = TPOTClassifier(generations= 5, population_size= 32, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,mutation_rate=0.9,
                                 config_dict=
                                 {'sklearn.ensemble.RandomForestClassifier': parameters}, 
                                 cv = 4, scoring = 'accuracy')
tpot_classifier.fit(X_train,y_train) 


# In[54]:


accuracy = tpot_classifier.score(X_test, y_test)
print(accuracy)


# In[82]:


import xgboost as xgb
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,params=params)

xgb_model.fit(X_train,y_train)


# In[83]:


xgb_model.score(X_test,y_test)


# In[84]:


xgb_model.score(X_train,y_train)


# In[88]:


from sklearn.metrics import confusion_matrix
y_pred = xgb_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

