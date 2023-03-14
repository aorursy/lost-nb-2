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


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


data1 = pd.read_csv('../input/application_test.csv')
data1.head()
data2 = pd.read_csv('../input/application_train.csv')


# In[ ]:


data1 = data1.replace('NaN',np.nan)
data2 = data2.replace('NaN',np.nan)


# In[ ]:


s1 = data1.shape
s1


# In[ ]:


s2 = data2.shape
s2


# In[ ]:


g = []


# In[ ]:


for col in data2:
    if (data2[col].isnull().sum()/s2[0] > .6 ):
        print(col, data2[col].isnull().sum()/s2[0])
        g.append(col)


# In[ ]:


g1 = []


# In[ ]:


for col in data1:
    if (data1[col].isnull().sum()/s1[0] > .6 ):
        print(col, data1[col].isnull().sum()/s1[0])
        g1.append(col)


# In[ ]:


print(g)
print(g1)


# In[ ]:


test = data1.drop(g1, axis = 1)
print(test.columns)
train = data2.drop(g, axis = 1)
print(train.columns)


# In[ ]:


c = train.describe(include='number').columns
d = train.describe(include = 'object').columns


# In[ ]:


a = test.describe(include='number').columns
b = test.describe(include = 'object').columns


# In[ ]:


numdata_test = test[a]
objdata_test = test[b]


# In[ ]:


numdata_train = train[c]
objdata_train = train[d]


# In[ ]:


objdata_train.isnull().sum()


# In[ ]:


# numdata_train.isnull().sum()


# In[ ]:


pd.value_counts(objdata_train.columns).sum()


# In[ ]:


pd.value_counts(objdata_test.columns).sum()


# In[ ]:


objdata_test.isnull().sum()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'pd.get_dummies')


# In[ ]:


objdata_train.shape


# In[ ]:


# for col in objdata_train.iloc[0:15]:
#     i = objdata_train.columns.get_loc(col)
#     #print(i,col)
# #     print((pd.get_dummies(objdata_train[col],drop_first=True)).head())
# #     objdata_train = pd.concat([objdata_train.iloc[:,:i],pd.get_dummies(objdata_train[col],drop_first=True),objdata_train.iloc[:,(i+1):]],axis = 1)


# In[ ]:


for col in objdata_train:
    objdata_train[col] = objdata_train[col].astype('category',copy=False)
#     objdata_train[col] = objdata_train[col].cat.codes
#     objdata_train[col] = objdata_train[col].astype('category',copy=False)


# In[ ]:


objdata_train.dtypes


# In[ ]:


objdata_train.head()


# In[ ]:


for col in objdata_test:
    objdata_test[col] = objdata_test[col].astype('category')
#     objdata_test[col] = objdata_test[col].cat.codes
#     objdata_test[col] = objdata_test[col].astype('category')


# In[ ]:





# In[ ]:


objdata_test.describe()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'pd.DataFrame.astype')


# In[ ]:


pd.value_counts(objdata_train.columns).sum()


# In[ ]:


pd.value_counts(objdata_test.columns).sum()


# In[ ]:


# for col in objdata1:
#      objdata1[col] = objdata1[col].astype('category',copy=False)
#      objdata1[col] = objdata1[col].cat.codes


# In[ ]:


# for col in objdata:
#      objdata[col] = objdata[col].astype('category',copy=False)
#      objdata[col] = objdata[col].cat.codes


# In[ ]:


# numdata = data1[a]
# objdata = data1[b]
# print(pd.value_counts(b).sum() + pd.value_counts(a).sum())


# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


imp_test = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp2_test = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)


# In[ ]:


imp_train = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp2_train = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)


# In[ ]:


imp_test.fit(numdata_test)


# In[ ]:


imp2_test.fit(objdata_test)


# In[ ]:


imp_train.fit(numdata_train)


# In[ ]:


imp2_train.fit(objdata_train)


# In[ ]:


impnumdata_test = pd.DataFrame(imp_test.transform(numdata_test))
# impnumdata.isnull().sum()
# print(impnumdata.shape)
# print(objdata.shape)


# In[ ]:


impobjdata_test = pd.DataFrame(imp2_test.transform(objdata_test))


# In[ ]:


impnumdata_train = pd.DataFrame(imp_train.transform(numdata_train))


# In[ ]:


impobjdata_train = pd.DataFrame(imp2_train.transform(objdata_train))


# In[ ]:


# impobjdata1 = pd.DataFrame(imp3.transform(objdata_train))


# In[ ]:


# impcatdata = pd.DataFrame(imp2.transform(objdata))
# impcatdata.isnull().sum()


# In[ ]:


impnumdata_test.columns = a
impobjdata_test.columns = b
impnumdata_train.columns = c
impobjdata_train.columns = d


# In[ ]:


for col in impobjdata_train:
    impobjdata_train[col] = impobjdata_train[col].astype('category')


# In[ ]:


impobjdata_train.dtypes


# In[ ]:


for col in impobjdata_test:
    impobjdata_test[col] = impobjdata_test[col].astype('category')


# In[ ]:


final_data_train = pd.concat([impnumdata_train,impobjdata_train],axis = 1)


# In[ ]:


# final_data_train.columns


# In[ ]:


final_data_test = pd.concat([impnumdata_test,impobjdata_test],axis = 1)


# In[ ]:


# final_data_test.dtypes


# In[ ]:


# final_data_test.isnull().sum()


# In[ ]:


from sklearn.utils import shuffle


# In[ ]:


train_data= final_data_train
train_data.columns
X_1 =train_data[ train_data["TARGET"]==1 ]
X_0=train_data[train_data["TARGET"]==0]
X_0=shuffle(X_0,random_state=42).reset_index(drop=True)
X_1=shuffle(X_1,random_state=42).reset_index(drop=True)

ALPHA=1.2

X_0=X_0.iloc[:round(len(X_1)*ALPHA),:]
final_data_train1=pd.concat([X_1, X_0])


# In[ ]:


d = pd.value_counts(final_data_train1['TARGET'])
d
c1 = d[0]/(d[0]+d[1] )
c2  = d[1]/(d[0]+d[1])
sizes = [c1,c2]
plot = plt.pie(sizes, labels = ['no','yes'],autopct='%1.1f%%',
        shadow=True, startangle=45 )
plt.axis('equal') 
plt.title("Balanced Data")
plt.show()


# In[ ]:


y = final_data_train1['TARGET']


# In[ ]:


x = final_data_train1.drop(['TARGET','SK_ID_CURR'],axis = 1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)


# In[ ]:


num = pd.value_counts(y)
num


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(x,y)


# In[ ]:


y_scores = model.predict(X_test)
model.score(X_test,y_test)


# In[ ]:


roc_auc_score(y_test, y_scores)


# In[ ]:


sample = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


pd.value_counts(sample.TARGET)


# In[ ]:


test = final_data_test.drop('SK_ID_CURR',axis = 1)


# In[ ]:


y_pred1 = model.predict(test)


# In[ ]:


sample['TARGET'] = y_pred1


# In[ ]:


sample.to_csv('model1.csv')


# In[ ]:


model2 = GradientBoostingClassifier()


# In[ ]:


model2.fit(x, y)


# In[ ]:


y_pred = model2.predict(test)


# In[ ]:


sample['TARGET'] = y_pred


# In[ ]:


sample.to_csv('model2.csv')


# In[ ]:





# In[ ]:





# In[ ]:




