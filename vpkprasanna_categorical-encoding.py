#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,f1_score
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')


# In[3]:


train.describe()


# In[4]:


train.head()


# In[5]:


list(train['ord_0'].unique())


# In[6]:


np.ceil(train['ord_0'].mean())


# In[7]:


train.info()


# In[8]:


def summary(df):
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values
    return summary


summary(train)


# In[9]:


set(train['bin_0'].isna())


# In[10]:


corr = train.corr()
plt.figure(figsize=(10,10))
ax = sns.heatmap(corr,vmin=-1, vmax=1, center=0,cmap='coolwarm',square=True)


# In[11]:


def missing_value(df):
    final = []
    empty = {}
    list_of_columns = list(df.columns)
    for column in range(0,len(list_of_columns)):
        data = set(df[list_of_columns[column]].isna())
        empty.update({
            list_of_columns[column]:data
        })
    final.append(empty)
    return final


# In[12]:


missing_value(train)


# In[13]:


train.isna().sum()


# In[14]:


train['bin_0'].mode()[0]


# In[15]:


train['bin_0'].fillna(train['bin_0'].mode()[0], inplace = True)
test['bin_0'].fillna(test['bin_0'].mode()[0], inplace = True)

train['bin_1'].fillna(train['bin_1'].mode()[0], inplace = True)
test['bin_1'].fillna(test['bin_1'].mode()[0], inplace = True)

train['bin_2'].fillna(train['bin_2'].mode()[0],inplace = True)
test['bin_2'].fillna(test['bin_2'].mode()[0],inplace = True)

train['bin_3'].fillna(train['bin_3'].mode()[0], inplace = True)
test['bin_3'].fillna(test['bin_3'].mode()[0], inplace = True)

train['bin_4'].fillna(train['bin_4'].mode()[0], inplace = True)
test['bin_4'].fillna(test['bin_4'].mode()[0], inplace = True)

train['nom_0'].fillna(train['nom_0'].mode()[0],inplace = True)
test['nom_0'].fillna(test['nom_0'].mode()[0],inplace = True)

train['nom_1'].fillna(train['nom_1'].mode()[0],inplace = True)
test['nom_1'].fillna(test['nom_1'].mode()[0],inplace = True)

train['nom_2'].fillna(train['nom_2'].mode()[0],inplace = True)
test['nom_2'].fillna(test['nom_2'].mode()[0],inplace = True)

train['nom_3'].fillna(train['nom_3'].mode()[0],inplace = True)
test['nom_3'].fillna(test['nom_3'].mode()[0],inplace = True)

train['nom_4'].fillna(train['nom_4'].mode()[0],inplace = True)
test['nom_4'].fillna(test['nom_4'].mode()[0],inplace = True)

train['nom_5'].fillna(train['nom_5'].mode()[0],inplace = True)
test['nom_5'].fillna(test['nom_5'].mode()[0],inplace = True)

train['nom_6'].fillna(train['nom_6'].mode()[0],inplace = True)
test['nom_6'].fillna(test['nom_6'].mode()[0],inplace = True)

train['nom_7'].fillna(train['nom_7'].mode()[0],inplace = True)
test['nom_7'].fillna(test['nom_7'].mode()[0],inplace = True)

train['nom_8'].fillna(train['nom_8'].mode()[0],inplace = True)
test['nom_8'].fillna(test['nom_8'].mode()[0],inplace = True)

train['nom_9'].fillna(train['nom_9'].mode()[0],inplace = True)
test['nom_9'].fillna(test['nom_9'].mode()[0],inplace = True)

train['ord_0'].fillna(np.ceil(train['ord_0'].mean()),inplace = True)
test['ord_0'].fillna(np.ceil(train['ord_0'].mean()),inplace = True)

train['ord_1'].fillna(train['ord_1'].mode()[0],inplace = True)
test['ord_1'].fillna(test['ord_1'].mode()[0],inplace = True)

train['ord_2'].fillna(train['ord_2'].mode()[0],inplace = True)
test['ord_2'].fillna(test['ord_2'].mode()[0],inplace = True)

train['ord_3'].fillna(train['ord_3'].mode()[0],inplace = True)
test['ord_3'].fillna(test['ord_3'].mode()[0],inplace = True)

train['ord_4'].fillna(train['ord_4'].mode()[0],inplace = True)
test['ord_4'].fillna(test['ord_4'].mode()[0],inplace = True)

train['ord_5'].fillna(train['ord_5'].mode()[0],inplace = True)
test['ord_5'].fillna(test['ord_5'].mode()[0],inplace = True)

train['day'].fillna(train['day'].mode()[0],inplace = True)
test['day'].fillna(test['day'].mode()[0],inplace = True)

train['month'].fillna(train['month'].mode()[0],inplace = True)
test['month'].fillna(test['month'].mode()[0],inplace = True)


# In[16]:


train['bin_3'] = train['bin_3'].apply(lambda x: 0 if x == 'F' else 1)
test['bin_3'] = test['bin_3'].apply(lambda x: 0 if x == 'F' else 1)

# bin_4
train['bin_4'] = train['bin_4'].apply(lambda x: 0 if x == 'N' else 1)
test['bin_4'] = test['bin_4'].apply(lambda x: 0 if x == 'N' else 1)

# ord_1
train.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)
test.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)

# ord_2
train.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)
test.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

# ord_3
train.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)
test.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)

# ord_4
train.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 
                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                                  22, 23, 24, 25], inplace = True)
test.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 
                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                                  22, 23, 24, 25], inplace = True)

high_card = ['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9','ord_5']
for col in high_card:
    enc_nom = (train.groupby(col).size()) / len(train)
    train[f'{col}'] = train[col].apply(lambda x: hash(str(x)) % 5000)
    test[f'{col}'] = test[col].apply(lambda x: hash(str(x)) % 5000)


# In[17]:


train.head()


# In[18]:


test.head()


# In[19]:


y = train['target']
x = train.drop(['target','id','ord_3','ord_4'],axis = 1)
test = test.drop(['id','ord_3','ord_4'],axis = 1)


# In[20]:


len(x)


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[23]:


clf = LogisticRegression(C=0.20, solver="lbfgs", tol=0.020, max_iter=2020)
clf.fit(X_train,y_train)


# In[24]:


pred = clf.predict_proba(X_test)
pred


# In[25]:


y_pred=clf.predict(X_test)


# In[26]:


value = accuracy_score(y_test,y_pred)
print(value)


# In[27]:


plt.figure(figsize=(8,8))
df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')


# In[28]:


pres = precision_score(y_test, y_pred)
print(pres)


# In[29]:


f1 = f1_score(y_test, y_pred,average='micro')
print(f1*100)


# In[30]:


submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')


# In[31]:


test_id = test.pop('id')


# In[32]:



submission['id'] = test_id
submission['target'] = clf.predict_proba(test)


# In[33]:


submission.head()


# In[34]:


submission.to_csv("submission.csv", index=False)


# In[35]:


calibrate = CalibratedClassifierCV(clf,method='sigmoid',cv = 5)
calibrate.fit(X_train,y_train)
pred = calibrate.predict(test)
print(pred)


# In[36]:


submi = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')


# In[37]:


submi['id'] = test_id
submi['target'] = calibrate.predict_proba(test)


# In[38]:


submi.to_csv("submission_new.csv", index=False)


# In[39]:


train.boxplot(column=['nom_0','nom_1','nom_2', 'nom_3' ,'nom_4'])


# In[40]:


X_train.head()


# In[41]:


print(len(list(test.columns)))
print(len(list(X_train.columns)))


# In[42]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)


# In[43]:


fianl = xgb.predict(test)


# In[44]:


xgb_submi = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
xgb_submi['id'] = test_id
xgb_submi['target'] = xgb.predict_proba(test)
xgb_submi.to_csv("xgb_submission.csv", index=False)


# In[45]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
rfc = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
rfc.fit(X_train,y_train)


# In[46]:


rfc_submi = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
rfc_submi['id'] = test_id
rfc_submi['target'] = xgb.predict_proba(test)
rfc_submi.to_csv("rfc_submission.csv", index=False)


# In[ ]:




