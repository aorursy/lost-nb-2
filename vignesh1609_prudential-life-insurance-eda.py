#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd 
import pandas_profiling as pdp
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.float_format = '{:.3f}'.format
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[3]:


train=pd.read_csv('../input/prudential-life-insurance-assessment/train.csv.zip')
test=pd.read_csv('../input/prudential-life-insurance-assessment/test.csv.zip')


# In[4]:


train.head()


# In[5]:


train.info()


# In[6]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return tt
    #return(np.transpose(tt))


# In[7]:


#checking missing value percentage in train data
missing_data(train)['Percent'].sort_values(ascending=False)


# In[8]:


#checking missing value percentage in train data
missing_data(test)['Percent'].sort_values(ascending=False)


# In[9]:


train=train[train.columns[train.isnull().mean() <= 0.75]]


# In[10]:


test=test[test.columns[test.isnull().mean() <= 0.75]]


# In[11]:


train.isnull().sum().sort_values(ascending=False)


# In[12]:


test.isnull().sum().sort_values(ascending=False)


# In[13]:


list_train=train.columns[train.isna().any()].tolist()


# In[14]:


list_test=test.columns[test.isna().any()].tolist()


# In[15]:


for i in range(0,len(list_train)):
    print('column name: ',list_train[i],' Dtype:',train[list_train[i]].dtypes)


# In[16]:


for i in range(0,len(list_test)):
    print('column name: ',list_test[i],' Dtype:',train[list_test[i]].dtypes)


# In[17]:


for column in list_train:
    train[column].fillna(train[column].mean(), inplace=True)


# In[18]:


for column in list_test:
    test[column].fillna(test[column].mean(), inplace=True)


# In[19]:


train.info()
test.info()


# In[20]:


obj_train=list(train.select_dtypes(include=['object']).columns)
obj_test=list(test.select_dtypes(include=['object']).columns)


# In[21]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train[obj_train]=le.fit_transform(train[obj_train])
test[obj_test]=le.transform(test[obj_test])


# In[22]:


f, axes = plt.subplots(1, 2, figsize=(15,7))
sns.boxplot(x = 'Wt', data=train,  orient='v' , ax=axes[0])
sns.distplot(train['Wt'],  ax=axes[1])


# In[23]:


f, axes = plt.subplots(1, 2, figsize=(15,7))
sns.boxplot(x = 'Ht', data=train,  orient='v' , ax=axes[0])
sns.distplot(train['Ht'],  ax=axes[1])


# In[24]:


f, axes = plt.subplots(1, 2, figsize=(15,7))
sns.boxplot(x = 'BMI', data=train,  orient='v' , ax=axes[0])
sns.distplot(train['BMI'],  ax=axes[1])


# In[25]:


f,axes=plt.subplots(1,2,figsize=(15,7))
sns.boxplot(x='Ins_Age',data=train,orient='v',ax=axes[0])
sns.distplot(train['Ins_Age'],ax=axes[1])


# In[26]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['Response'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Response')
ax[0].set_ylabel('')
sns.countplot('Response',data=train,ax=ax[1])
ax[1].set_title('Response')
plt.show()


# In[27]:


#create a funtion to create a  new target variable based on conditions 

def new_target(row):
    if (row['Response']<=7) & (row['Response']>=0):
        val=0
    elif (row['Response']==8):
        val=1
    else:
        val=-1
    return val


# In[28]:


#create a copy of original dataset
new_data=train.copy()


# In[29]:


#create a new column
new_data['Final_Response']=new_data.apply(new_target,axis=1)


# In[30]:


new_data['Final_Response'].value_counts()


# In[31]:


#distribution plot for target classes
sns.countplot(x=new_data.Final_Response).set_title('Distribution of rows by response categories')


# In[32]:


#dropping already existing column
new_data.drop(['Response'],axis=1,inplace=True)
train=new_data
del new_data


# In[33]:


train.rename(columns={'Final_Response':'Response'},inplace=True)


# In[34]:


# BMI Categorization
conditions = [
    (train['BMI'] <= train['BMI'].quantile(0.25)),
    (train['BMI'] > train['BMI'].quantile(0.25)) & (train['BMI'] <= train['BMI'].quantile(0.75)),
    (train['BMI'] > train['BMI'].quantile(0.75))]

choices = ['under_weight', 'average', 'overweight']

train['BMI_Wt'] = np.select(conditions, choices)

# Age Categorization
conditions = [
    (train['Ins_Age'] <= train['Ins_Age'].quantile(0.25)),
    (train['Ins_Age'] > train['Ins_Age'].quantile(0.25)) & (train['Ins_Age'] <= train['Ins_Age'].quantile(0.75)),
    (train['Ins_Age'] > train['Ins_Age'].quantile(0.75))]

choices = ['young', 'average', 'old']
train['Old_Young'] = np.select(conditions, choices)

# Height Categorization
conditions = [
    (train['Ht'] <= train['Ht'].quantile(0.25)),
    (train['Ht'] > train['Ht'].quantile(0.25)) & (train['Ht'] <= train['Ht'].quantile(0.75)),
    (train['Ht'] > train['Ht'].quantile(0.75))]

choices = ['short', 'average', 'tall']

train['Short_Tall'] = np.select(conditions, choices)

# Weight Categorization
conditions = [
    (train['Wt'] <= train['Wt'].quantile(0.25)),
    (train['Wt'] > train['Wt'].quantile(0.25)) & (train['Wt'] <= train['Wt'].quantile(0.75)),
    (train['Wt'] > train['Wt'].quantile(0.75))]

choices = ['thin', 'average', 'fat']

train['Thin_Fat'] = np.select(conditions, choices)


# In[35]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'BMI_Wt', hue = 'Response', data = train)


# In[36]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'Old_Young', hue = 'Response', data = train)


# In[37]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'Short_Tall', hue = 'Response', data = train)


# In[38]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'Thin_Fat', hue = 'Response', data = train)


# In[39]:


def new_target(row):
    if (row['BMI_Wt']=='overweight') or (row['Old_Young']=='old')  or (row['Thin_Fat']=='fat'):
        val='extremely_risky'
    else:
        val='not_extremely_risky'
    return val

train['extreme_risk'] = train.apply(new_target,axis=1)


# In[40]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'extreme_risk', hue = 'Response', data = train)


# In[41]:


def new_target(row):
    if (row['BMI_Wt']=='average') or (row['Old_Young']=='average')  or (row['Thin_Fat']=='average'):
        val='average'
    else:
        val='non_average'
    return val

train['average_risk'] = train.apply(new_target,axis=1)


# In[42]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'average_risk', hue = 'Response', data = train)


# In[43]:


def new_target(row):
    if (row['BMI_Wt']=='under_weight') or (row['Old_Young']=='young')  or (row['Thin_Fat']=='thin'):
        val='low_end'
    else:
        val='non_low_end'
    return val

train['low_end_risk'] = train.apply(new_target,axis=1)


# In[44]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'low_end_risk', hue = 'Response', data = train)


# In[45]:


plt.hist(train['Employment_Info_1']);
plt.title('Distribution of Employment_Info_1 variable');


# In[46]:


train['Product_Info_1'].value_counts()


# In[47]:


#product1 vs response
sns.distplot(train[train['Response']==0]['Product_Info_1'],hist=False,label='Rejected')
sns.distplot(train[train['Response']==1]['Product_Info_1'],hist=False,label='Accepted')


# In[48]:


#product2 vs response
sns.distplot(train[train['Response']==0]['Product_Info_2'],hist=False,label='Rejected')
sns.distplot(train[train['Response']==1]['Product_Info_2'],hist=False,label='Accepted')


# In[49]:


#product3 vs response
sns.distplot(train[train['Response']==0]['Product_Info_3'],hist=False,label='Rejected')
sns.distplot(train[train['Response']==1]['Product_Info_3'],hist=False,label='Accepted')


# In[50]:


#product5 vs response
sns.distplot(train[train['Response']==0]['Product_Info_5'],hist=False,label='Rejected')
sns.distplot(train[train['Response']==1]['Product_Info_5'],hist=False,label='Accepted')


# In[51]:


#product6 vs response
sns.distplot(train[train['Response']==0]['Product_Info_6'],hist=False,label='Rejected')
sns.distplot(train[train['Response']==1]['Product_Info_6'],hist=False,label='Accepted')


# In[52]:


#product7 vs response
sns.distplot(train[train['Response']==0]['Product_Info_7'],hist=False,label='Rejected')
sns.distplot(train[train['Response']==1]['Product_Info_7'],hist=False,label='Accepted')


# In[53]:


corr = train.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(100, 370, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

