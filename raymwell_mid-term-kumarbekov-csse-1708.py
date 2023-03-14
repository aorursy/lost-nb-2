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


train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[3]:


train.info()


# In[4]:


test.info()


# In[5]:


train_32 = train.drop(['ID_code', 'target'], axis = 1).astype('float32')


# In[6]:


train_32.info()


# In[7]:


test_32 = train.drop(['ID_code'], axis = 1).astype('float32')


# In[8]:


test_32.info()


# In[9]:


train.head(2)


# In[10]:


train_32.head(2)


# In[11]:


test.head(2)


# In[12]:


test_32.head(2)


# In[13]:


train_32.shape, test_32.shape


# In[14]:


train_32.describe()


# In[15]:


test_32.describe()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='target',data=train, palette='hls')
plt.show()


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[18]:


X = train.iloc[:, 2:].values
y = train.target.values
X_test = test.iloc[:, 1:].values


# In[19]:


X_train = X
y_train = y


# In[20]:


#create an instance and fit the model 
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[21]:


predictions = logmodel.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_train,predictions))


# In[23]:


sub_df = pd.DataFrame({'ID_code':test.ID_code.values})
sub_df['target'] = predictions
sub_df.to_csv('submission.csv', index=False)


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm


# In[25]:


x = np.linspace(-5, 5)
y = norm.pdf(x)
plt.plot(x, y)
plt.vlines(ymin=0, ymax=0.4, x=1, colors=['red'])


# In[26]:


target = train.target.values
train.drop('target', axis=1, inplace=True)
train.shape, target.shape, test.shape


# In[27]:


pos_idx = (target == 1)
neg_idx = (target == 0)
stats = []
for col in train_32.columns:
    stats.append([
        train_32.loc[pos_idx, col].mean(),
        train_32.loc[pos_idx, col].std(),
        train_32.loc[neg_idx, col].mean(),
        train_32.loc[neg_idx, col].std()
    ])
    
stats_df = pd.DataFrame(stats, columns=['pos_mean', 'pos_sd', 'neg_mean', 'neg_sd'])
stats_df.head()


# In[28]:


# priori probability
ppos = pos_idx.sum() / len(pos_idx)
pneg = neg_idx.sum() / len(neg_idx)

def get_proba(x):
    # we use odds P(target=1|X=x)/P(target=0|X=x)
    return (ppos * norm.pdf(x, loc=stats_df.pos_mean, scale=stats_df.pos_sd).prod()) /           (pneg * norm.pdf(x, loc=stats_df.neg_mean, scale=stats_df.neg_sd).prod())


# In[29]:


tr_pred = train_32.apply(get_proba, axis=1)


# In[30]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target, tr_pred)


# In[31]:


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(train.loc[pos_idx, 'var_0'])
plt.plot(np.linspace(0, 20), norm.pdf(np.linspace(0, 20), loc=stats_df.loc[0, 'pos_mean'], scale=stats_df.loc[0, 'pos_sd']))
plt.title('target==1')
plt.subplot(1, 2, 2)
sns.distplot(train.loc[neg_idx, 'var_0'])
plt.plot(np.linspace(0, 20), norm.pdf(np.linspace(0, 20), loc=stats_df.loc[0, 'neg_mean'], scale=stats_df.loc[0, 'neg_sd']))
plt.title('target==0')


# In[32]:


from scipy.stats.kde import gaussian_kde


# In[33]:


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(train.loc[pos_idx, 'var_0'])
kde = gaussian_kde(train.loc[pos_idx, 'var_0'].values)
plt.plot(np.linspace(0, 20), kde(np.linspace(0, 20)))
plt.title('target==1')
plt.subplot(1, 2, 2)
sns.distplot(train.loc[neg_idx, 'var_0'])
kde = gaussian_kde(train.loc[neg_idx, 'var_0'].values)
plt.plot(np.linspace(0, 20), kde(np.linspace(0, 20)))
plt.title('target==0')


# In[34]:


stats_df['pos_kde'] = None
stats_df['neg_kde'] = None
for i, col in enumerate(train_32.columns):
    stats_df.loc[i, 'pos_kde'] = gaussian_kde(train_32.loc[pos_idx, col].values)
    stats_df.loc[i, 'neg_kde'] = gaussian_kde(train_32.loc[neg_idx, col].values)


# In[35]:


def get_col_prob(df, coli, bin_num=100):
    bins = pd.cut(df.iloc[:, coli].values, bins=bin_num)
    uniq = bins.unique()
    uniq_mid = uniq.map(lambda x: (x.left + x.right) / 2)
    dense = pd.DataFrame({
        'pos': stats_df.loc[coli, 'pos_kde'](uniq_mid),
        'neg': stats_df.loc[coli, 'neg_kde'](uniq_mid)
    }, index=uniq)
    return bins.map(dense.pos).astype(float) / bins.map(dense.neg).astype(float)


# In[36]:


tr_pred = ppos / pneg
for i in range(200):
    tr_pred *= get_col_prob(train_32, i)


# In[37]:


roc_auc_score(target, tr_pred)


# In[38]:


te_pred = ppos / pneg
for i in range(200):
    te_pred *= get_col_prob(test_32, i)


# In[39]:


pd.DataFrame({
    'ID_code': test.index,
    'target': te_pred
}).to_csv('sub.csv', index=False)


# In[ ]:




