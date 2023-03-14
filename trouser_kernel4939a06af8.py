#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
sample_submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")
test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")
test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")
train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")
train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")


# In[2]:


train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


# In[3]:


print(f"Training data contains {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"Test data contains {test.shape[0]} rows and {test.shape[1]} columns.")


# In[4]:


train.head()


# In[5]:


del test_identity, test_transaction, train_identity, train_transaction


# In[6]:


print(f"There are {train.isnull().any().sum()} columns in training data with null values.")
print(f"There are {test.isnull().any().sum()} columns in test data with null values.")


# In[7]:


[c for c in train.columns if train[c].nunique() <= 1]


# In[8]:


[c for c in test.columns if test[c].nunique() <= 1]


# In[9]:


import numpy as np
import matplotlib.pyplot as plt

color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


# In[10]:


train['TransactionDT'].plot(kind='hist',
                                        figsize=(15, 5),
                                        label='train',
                                        bins=50,
                                        title='Train vs Test TransactionDT distribution')
test['TransactionDT'].plot(kind='hist',
                                       label='test',
                                       bins=50)
plt.legend()
plt.show()


# In[11]:


train['TransactionAmt'].plot(
    kind='hist',
    bins=1000,
    figsize=(15, 5),
    xlim=(0,10000),
    label='train',
    title="Distribution of TransactionAmt in linear scale"
)
test['TransactionAmt'].plot(
    kind='hist',
    bins=1000,
    figsize=(15, 5),
    xlim=(0,10000),
    label='test',
)
plt.legend()
plt.show()


# In[12]:


train['TransactionAmt'].apply(np.log)     .plot(
        kind='hist',
        bins=100,
        figsize=(15, 5),
        title="Distribution of TransactionAmt in log scale",
        label='train',
    
)
test['TransactionAmt'].apply(np.log)     .plot(
        kind='hist',
        bins=100,
        figsize=(15, 5),
        label='test',
)
plt.legend()
plt.show()


# In[13]:


train.groupby("ProductCD")["TransactionID"].count()     .plot(kind='barh', title="Number of transactions by ProductCD")


# In[14]:


train.groupby("ProductCD")["isFraud"].mean()     .plot(kind='barh', title="Percentage of fraud by ProductCD")

plt.show()


# In[15]:


card_features = [c for c in train.columns if 'card' in c]
card_cols = train[card_features]


# In[16]:


card_cols.head()


# In[17]:


card_cols.nunique()


# In[18]:


color_idx = 0
for c in card_cols:
    if train[c].dtype in ['float64','int64']:
        train[c].plot(kind='hist',
              title=c,
              bins=50,
              figsize=(15, 2),
              color=color_pal[color_idx])
    color_idx += 1
    plt.show()


# In[19]:


addr_features = [c for c in train.columns if 'addr' in c]
addr_cols = train[addr_features]


# In[20]:


addr_cols.head()


# In[21]:


addr_cols.nunique()


# In[ ]:




