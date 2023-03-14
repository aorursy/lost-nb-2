#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
train_df = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
train_df.head()


# In[2]:


prop_df = pd.read_csv("../input/properties_2016.csv")
prop_df.head()


# In[3]:


train_df['transaction_month'] = train_df['transactiondate'].dt.month
train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
train_df.head()


# In[4]:


LABEL_COLUMN = 'logerror'
COLUMNS = train_df.columns
CATEGORICAL_COLUMNS = ['transaction_month',
       'airconditioningtypeid', 'architecturalstyletypeid', 
       'buildingclasstypeid',
       'buildingqualitytypeid', 'decktypeid',
       'hashottuborspa', 'poolcnt',
       'heatingorsystemtypeid', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
       'propertycountylandusecode', 'propertylandusetypeid',
       'propertyzoningdesc', 'regionidcity',
       'regionidcounty', 'regionidneighborhood', 'regionidzip', 
       'fireplaceflag', 'taxdelinquencyflag',
        'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
       'fireplacecnt', 'fullbathcnt','garagecarcnt', 
       'roomcnt','storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid',
       'unitcnt', 'yearbuilt',
       'numberofstories', 'assessmentyear']
#CONTINUOUS_COLUMNS = ['{}'.format(x) for x in COLUMNS if x not in CATEGORICAL_COLUMNS]
CONTINUOUS_COLUMNS = COLUMNS.drop(CATEGORICAL_COLUMNS).drop(LABEL_COLUMN)


# In[5]:


COLUMNS


# In[6]:


CONTINUOUS_COLUMNS


# In[7]:


df_train=train_df.sample(frac=0.8,random_state=200)
df_test=train_df.drop(df_train.index)


# In[8]:


df_train


# In[9]:


df_test.shape


# In[10]:


continuous_cols = {k: tf.constant(df_train[k].values)
                     for k in CONTINUOUS_COLUMNS}


# In[11]:


continuous_cols


# In[12]:





# In[12]:





# In[12]:





# In[12]:


def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  continuous_cols.head()
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  print(feature_cols, label)
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)


def eval_input_fn():
  return input_fn(df_test)


# In[13]:


import tensorflow as tf
# Creates a dictionary mapping from each continuous feature column name (k) to
# the values of that column stored in a constant Tensor.
continuous_cols = {k: tf.constant(df_train[k].values) for k in CONTINUOUS_COLUMNS}
continuous_cols.head()


# In[14]:


tf.constant(df_train['logerror'].values)


# In[15]:


# Creates a dictionary mapping from each categorical feature column name (k)
# to the values of that column stored in a tf.SparseTensor.
categorical_cols = {k: tf.SparseTensor(
    indices=[[i, 0] for i in range(df[k].size)],
    values=df[k].values,
    dense_shape=[df[k].size, 1])
                    for k in CATEGORICAL_COLUMNS}
# Merges the two dictionaries into one.
feature_cols = dict(continuous_cols.items() + categorical_cols.items())
# Converts the label column into a constant Tensor.
label = tf.constant(df[LABEL_COLUMN].values)
# Returns the feature columns and the label.
print(feature_cols, label)

