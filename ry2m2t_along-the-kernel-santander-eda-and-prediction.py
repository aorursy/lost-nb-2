#!/usr/bin/env python
# coding: utf-8

# In[1]:


# check python runtime version
import platform
platform.python_version()


# In[2]:


import gc
# garbage collector interface
# https://docs.python.org/3/library/gc.html

import logging
# logger
# https://docs.python.org/3/library/logging.html

import datetime
# datetime utils
# https://docs.python.org/3/library/datetime.html

import warnings
# warning control
# https://docs.python.org/3/library/warnings.html
warnings.filterwarnings('ignore')


# In[3]:


# EDA tools
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 6] # set default figure size


# In[4]:


# # ML tools
# import lightgbm as lgb
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.model_selection import StratifiedKFold


# In[5]:


# other utils
from tqdm import tqdm_notebook


# In[6]:


import os
# print(os.listdir("./data"))


# In[7]:


IS_LOCAL = False
if(IS_LOCAL):
    PATH = '../input/Santander/'
else:
    PATH = '../input/santander-customer-transaction-prediction/'
os.listdir(PATH)


# In[8]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv(PATH + 'train.csv')\ntest_df = pd.read_csv(PATH + 'test.csv')")


# In[9]:


train_df.shape, test_df.shape


# In[10]:


train_df.head()


# In[11]:


test_df.head()


# In[12]:


def missing_data(data):
    # no. of missing values for each column
    total = data.isnull().sum() 
    
    # no. of values (incl. missing) for each column
    count = data.isnull().count()
    
    # calculate ratio of missing values for each column
    percent = (total / count * 100)
    
    # concatenate horizontally
    tt = pd.concat([total, percent], axis=1,
                   keys=['Total', 'Percent'])
    
    # get data type for each column
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    
    return(np.transpose(tt))


# In[13]:


get_ipython().run_cell_magic('time', '', '# get number and rate of missing values, \n# with data types for each column of train data\nmissing_data(train_df)')


# In[14]:


get_ipython().run_cell_magic('time', '', '# check the same for test data\nmissing_data(test_df)')


# In[15]:


get_ipython().run_cell_magic('time', '', '# numerical statistics for each column\ntrain_df.describe()')


# In[16]:


get_ipython().run_cell_magic('time', '', '# same for test data\ntest_df.describe()')


# In[17]:


def plot_feature_scatter(df1, df2, features):
    i = 0
    sns.set_style()
    
    # prepare the whole drawing area, with 4 rows and 4 columns
    # plt.figure() <- not necessary
    fig, ax = plt.subplots(4, 4, figsize=(14, 14))
    
    # draw each subplots
    for feature in features:
        i += 1
        plt.subplot(4, 4, i)
        plt.scatter(df1[feature], df2[feature], marker='+')
        plt.xlabel(feature, fontsize=9)
    
    plt.show()


# In[18]:


# prepare list of feature names
feature_ids = range(16)
features = ['var_' + str(i) for i in feature_ids]
features


# In[19]:


get_ipython().run_cell_magic('time', '', '# use data reduced to only 5% of all\nplot_feature_scatter(train_df[::20], test_df[::20], features)')


# In[20]:


sns.countplot(train_df['target'])


# In[21]:


# target value composition
train_df['target'].value_counts() / train_df.shape[0]


# In[22]:


def plot_feature_distribution(df1, df2, label1, label2):
    i = 0
    nrows = 20
    ncols = 10
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 60))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    features = ['var_' + str(i) for i in range(200)]
    for feature in features:
        i += 1
        plt.subplot(nrows, ncols, i)
        
        # draw kde plots; bw is scalar factor for granularity of kde calculation
        sns.kdeplot(df1[feature], bw=0.5, label=label1)
        sns.kdeplot(df2[feature], bw=0.5, label=label2)
        
        plt.xlabel(feature, fontsize=9)
#         locs, labels = plt.xticks()
        
#     plt.show()
        


# In[23]:


t0 = train_df[train_df['target'] == 0]
t1 = train_df[train_df['target'] == 1]


# In[24]:


# %%time
# plot_feature_distribution(t0, t1, '0', '1')


# In[25]:


# from PIL import Image
# Image.open('../input/santander/download_1_.png')


# In[26]:


# use only 'var_XX' columns
features = train_df.columns.values[2:202] # same for train and test data


# In[27]:


# draw dist of mean value per row
plt.figure(figsize=(16, 6))
plt.title('Distribution of mean value per row, train / test')
sns.distplot(train_df[features].mean(axis=1), color='green', kde=True, bins=120, label='train')
sns.distplot(train_df[features].mean(axis=1), color='blue',  kde=True, bins=120, label='test' )
plt.legend()


# In[28]:


plt.figure()
plt.title('Distribution of mean value per row, by target')
sns.distplot(t0[features].mean(axis=1), color='orange', kde=True, bins=120, label='0')
sns.distplot(t1[features].mean(axis=1), color='red', kde=True, bins=120,  label='1')
plt.legend()


# In[29]:


# draw dist of std value per row
plt.figure()
plt.title('Distribution of std value per row, by train / test')
sns.distplot(train_df[features].std(axis=1), color='green', kde=True, bins=120, label='train')
sns.distplot(test_df[features].std(axis=1), color='blue',  kde=True, bins=120, label='test')
plt.legend()


# In[30]:


plt.figure()
plt.title('Distribution for std value per row, by target')
sns.distplot(t0[features].std(axis=1), color='orange', kde=True, bins=120, label='0')
sns.distplot(t1[features].std(axis=1), color='red', kde=True, bins=120, label='1')
plt.legend()


# In[31]:


# dist for min per row
plt.figure()
plt.title('Distribution for min value per row, by train / test')
sns.distplot(train_df[features].min(axis=1), color='green', kde=True, bins=120, label='train')
sns.distplot(test_df[features].min(axis=1), color='blue', kde=True, bins=120, label='test')
plt.legend()


# In[32]:


plt.figure()
plt.title('Distribution of min value per row, by target')
sns.distplot(t0[features].min(axis=1), color='orange', kde=True, bins=120, label='0')
sns.distplot(t1[features].min(axis=1), color='red', kde=True, bins=120, label='1')
plt.legend()


# In[33]:


# dist for max per row
plt.figure()
plt.title('Distribution for max value per row, by train / test')
sns.distplot(train_df[features].max(axis=1), color='green', kde=True, bins=120, label='train')
sns.distplot(test_df[features].max(axis=1), color='blue', kde=True, bins=120, label='test')
plt.legend()


# In[34]:


plt.figure()
plt.title('Distribution of max value per row, by target')
sns.distplot(t0[features].max(axis=1), color='orange', kde=True, bins=120, label='0')
sns.distplot(t1[features].max(axis=1), color='red', kde=True, bins=120, label='1')
plt.legend()


# In[35]:


# projection along index
plt.figure()
plt.title('Projection of mean value per row (along index)')
plt.xlabel('row index')
sns.scatterplot(data=train_df[features].mean(axis=1), marker='+', color='green', label='train')
plt.legend()


# In[36]:


# projections of min along index
plt.figure()
plt.title('Projection of min value per row (along index)')
plt.xlabel('row index')
sns.scatterplot(data=train_df[features].min(axis=1), marker='+', color='green', label='train')
plt.legend()


# In[37]:


# projections of max along index
plt.figure()
plt.title('Projection of max value per row (along index)')
plt.xlabel('row index')
sns.scatterplot(data=train_df[features].max(axis=1), marker='+', color='green', label='train')
plt.legend()


# In[38]:


proj = pd.DataFrame({
    'min': train_df[features].min(axis=1),
    'mean': train_df[features].mean(axis=1),
    'max': train_df[features].max(axis=1),
})
proj_sorted = proj.sort_values(by='mean')


# In[39]:


# projection of mean along index, sorted
plt.figure()
plt.title('Projection of mean value along index, sorted')
sns.scatterplot(data=proj_sorted['min'].values, marker='+', label='min')
sns.scatterplot(data=proj_sorted['mean'].values, marker='+', label='mean')
sns.scatterplot(data=proj_sorted['max'].values, marker='+', label='max')
plt.legend()


# In[40]:


plt.figure(figsize=(16, 16))
plt.title('Scatter plot for (mean, std) along index')
plt.xlabel('mean per row')
plt.ylabel('std per row')
sns.scatterplot(x=train_df[features].mean(axis=1), y=train_df[features].std(axis=1), marker='+')


# In[41]:


# draw dist of mean value per column
plt.figure()
plt.title('Distribution of mean value per column, by train / test')
sns.distplot(train_df[features].mean(axis=0), color='green', kde=True, bins=120, label='train')
sns.distplot(test_df[features].mean(axis=0),  color='blue',  kde=True, bins=120, label='test')
plt.legend()


# In[42]:


plt.figure()
plt.title('Distribution of mean value per column, by target')
sns.distplot(t0[features].mean(axis=0), color='orange', kde=True, bins=120, label='0')
sns.distplot(t1[features].mean(axis=0), color='red', kde=True, bins=120, label='1')
plt.legend()


# In[43]:


# draw dist of std value per column
plt.figure()
plt.title('Distribution of std value per column')
sns.distplot(train_df[features].std(axis=0), color='green', kde=True, bins=120, label='train')
sns.distplot(test_df[features].std(axis=0), color='blue', kde=True, bins=120, label='test')
plt.legend()


# In[44]:


plt.figure()
plt.title('Distribution of std value per column, by target')
sns.distplot(t0[features].std(axis=0), color='orange', kde=True, bins=120, label='0')
sns.distplot(t1[features].std(axis=0), color='red', kde=True, bins=120, label='1')
plt.legend()


# In[45]:


# dist of min per column
plt.figure()
plt.title('Distribution of min value per column')
sns.distplot(train_df[features].min(axis=0), color='green', kde=True, bins=120, label='train')
sns.distplot(test_df[features].min(axis=0), color='blue', kde=True, bins=120, label='test')
plt.legend()


# In[46]:


plt.figure()
plt.title('Distribution of min value per column, by target')
sns.distplot(t0[features].min(axis=0), color='orange', kde=True, bins=120, label='0')
sns.distplot(t1[features].min(axis=0), color='red', kde=True, bins=120, label='1')
plt.legend()


# In[47]:


# dist of max per column
plt.figure()
plt.title('Distribution of max value per column')
sns.distplot(train_df[features].max(axis=0), color='green', kde=True, bins=120, label='train')
sns.distplot(test_df[features].max(axis=0), color='blue', kde=True, bins=120, label='test')
plt.legend()


# In[48]:


plt.figure()
plt.title('Distribution of max value per column, by target')
sns.distplot(t0[features].max(axis=0), color='orange', kde=True, bins=120, label='0')
sns.distplot(t1[features].max(axis=0), color='red', kde=True, bins=120, label='1')
plt.legend()


# In[49]:


proj_per_columns = pd.DataFrame({
    'min': train_df[features].min(axis=0),
    'mean': train_df[features].mean(axis=0),
    'median': train_df[features].median(axis=0),
    'max': train_df[features].max(axis=0),
})
proj_per_columns_sorted = proj_per_columns.sort_values(by='mean')
proj_per_columns_sorted_by_median = proj_per_columns.sort_values(by='median')


# In[50]:


plt.figure()
plt.title('Projection of mean value along columns, sorted by mean')
sns.scatterplot(data=proj_per_columns_sorted['min'].values, label='min')
sns.scatterplot(data=proj_per_columns_sorted['mean'].values, label='mean')
sns.scatterplot(data=proj_per_columns_sorted['max'].values, label='max')
plt.legend()


# In[51]:


plt.figure()
plt.title('Projection of mean value along columns, sorted by median')
sns.scatterplot(data=proj_per_columns_sorted_by_median['min'].values, label='min')
sns.scatterplot(data=proj_per_columns_sorted_by_median['mean'].values, label='mean')
sns.scatterplot(data=proj_per_columns_sorted_by_median['max'].values, label='max')
sns.scatterplot(data=proj_per_columns_sorted_by_median['median'].values, label='median')
plt.legend()


# In[52]:


# # 2d diagram
# plt.figure(figsize=(16, 16))
# plt.title('Relationships between pairs of features (columns)')

# sns.PairGrid(data=train_df[features]).map_diag(plt.scatter)


# In[53]:


corr = train_df[features].corr()


# In[54]:


corr = corr.reset_index().melt(id_vars='index', var_name='var1')    .set_index(['index', 'var1']).abs().sort_values(by='value')    .reset_index()


# In[55]:


corr.loc[corr['index'] != corr['var1']].tail(20)


# In[56]:


vs = corr.loc[corr['index'] != corr['var1']]['value']
plt.figure()
sns.distplot(vs, kde=True, bins=120, color='green')


# In[57]:


# train_df[features].T.corr()


# In[58]:




