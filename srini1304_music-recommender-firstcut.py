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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


train_df = pd.read_csv('../input/train.csv')


# In[3]:


songs_df = pd.read_csv('../input/songs.csv')


# In[4]:


members_df = pd.read_csv('../input/members.csv')


# In[5]:


train_df.head()


# In[6]:


train_df.info()


# In[7]:


train_df['source_system_tab'].nunique()


# In[8]:


train_df['source_screen_name'].nunique()


# In[9]:


train_df['source_type'].nunique()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


train_df.isnull().aggregate('sum')


# In[13]:


plt.figure(figsize=(14,7))
sns.countplot(x='source_system_tab', data=train_df, palette='RdBu_r')


# In[14]:


plt.figure(figsize=(14, 7))
sns.countplot(x='source_screen_name', data=train_df, palette='RdBu_r')


# In[15]:


plt.figure(figsize=(12,6))
sns.countplot(x='source_type', data=train_df, palette='RdBu_r')


# In[16]:


temp_df = train_df.groupby(['source_system_tab', 'source_screen_name'])['target'].aggregate(['count', 'sum']).reset_index()


# In[17]:


temp_df['ratio'] = temp_df['sum']/temp_df['count']


# In[18]:


temp_df_pivot = temp_df.pivot('source_system_tab', 'source_screen_name', 'ratio')


# In[19]:


temp_df_pivot.head()


# In[20]:


plt.figure(figsize=(14, 7))
sns.heatmap(temp_df_pivot, annot=True)


# In[21]:


temp_df_pivot = temp_df.pivot('source_system_tab', 'source_screen_name', 'count')
plt.figure(figsize=(14,7))
sns.heatmap(temp_df_pivot)


# In[22]:


temp_df['count'].hist(bins=100)


# In[23]:


target_count_df = train_df.groupby(['source_system_tab', 'source_screen_name', 'source_type'])['target'].aggregate('count').reset_index()


# In[24]:


target_sum_df = train_df.groupby(['source_system_tab', 'source_screen_name', 'source_type'])['target'].aggregate('sum').reset_index()


# In[25]:


target_count_df['target_count'] = target_sum_df['target']


# In[26]:


target_count_df['ratio'] = target_count_df['target_count']/target_count_df['target']


# In[27]:


target_count_df.head()


# In[28]:


all3_cat_prob = target_count_df.copy()


# In[29]:


del target_count_df, target_sum_df


# In[30]:


all3_cat_prob.sort_values('target', ascending=False)


# In[31]:


source_system_df = train_df.groupby(['source_system_tab'])['target'].aggregate(['count', 'sum']).reset_index()


# In[32]:


source_system_df['ratio'] = source_system_df['sum']/source_system_df['count']


# In[33]:


source_system_df.head(20)


# In[34]:


source_screen_df = train_df.groupby(['source_screen_name'])['target'].aggregate(['count', 'sum']).reset_index()


# In[35]:


source_screen_df['ratio'] = source_screen_df['sum']/source_screen_df['count']


# In[36]:


source_screen_df.head()


# In[37]:


source_type_df = train_df.groupby(['source_type'])['target'].aggregate(['count', 'sum']).reset_index()


# In[38]:


source_type_df['ratio'] = source_type_df['sum']/source_type_df['count']


# In[39]:


source_type_df.head()


# In[40]:


combined_df = pd.merge(train_df, songs_df, on='song_id', how='inner')


# In[41]:


combined_df.head()


# In[42]:


combined_final_df = pd.merge(combined_df, members_df, on='msno', how='inner')


# In[43]:


combined_final_df.head()


# In[44]:


del combined_df, songs_df, members_df


# In[45]:


artistwise_target_df = combined_final_df.groupby('artist_name')['target'].aggregate(['count', 'sum']).reset_index()


# In[46]:


artistwise_target_df.sort_values('count', ascending=False).head(10)


# In[47]:


artistwise_target_df['ratio'] = artistwise_target_df['sum']/artistwise_target_df['count']


# In[48]:


artistwise_target_df[artistwise_target_df['ratio'] != 1.0].sort_values('ratio', ascending=False).head(10)


# In[49]:


langwise_target_df = combined_final_df.groupby('language')['target'].aggregate(['count', 'sum']).reset_index()
langwise_target_df['ratio'] = langwise_target_df['sum']/langwise_target_df['count']


# In[50]:


langwise_target_df.head(10)


# In[51]:


combined_final_df['song_length'].describe()


# In[52]:


combined_final_df.columns


# In[53]:


combined_final_df['registered_via'].unique()


# In[54]:


combined_final_df['registration_init_time'].nunique()


# In[55]:


combined_final_df['msno'].nunique()


# In[56]:


combined_final_df['expiration_date'].nunique()


# In[57]:


combined_final_df['song_id'].nunique()


# In[58]:


trial_comb_df = combined_final_df.copy()


# In[59]:


trial_comb_df.columns


# In[60]:


del trial_comb_df['msno']


# In[61]:


trial_comb_df.columns


# In[62]:


del trial_comb_df['song_id']


# In[63]:


trial_comb_df.isnull().aggregate('sum')/trial_comb_df.shape[0]


# In[64]:


del trial_comb_df['gender']
del trial_comb_df['composer']
del trial_comb_df['lyricist']


# In[65]:


trial_comb_df.isnull().aggregate('sum')/trial_comb_df.shape[0]


# In[66]:


trial_comb_df['language'].fillna('median', inplace=True)


# In[67]:


trial_comb_df['source_system_tab'].fillna('median', inplace=True)


# In[68]:


trial_comb_df['genre_ids'].fillna('median', inplace=True)


# In[69]:


trial_comb_df['source_type'].fillna('median', inplace=True)


# In[70]:


trial_comb_df['source_screen_name'].fillna('median', inplace=True)


# In[71]:


trial_comb_df.info()


# In[72]:


source_system_tab_dummies = pd.get_dummies(trial_comb_df['source_system_tab'], drop_first=True)


# In[73]:


source_screen_name_dummies = pd.get_dummies(trial_comb_df['source_screen_name'], drop_first=True)


# In[74]:


source_type_dummies = pd.get_dummies(trial_comb_df['source_type'], drop_first=True)


# In[75]:


genre_id_dummies = pd.get_dummies(trial_comb_df['genre_ids'], drop_first=True)


# In[76]:


del trial_comb_df['artist_name']


# In[77]:


trial_comb_df['expiration_date'].nunique()


# In[78]:


all3_cat_prob.info()


# In[79]:


all3_cat_prob['target_prob'] = all3_cat_prob['target']/all3_cat_prob['target'].aggregate('sum')


# In[80]:


all3_cat_prob['target_count_prob'] = all3_cat_prob['target_count']/all3_cat_prob['target_count'].aggregate('sum')


# In[81]:


all3_cat_prob.head()


# In[82]:


overall_prob = all3_cat_prob['target_count'].aggregate('sum')/all3_cat_prob['target'].aggregate('sum')


# In[83]:


overall_prob


# In[84]:


test_df = pd.read_csv('../input/test.csv')


# In[85]:


test_df.head()


# In[86]:


del test_df['msno']


# In[87]:


del test_df['song_id']


# In[88]:


test_df = pd.merge(test_df, all3_cat_prob, on =['source_system_tab', 'source_screen_name', 'source_type'], how='left')


# In[89]:


del test_df['target']


# In[90]:


del test_df['target_count']


# In[91]:


del test_df['target_prob']


# In[92]:


del test_df['target_count_prob']


# In[93]:


test_df.head()


# In[94]:


test_df.isnull().aggregate('sum')


# In[95]:


test_df['ratio'].fillna(0.5055, inplace=True)


# In[96]:


test_df.isnull().aggregate('sum')


# In[97]:


submit_df = pd.DataFrame()


# In[98]:


submit_df['id'] = test_df['id']
submit_df['target'] = test_df['ratio']


# In[99]:


submit_df.head()


# In[100]:


ids = test_df['id'].values


# In[101]:


targets = test_df['ratio'].values


# In[102]:


submit_df.to_csv('submission.csv', index=False, float_format='%.5f')


# In[103]:


temp = pd.read_csv('submission.csv')


# In[104]:


temp.head()


# In[105]:




