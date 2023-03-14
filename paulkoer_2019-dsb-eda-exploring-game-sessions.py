#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


PATH = '/kaggle/input/data-science-bowl-2019/'


# In[3]:


df_train = pd.read_csv(f'{PATH}train.csv')


# In[4]:


all_installation_ids = df_train['installation_id'].drop_duplicates()
all_installation_ids.count()


# In[5]:


ids_to_keep = df_train[df_train.type == 'Assessment']['installation_id'].drop_duplicates()


# In[6]:


df_train = pd.merge(df_train, ids_to_keep, on='installation_id', how='inner')


# In[7]:


ids_to_keep.count()


# In[8]:


sample_installation_ids = ids_to_keep.sample(800)
df_sample = pd.merge(df_train, sample_installation_ids, on='installation_id', how='inner')


# In[9]:


df_sample.describe()


# In[10]:


df_sample.head()


# In[11]:


df_sample['timestamp'] = pd.to_datetime(df_sample['timestamp'])


# In[12]:


df_sample.groupby('installation_id')     .count()['event_id']     .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
         title='Count of Observations by installation_id')
plt.show()


# In[13]:


df_sample.groupby('type')     .count()['event_id']     .plot(kind='barh',
          figsize=(15, 5),
         title='Count of Event Types')
plt.show()


# In[14]:


for act_type in ['Assessment', 'Clip', 'Game', 'Activity']:
    df_sample[df_sample['type'] == act_type].groupby('installation_id')         .count()['event_id']        .plot(kind='hist',
            bins=100,
            figsize=(12, 4),
            title=f'{act_type} by installation_id')
    plt.show()


# In[15]:


df_sample.groupby('installation_id')['timestamp']    .transform(lambda x: (x.max() - x.min()).days)    .plot(kind='hist',
          bins=100,
          figsize=(12, 4),
          title=f'Time played by installation_id')
plt.show()


# In[16]:


df_sample.groupby('installation_id')['game_session']    .transform(lambda x: x.nunique())    .plot(
        kind='hist',
        bins=100,
        figsize=(12, 4),
        title='Game sessions per installation id')
plt.show()


# In[17]:


game_sessions = df_sample.groupby('installation_id')['game_session'].nunique()


# In[18]:


dates_plaid = df_sample.groupby('installation_id')['timestamp'].apply(lambda x: (x.max() - x.min()).days)


# In[19]:


plt.scatter(
    dates_plaid, 
    game_sessions)
plt.xlabel('days played')
plt.ylabel('game sessions')
plt.title('Game sessions vs days played')


# In[20]:


sample_session_length = df_sample.groupby('game_session')['timestamp']    .transform(lambda x: (x.max() - x.min()).delta / 60_000_000_000)
sample_session_length = sample_session_length[sample_session_length <= 40]

sample_session_length.plot(kind='hist',
    bins=100,
    figsize=(12, 4),
    title=f'Session length in minutes (sessions under 40 minutes)')
plt.show()


# In[21]:


session_length = df_sample.groupby('game_session')['timestamp']    .apply(lambda x: (x.max() - x.min()).delta / 60_000_000_000)
session_count = df_sample.groupby('game_session')['event_id']    .apply(lambda x: x.count())


# In[22]:


scatter_index = (session_length <= 120)
plt.figure(figsize=(15, 9))
plt.scatter(
    session_length[scatter_index], 
    session_count[scatter_index], 
    alpha=0.05)
plt.title('Game session length vs game event count')
plt.xlabel('Game session length in minutes')
plt.ylabel('Game event count')
plt.show()

