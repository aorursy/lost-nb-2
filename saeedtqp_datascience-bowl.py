#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from IPython.display import HTML
import warnings
pd.set_option('max_columns',100)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
my_pal = sns.color_palette(n_colors = 10)


# In[2]:


train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
train_labels  = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
space = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
ss = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')


# In[3]:


train_ = train.sample(1000000,replace=True)


# In[4]:


train_labels.head()


# In[5]:


train_labels.groupby('accuracy_group')['game_session'].count()    .plot(kind = 'bar' , figsize = (15,5),title = 'Target (accuracy group)')
plt.show()


# In[6]:


sns.pairplot(train_labels , hue = 'accuracy_group')
plt.show()


# In[7]:


train.head()


# In[8]:


train['event_id_as_int'] = train['event_id'].apply(lambda x:int(x,16))


# In[9]:


train.head()


# In[10]:


# Format and make data / hour features

train['timestamp'] = pd.to_datetime(train['timestamp'])
train['date'] = train['timestamp'].dt.date
train['hour'] = train['timestamp'].dt.hour
train['weekday_name'] = train['timestamp'].dt.weekday_name

# Same For test

test['timestamp'] = pd.to_datetime(test['timestamp'])
test['date'] = test['timestamp'].dt.date
test['hour'] = test['timestamp'].dt.hour
test['weekday_name'] = test['timestamp'].dt.weekday_name


# In[11]:


print(f'Train data has shape : {train.shape}')
print(f'Test data has shape : {test.shape}')
      


# In[12]:


train.groupby('date')['event_id']     .agg('count')    .plot(figsize = (15,3),
    title = 'Number Of Event Observation by Date',
    color = my_pal[2])
plt.show()


# In[13]:


train.groupby('hour')['event_id']     .agg('count')    .plot(figsize = (15,3),
    title = 'Number of event Observations by Hour',
    color = my_pal[1])
plt.show()


# In[14]:


train.groupby('weekday_name')['event_id']     .agg('count').T[['Monday','Tuesday','Wednesday',
                     'Thursday','Friday','Saturday',
                     'Sunday']].T.plot(figsize=(15, 3),
                                       title='Numer of Event Observations by Day of Week',
                                       color=my_pal[3])
plt.show()


# In[15]:


print(train['event_data'][4])
print(train['event_data'][5])


# In[16]:


train['installation_id'].nunique()


# In[17]:


train.groupby('installation_id')    .count()['event_id']    .plot(kind = 'hist',
    bins = 40,
    color = my_pal[4],
    figsize = (15,5),
     title = 'Count of Observation by installation_id')
plt.show()


# In[18]:


# natural log x + 1
train.groupby('installation_id')    .count()['event_id']   .apply(np.log1p)    .plot(kind = 'hist',
   bins = 40,
   color = my_pal[6],
   figsize = (15,5),
   title = 'log(Count) of Observations by installation_id')
plt.show()


# In[19]:


train.groupby('installation_id')     .count()['event_id'].sort_values(ascending = False).head(5)


# In[20]:


train.query('installation_id == "f1c21eda"')     .set_index('timestamp')['event_code']     .plot(figsize = (15,5),
         title = 'installation_id #f1c21eda event Id - event code vs time',
         style = '.',
         color = my_pal[8])
plt.show()


# In[21]:


train.groupby('event_code')     .count()['event_id']     .sort_values()     .plot(kind = 'bar',
         figsize = (15,5),
         title = 'Count of diffretnt event codes.')
plt.show()


# In[22]:


# Game_time

train['game_time'].apply(np.log1p)     .plot(kind = 'hist',
    figsize = (15,5),
    bins = 100,
    title = 'Log Transform of game_time',
    color = my_pal[1])

plt.show()


# In[23]:


train.groupby('title')['event_id']     .count()     .sort_values()     .plot(kind = 'barh',
         title = 'Count of Ob by Game / video title',
         figsize = (15,15))
plt.show()


# In[24]:


train.groupby('type')['event_id']     .count()     .sort_values()     .plot(kind = 'bar',
         figsize = (15,4),
         title = 'Count by Type',
         color = my_pal[2])
plt.show()


# In[25]:


train.groupby('world')['event_id']     .count()     .sort_values()     .plot(kind = 'bar',
        figsize = (15,4),
        title = 'Count by world',
        color = my_pal[3])
          
plt.show()


# In[26]:


# Log (game_time) vs game/video categories

train['log1p_game_time'] = train['game_time'].apply(np.log1p)


# In[27]:


fig , ax = plt.subplots(figsize = (15,5))
sns.catplot(x = 'type',y = 'log1p_game_time',
           data = train.sample(10000), alpha = 0.5 ,ax = ax);
ax.set_title('Distribution of log1p(game_time by type)')
plt.close()
plt.show()


# In[28]:


fig , ax = plt.subplots(figsize = (15,5))
sns.catplot(x = 'world' , y = 'log1p_game_time',
           data = train.sample(10000),alpha = 0.5,ax = ax)
ax.set_title('Distribution of log1p(game_time) by World')
plt.close()
plt.show()


# In[29]:


space.head()


# In[30]:


space.describe()


# In[31]:


train['cleared'] = True

train.loc[train['event_data'].str.contains('false') & train['event_code'].isin([4100,4110]),'cleared'] = False


# In[32]:


test['cleared'] = True
test.loc[test['event_data'].str.contains('false') & test['event_code'].isin([4100,4110]),'cleared'] = False

aggs = {'hour':['max','min','mean'],
       'cleared':['mean']}

train_aggs = train.groupby('installation_id').agg(aggs)
test_aggs = test.groupby('installation_id').agg(aggs)
train_aggs = train_aggs.reset_index()
test_aggs = test_aggs.reset_index()
train_aggs.columns = ['_'.join(col).strip() for col in train_aggs.columns.values]
test_aggs.columns = ['_'.join(col).strip() for col in test_aggs.columns.values]
train_aggs = train_aggs.rename(columns={'installation_id_' : 'installation_id'})


# In[33]:


train_aggs.merge(train_labels[['installation_id','accuracy_group']],
               how = 'left')


# In[ ]:




