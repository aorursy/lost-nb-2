#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from scipy.stats import skew
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


teams = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv')


# In[3]:


def get_team_dict(df):
    group = df.groupby(['TeamID','TeamName']).size().reset_index()
    team_ids = list(group['TeamID'])
    team_names = list(group['TeamName'])
    return dict((x,y) for x,y in list(zip(team_ids, team_names)))

team_dict = get_team_dict(teams)


# In[4]:


players = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MPlayers.csv')


# In[5]:


def get_player_dict(df):
    lol = players.groupby(['PlayerID','LastName','FirstName']).size().reset_index()
    lol['Name'] = lol['FirstName'] + ['_']*lol.shape[0] + lol['LastName']
    player_ids = list(lol['PlayerID'])
    player_names = list(lol['Name'])
    return dict((x,y) for x,y in list(zip(player_ids, player_names)))


# In[6]:


results = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')
results = results.drop('WLoc', axis = 1) #useless col (1115 'N' values)


# In[7]:


def plot_kde(results, col):
    
    win_col = 'W' + col
    lost_col = "L" + col
    
    plt.figure(figsize = (12,5))
    plt.title("Team avg per match " + col)
    sns.kdeplot(results[win_col])
    sns.kdeplot(results[lost_col])
    
w_l_cols = [re.sub('(L|W){1}', "", i) for i in results.columns if re.match('[LW]{1}', i)]
stat_cols = list(set(w_l_cols))
stat_cols.remove('TeamID')

for col in stat_cols:
    plot_kde(results, col)


# In[8]:


def plot_kde(results, col):
    
    win_col = 'W' + col
    lost_col = "L" + col
    
    plt.figure(figsize = (12,5))
    sns.kdeplot(results[win_col])
    sns.kdeplot(results[lost_col])


# In[9]:


team_dict = get_team_dict(teams)
results['WTeam'] = results.WTeamID.map(lambda x:team_dict[x])
results['LTeam'] = results.LTeamID.map(lambda x:team_dict[x])


# In[10]:


def basic_eda(train):
    tr_nulls = train.isnull().sum()
    for col in train.columns[1:]:
        if train[col].dtype != object:
            plt.figure(figsize = (12,4))
            plt.title('%s has %s nulls and %s nunique, %s dtype, %s skew' %(col,tr_nulls[col], train[col].nunique(), train[col].dtype, skew(train[col])))
            sns.distplot(train[col].dropna())
        if train[col].dtype == object:
            print ('%s has %s nuniq' %(col, train[col].nunique()))
            print (train[col].unique()[:10])


# In[11]:


ev_16 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2016.csv')
ev_17 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2017.csv')
ev_18 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2018.csv')
ev_19 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2019.csv')
players = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MPlayers.csv')


# In[12]:


ev_full = pd.concat([ev_16, ev_17, ev_18, ev_19], ignore_index = True)
del ev_16, ev_17, ev_18, ev_19


# In[13]:


ev_full['win_bool'] = ev_full['WTeamID'] - ev_full['EventTeamID']
ev_full['win_bool'] = ev_full['win_bool'].map(lambda x: 1 if x == 0 else 0)

ev_full['event_name'] = ev_full['EventType'] + ['_'] * ev_full.shape[0] + ev_full['EventSubType']

ev_full.loc[ev_full['event_name'] == 'jumpb_start', 'EventTeamID'] = 1

#np.where to replace event_names whose EventTeamID == 0
a = np.array(['id0_error'] * ev_full.shape[0]) #error arr
b = np.array(list(ev_full['EventTeamID']))
c = np.array(list(ev_full['event_name']))
ev_full['event_name'] = np.where(b == 0, a, c)

#ignore player id 0 when eda player stats


# In[14]:


ev_full = ev_full.sort_values(['Season','DayNum','WTeamID','WCurrentScore','ElapsedSeconds']).reset_index().drop('index', axis = 1)


# In[15]:


val_cnts = ev_full['event_name'].value_counts()


# In[16]:


val_cnts[-30:-20]


# In[ ]:





# In[17]:


val_cnts[:45]


# In[18]:


win = ev_full[ev_full.win_bool == 1].groupby(['Season',"DayNum", "WTeamID", "event_name"]).size()
lost = ev_full[ev_full.win_bool == 0].groupby(['Season',"DayNum", "WTeamID", "event_name"]).size()


# In[19]:


lol = ev_full.iloc[7897547:7897557, :]
lol.head(20)


# In[20]:


lol = ev_full.iloc[7904503:7904509, :]
lol.head(20)


# In[21]:


ev_full[ev_full.event_name == "miss1_1of3"]


# In[22]:



evname = 'made2_stepb'
plt.figure(figsize = (15,4))


# In[23]:


for event in ['s':
    if match_group[match_group.event_name == event]['ev_count'].nunique() > 10:
    
        plt.figure(figsize = (10,4))
        plt.title(event)
        win_ax = sns.kdeplot(match_group[match_group.win_bool == 0][match_group.event_name == event].rename(columns = {'ev_count':'lost'})['lost'].dropna())
    
        lost_ax = sns.kdeplot(match_group[match_group.win_bool == 1][match_group.event_name == event].rename(columns = {'ev_count':'won'})['won'].dropna()) 


# In[24]:


ev_full[ev_full.event_name == 'jumpb_won']


# In[25]:


ev_name = 'jumpb_won'

lol_win = ev_full[ev_full.win_bool == 1].groupby(['Season','DayNum','WTeamID','event_name']).size().reset_index()
lol_lost = ev_full[ev_full.win_bool == 0].groupby(['Season','DayNum','WTeamID','event_name']).size().reset_index()
lol_win.columns = ['Season', 'DayNum', 'WTeamID', 'event_name', 'win_count']
lol_lost.columns = ['Season', 'DayNum', 'WTeamID', 'event_name', 'lost_count']


# In[26]:


for i in lol_win.event_name.unique():
    if 'jumpb' in i:
        print (i)


# In[27]:


ev_name = 'reb_def'


# In[28]:


lol_lost[lol_lost.event_name == ev_name]['lost_count'].value_counts()


# In[29]:


lol_win[lol_win.event_name == ev_name]['win_count'].value_counts()


# In[ ]:





# In[30]:


ev_name = 'made2_stepb'

plt.figure(figsize = (15,6))
sns.distplot(lol_win[lol_win.event_name == ev_name]['win_count'])
sns.plot(lol_lost[lol_lost.event_name == ev_name]['lost_count'])


# In[31]:


year = 2017
count_num = 15

lol_win = ev_full[ev_full.Season == year][ev_full.win_bool == 1].groupby(['DayNum','WTeamID','event_name'])size().reset_index()
lol_lost = ev_full[ev_full.Season == year][ev_full.win_bool == 0].groupby(['DayNum','WTeamID','event_name']).size().reset_index()

win1 = lol_win.groupby('event_name')['count'].mean().sort_values(ascending = False).reset_index()
lost1 = lol_lost.groupby('event_name')['count'].mean().sort_values(ascending = False).reset_index()

win_top = win1.head(count_num)
lost_top = lost1.head(count_num)

plt.figure(figsize = (20,6))

plt.subplot(1,2,1)
plt.title('win top %s of %s' %(count_num, year))
sns.barplot(x = 'event_name', y = 'count', data = win_top)
plt.xticks(rotation = 45)

plt.subplot(1,2,2)
plt.title('lost top %s of %s' %(count_num, year))
sns.barplot(x = 'event_name', y = 'count', data = lost_top)
plt.xticks(rotation = 45)


# In[32]:


full_diff_df = pd.DataFrame({'event_name':list(ev_full.event_name.unique())})


for year in [2016,2017,2018,2019]:

    lol_win = ev_full[ev_full.Season == year][ev_full.win_bool == 1].groupby(['DayNum','WTeamID','event_name'])['ElapsedSeconds'].agg(['count','mean']).reset_index()
    lol_lost = ev_full[ev_full.Season == year][ev_full.win_bool == 0].groupby(['DayNum','WTeamID','event_name'])['ElapsedSeconds'].agg(['count','mean']).reset_index()

    win1 = lol_win.groupby('event_name')['count'].mean().sort_values(ascending = False).reset_index()
    win1.columns = ['event_name', 'win_count']
    lost1 = lol_lost.groupby('event_name')['count'].mean().sort_values(ascending = False).reset_index()
    lost1.columns = ['event_name','lost_count']

    diff_df = pd.merge(win1, lost1, how = 'left', on = 'event_name')
    diff_df['diff'] = diff_df['win_count'] / diff_df['lost_count']
    diff_df = diff_df[['event_name','diff']]
    diff_df.columns = ['event_name', 'diff_' + str(year)]
    
    full_diff_df = pd.merge(full_diff_df, diff_df, how = 'left', on = 'event_name')


# In[33]:


event_names = full_diff_df.sort_values('diff_2017').head(23).event_name.unique()


# In[34]:


match_group = ev_full.groupby(['Season','DayNum','EventTeamID','event_name','win_bool'])['ElapsedSeconds'].count().reset_index()


# In[35]:


ev_full.columns


# In[36]:


match_group.columns = ['Season', 'DayNum', 'EventTeamID', 'event_name', 'win_bool',
       'ev_count']


# In[37]:


match_group.head()


# In[38]:


for event in event_names:
    if match_group[match_group.event_name == event]['ev_count'].nunique() > 20:
    
        plt.figure(figsize = (10,4))
        plt.title(event)
        win_ax = sns.kdeplot(match_group[match_group.win_bool == 0][match_group.event_name == event].rename(columns = {'ev_count':'lost'})['lost'].dropna())
    
        lost_ax = sns.kdeplot(match_group[match_group.win_bool == 1][match_group.event_name == event].rename(columns = {'ev_count':'won'})['won'].dropna())
    
    
    
    


# In[39]:


match_group = ev_full.groupby(['Season','DayNum','WTeamID','LTeamID'])['']


# In[40]:


#all of the nan event_names only exists in 2019
lol_list = list(lol[lol.diff_2016.isnull()].event_name)
lol_list.remove(np.nan)
ev_full[ev_full.event_name.isin(lol_list)].Season.unique()


# In[41]:


ev_full.isnull().sum()


# In[42]:


ev_full.EventSubType.unique()


# In[43]:


ev_full[ev_full.EventSubType == 'coate'].DayNum.unique()


# In[44]:


ev_names = list(ev_full.event_name.unique())


# In[45]:


ev_full.event_name.nunique()


# In[46]:


diff_df['diff'] = diff_df['win_count'] / diff_df['lost_count']


# In[47]:


pd.DataFrame({'event_name':list(ev_full.event_name.unique())})


# In[48]:


diff_df.sort_values('diff')


# In[49]:


lol_top = lol1.head(10)
plt.figure()
sns.barplot(x = 'event_name', y = 'count', data = lol_top)
plt.xticks(rotation = 45)


# In[50]:


lol1.head(1)


# In[51]:


lol.tail(3)


# In[52]:


for season in ev_full.Season.unique():
    lol = ev_full[ev_full.Season == 2017].groupby(['event_name'])['win_bool'].agg(['count','mean']).sort_values('mean').reset_index()
    lol['name_count'] = lol['event_name'] + ['_'].shape[0] + lol['count'].astype('str')
    top10 = lol.tail(10)
    bottom1


# In[53]:


sample_df = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')


# In[54]:


sample_df


# In[ ]:




