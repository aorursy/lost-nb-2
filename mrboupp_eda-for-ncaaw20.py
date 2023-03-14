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


import matplotlib.pyplot as plt
import seaborn as sns
import collections


# In[3]:


WTeams = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WTeams.csv')

WTeams


# In[4]:


WSeasons = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WSeasons.csv')

WSeasons


# In[5]:


WTouneySeeds = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySeeds.csv')

WTouneySeeds


# In[6]:


SeasonResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

SeasonResults


# In[7]:


TouneyResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')

TouneyResults


# In[8]:


RegularDetailedResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonDetailedResults.csv')

RegularDetailedResults


# In[9]:


NCAADetailedResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyDetailedResults.csv')

NCAADetailedResults


# In[10]:


Cities = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/Cities.csv')

Cities


# In[11]:


GameCities = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WGameCities.csv')

GameCities


# In[12]:


WEvents2015 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2015.csv')
WEvents2016 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2016.csv')
WEvents2017 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2017.csv')
WEvents2018 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2018.csv')
WEvents2019 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2019.csv')

WEvents2015


# In[13]:


Player = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WPlayers.csv')

Player


# In[14]:


Slots = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySlots.csv')

Slots


# In[15]:


WTouneySeeds


# In[16]:


WTeams


# In[17]:


pd.merge(WTouneySeeds,WTeams,on='TeamID').set_index('TeamID')


# In[18]:


SeasonResults


# In[19]:


collections.Counter(SeasonResults['WLoc'])


# In[20]:


SeasonResults['WLoc'].value_counts()


# In[21]:


SeasonResults['WLoc'].value_counts().plot(kind='bar')
plt.legend()
plt.show()


# In[22]:


SeasonResults['WScore'].value_counts()


# In[23]:


fig,ax=plt.subplots(1,figsize=(15,10))

sns.kdeplot(SeasonResults['WScore'],color='green',shade=True,ax=ax)
sns.kdeplot(SeasonResults['LScore'],shade=True,ax=ax)

plt.legend()
plt.show()


# In[24]:


SortWScore= SeasonResults.sort_values('WScore',ascending=False).head(10)

SortWScore


# In[25]:


sns.barplot(x='WScore',y='WTeamID',data=SortWScore,palette='Set3_r',orient="h")


# In[26]:


fig,ax=plt.subplots(1,figsize=(15,10))

sns.kdeplot(SeasonResults.loc[(SeasonResults['WLoc']=='H'),'WScore'],color='green',shade=True,ax=ax)
sns.kdeplot(SeasonResults.loc[(SeasonResults['WLoc']=='A'),'WScore'],color='red',shade=True,ax=ax)
sns.kdeplot(SeasonResults.loc[(SeasonResults['WLoc']=='N'),'WScore'],color='blue',shade=True,ax=ax)

plt.show()


# In[27]:


WScoreFrequency = SeasonResults['WScore'].value_counts()
WScoreFrequency.index.names = ['WScore']

LScoreFrequency = SeasonResults['LScore'].value_counts()


# In[28]:


SortW = SeasonResults.sort_values('WScore',ascending=False)
SortL = SeasonResults.sort_values('LScore',ascending=False)

SortW['WScore'].plot(kind='hist',bins=50,label='WScore',alpha=0.5)
SortL['LScore'].plot(kind='hist',bins=50,label='LScore',alpha=0.5)
plt.title('Score Frequency')
plt.legend()
plt.show()


# In[29]:


SeasonResults.groupby(['WTeamID','LTeamID']).size().unstack().fillna(0).style.background_gradient(axis=1)


# In[30]:


SeasonResults['counter']=1
SeasonResults.groupby('WTeamID')['counter']     .count()     .sort_values()     .tail(20)     .plot(kind='barh',figsize=(15,8),xlim=(400,680))
plt.show()


# In[31]:


WEvents2015.head()


# In[32]:


WEvents2015['counter']=1
WEvents2015.groupby('EventType')['counter']     .sum()     .sort_values(ascending=False)    .plot(kind='bar',figsize=(15,5),title='Event Type Frequency 2015')
plt.show()

