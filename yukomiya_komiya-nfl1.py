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


df_all = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv")


# In[3]:


df_all.head()


# In[4]:


df_all.columns


# In[5]:


df_all[['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A',
        'Dis', 'Orientation', 'Dir', 'NflId', 'DisplayName', 'JerseyNumber']].head()


# In[6]:


df_all[['Season', 'YardLine', 'Quarter', 'GameClock', 
        'PossessionTeam', 'Down', 'Distance', 'FieldPosition']].head()


# In[7]:


df_all[['HomeScoreBeforePlay', 'VisitorScoreBeforePlay',
       'NflIdRusher', 'OffenseFormation', 'OffensePersonnel',
       'DefendersInTheBox', 'DefensePersonnel']].head()


# In[8]:


df_all[['PlayDirection', 'TimeHandoff','TimeSnap', 'Yards']].head()


# In[9]:


df_all[['PlayerHeight', 'PlayerWeight', 'PlayerBirthDate',
       'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr']].head()


# In[10]:


df_all[['Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather',
       'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']].head()


# In[11]:


# 選手一人毎の情報がある列
# 他の列は同一PlayId内では同じ
personal_columms = ['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 
                    'NflId', 'DisplayName', 'JerseyNumber', 
                   'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate','PlayerCollegeName', 'Position']


# In[12]:


df_all.shape


# In[13]:


pd.set_option("display.max_columns", 80)


# In[14]:


df_all.describe()


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


plt.tight_layout() 

df_all.hist()
#plt.show()


# In[17]:


df_all.columns


# In[18]:


# ボールを持っている選手の行のみを選択
df_play = df_all[df_all["NflId"]==df_all["NflIdRusher"]]


# In[19]:


df_play.tail()


# In[20]:


df_play["Position"].value_counts()


# In[21]:


df_play[df_play["Position"].isin(["DT", "DE", "CB"])]


# In[22]:


personal_columms


# In[23]:


import seaborn as sns


# In[24]:


sns.jointplot('A', 'Yards', data=df_play)


# In[25]:


sns.jointplot('YardLine', 'Yards', data=df_play[df_play["PossessionTeam"] == df_play["FieldPosition"]])


# In[26]:


sns.jointplot('YardLine', 'Yards', data=df_play[df_play["PossessionTeam"] != df_play["FieldPosition"]])


# In[27]:


sns.jointplot('X', 'Yards', data=df_all[df_all["Position"]=="HB"])


# In[28]:


df_play["Yards"].hist(range=(-20,30),bins=50)


# In[29]:


df_play["Yards"]


# In[30]:


from math import erf


# In[31]:


erf(1)


# In[32]:


df_all["Position"].unique()


# In[33]:


# オフェンスチーム
# QB
# RB（FB、HB、TB）
# WR（SE、FL、SB、WB）
# TE
# OL（C、G、T、E）
# ディフェンスチーム
# DL（DT（NT）、DE）
# LB（ILB（MLB）、OLB（LOLB、ROLB、SLB、WLB））
# DB（CB、S（SS、FS））
# スペシャルチーム
# K、P、LS、H、KR/PR

offence_position = ['WR', 'TE', 'T', 'QB', 'RB', 'G', 'C', 'FB', 'HB',  'OT', 'OG', ]
#OL_position = ['T', 'G', 'C', 'OT', 'OG']
#RB_position = ['RB', 'FB', 'HB']

defence_position = ['SS', 'DE', 'ILB', 'FS', 'CB', 'DT', 'OLB', 'NT', 'MLB', 'LB', 'S', 'DL', 'DB', 'SAF']
#DL_position = ['DL', 'DT', 'DE']
#LB_position = ['LB', 'ILB', ]


# In[34]:


import math


# In[35]:


df_all["offence"] = 0
df_all.loc[df_all["Position"].isin(offence_position), "offence"] = 1


# In[36]:


df_all.head(23)[["Team", "Position", "offence"]]


# In[37]:


position_count=df_all.groupby(["PlayId", "Position"]).count()
position_count


# In[38]:


df_position = position_count["GameId"].unstack().fillna(0).astype(int)
df_position


# In[39]:


pd.merge(df_play, df_position, on="PlayId").corr()["Yards"].sort_values()


# In[40]:


df_play


# In[41]:


df_play.loc[:, "OffensePersonnel"] = df_play["OffensePersonnel"].apply(
    lambda x : { i.split(" ")[-1]:int(i.split(" ")[-2]) for i in x.split(",")} )


# In[42]:


df_play.loc[:, "DefensePersonnel"] = df_play["DefensePersonnel"].apply(
    lambda x : { i.split(" ")[-1]:int(i.split(" ")[-2]) for i in x.split(",")} )


# In[43]:


for position in ["DL", "LB", "DB"]:
    df_play.loc[:, position] = [ d[position] for d in df_play["DefensePersonnel"]]
for position in ["RB", "TE", "WR"]:
    df_play.loc[:, position] = [ d[position] for d in df_play["OffensePersonnel"]]    


# In[44]:


df_play


# In[45]:


df_play.corr()["Yards"].sort_values()


# In[46]:


df_play[df_play["Team"]=="home"].groupby("HomeTeamAbbr").mean()["Yards"].sort_values()


# In[47]:


team_home = df_play[df_play["Team"]=="home"].groupby("HomeTeamAbbr")


# In[48]:


df_play["HomeTeamAbbr"].unique()


# In[49]:


team_yards = dict(list(df_play[df_play["Team"]=="home"].groupby("HomeTeamAbbr")["Yards"]))


# In[50]:


plt.hist(team_yards["CAR"], bins=30)
plt.hist(team_yards["WAS"], bins=30)


# In[51]:


dict


# In[52]:


df_play[df_play["Team"]=="home"].groupby("HomeTeamAbbr").mean()["Yards"].hist()


# In[53]:


df_play[df_play["Team"]=="away"].groupby("VisitorTeamAbbr").mean()["Yards"].hist()


# In[54]:


np.array([[1,2,2,5]]).mean()


# In[55]:


df["OffenseFormation"].unique()


# In[56]:


df["OffensePersonnel"].unique()


# In[57]:


df["StadiumType"]


# In[58]:


df["stadiumsype"] = df["StadiumType"].isin()


# In[59]:


df["PlayDirection"].unique()


# In[60]:


personal_yards = df_play[["NflId", "Yards"]].groupby("NflId").agg(["mean", "std", "max", "min", "count"])["Yards"]
personal_yards.dropna(inplace=True)


# In[61]:


personal_yards


# In[62]:


df_play = df_play.merge(personal_yards, on="NflId", how="left")


# In[63]:


df_play.corr()["Yards"].sort_values()


# In[64]:


len(df_play.index)


# In[65]:


df_play["Yards"].value_counts().head(30)


# In[66]:


pd.


# In[67]:


df_play[df_play["Yards"]>50].sort_values("Yards")[["YardLine", "Yards"]]


# In[ ]:





# In[68]:


len(df_play[df_play["Yards"]+df_play["YardLine"]==100].index)


# In[69]:


df_play[df_play["Yards"]>20].corr()["Yards"].sort_values()


# In[70]:


df_play.loc[:, "over20"] = (df_play["Yards"]>20)*1


# In[71]:


df_play.loc[:, "touchdown"] = (df_play["Yards"]+df_play["YardLine"]==100)


# In[72]:


(df_play["Yards"]+df_play["YardLine"]==100)


# In[73]:


df_play[["over20", "touchdown"]]


# In[74]:


df_play.corr()["touchdown"].sort_values()


# In[75]:


df_play.corr()["over20"].sort_values()


# In[ ]:




