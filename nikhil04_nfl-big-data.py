#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from kaggle.competitions import nflrush
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
def label_bars(ax, bars, text_format, **kwargs):
    """
    Attaches a label on every bar of a regular or horizontal bar chart
    """
    ys = [bar.get_y() for bar in bars]
    y_is_constant = all(y == ys[0] for y in ys)  # -> regular bar chart, since all all bars start on the same y level (0)

    if y_is_constant:
        _label_bar(ax, bars, text_format, **kwargs)
    else:
        _label_barh(ax, bars, text_format, **kwargs)


def _label_bar(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    """
    max_y_value = ax.get_ylim()[1]
    inside_distance = max_y_value * 0.05
    outside_distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2

        is_inside = bar.get_height() >= max_y_value * 0.15
        if is_inside:
            color = "white"
            text_y = bar.get_height() - inside_distance
        else:
            color = "black"
            text_y = bar.get_height() + outside_distance

        ax.text(text_x, text_y, text, ha='center', va='bottom', color=color, **kwargs)


def _label_barh(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    Note: label always outside. otherwise it's too hard to control as numbers can be very long
    """
    max_x_value = ax.get_xlim()[1]
    distance = max_x_value * 0.0025

    for bar in bars:
        text = text_format.format(bar.get_width())

        text_x = bar.get_width() + distance
        text_y = bar.get_y() + bar.get_height() / 2

        ax.text(text_x, text_y, text, va='center', **kwargs)


# In[3]:


train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
train_df.head()


# In[4]:


train_df.shape


# In[5]:


env = nflrush.make_env()
iter_test = env.iter_test()
(test_df, sample_prediction_df) = next(iter_test)
test_df.head()


# In[6]:


sample_prediction_df.head()


# In[7]:


print(test_df.shape)
print(len(test_df['DisplayName'].unique()))


# In[8]:


train_df.columns


# In[9]:


train_df.info()


# In[10]:


missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values>0]
missing_values.sort_values(ascending=False,inplace=True)
missing_values


# In[11]:


print('Total of Games Played: ', len(train_df.GameId.unique()))


# In[12]:


train_df.Team.value_counts()


# In[13]:


train_df.groupby('PlayId').first()['Yards'].plot(
    kind = 'hist',
    figsize=(15,5),
    bins=50,
    title='Distribution of yard gain'
)


# In[14]:


color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

fig, axes = plt.subplots(4,1, figsize=(15,8), sharex=True)
n=0
for i, d in train_df.groupby('Down'):
    d['Yards'].plot(kind='hist', 
                    bins=30,
                   color=color_pal[n],
                   ax=axes[n],
                   title = f'Yards Gained on down {i}')
    n+=1


# In[15]:


fig, ax= plt.subplots(figsize=(20,5))
sns.violinplot(x='Distance-to-Gain',
              y='Yards',
              data=train_df.rename(columns={'Distance':'Distance-to-Gain'}),
              ax=ax)
plt.ylim(-10,20)
plt.title('Yards vs Distance-to-gain')
plt.show()


# In[16]:


print('Unique game data provided: {}'.format(train_df['GameId'].nunique()))
print('Unique play data provided: {}'.format(train_df['PlayId'].nunique()))


# In[17]:


train_df.groupby('GameId')['PlayId'].nunique().plot(kind='hist',
                                                   figsize=(15,5),
                                                   title='Distribution of plays per gameid',
                                                   bins=50)


# In[18]:


fig , (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
sns.boxplot(data = train_df.groupby('PlayId').first()[['Distance','Down']], x='Down', y= 'Distance', ax=ax1)
ax1.set_title('Distance-to-Gain by Down')
sns.boxplot(data = train_df.groupby('PlayId').first()[['Yards','Down']], x='Down', y= 'Yards', ax=ax2)
ax2.set_title('Yards-to-Gain by Down')


# In[19]:


# train_df['Distance'].plot(kind='hist', title='Distribution of distance to go', figsize=(15,5), bins=30)

sns.distplot(train_df['Distance'])


# In[20]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,4))
train_df['S'].plot(kind='hist', title="Distribution of speed",ax=ax1, bins=20)
train_df['A'].plot(kind='hist', title="Distribution of Acceleration",ax=ax2, bins=20)
train_df['Dis'].plot(kind='hist', title="Distribution of Distance",ax=ax3, bins=20)


# In[21]:


fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,4))

train_df.query('NflIdRusher == NflId')['S'].plot(kind='hist', title='Distribution of speed (ball carrier only)', ax=ax1, bins=20)
train_df.query('NflIdRusher == NflId')['A'].plot(kind='hist', title='Distribution of speed (ball carrier only)', ax=ax2, bins=20)
train_df.query('NflIdRusher == NflId')['Dis'].plot(kind='hist', title='Distribution of speed (ball carrier only)', ax=ax3, bins=20)


# In[22]:


sns.pairplot(train_df.query('NflIdRusher==NflId')[['S','A','Dis','Yards','DefensePersonnel']], hue='DefensePersonnel')


# In[23]:


fig,ax = plt.subplots(1,1,figsize=(16,8))
train_df['DefensePersonnel'].value_counts().sort_values().head(15).plot(kind='barh', ax=ax)


# In[24]:


fig,ax = plt.subplots(1,1,figsize=(16,8))
train_df['OffensePersonnel'].value_counts().sort_values().head(30).plot(kind='barh', ax=ax)


# In[25]:


top_10_defenses = train_df.groupby('DefensePersonnel')['GameId'].count().sort_values(ascending=False).index[:10].tolist()
top_10_defenses


# In[26]:


train_play = train_df.groupby('PlayId').first()
train_top10_defense = train_play.loc[train_play['DefensePersonnel'].isin(top_10_defenses)]

fig, ax = plt.subplots(1,1,figsize=(16,8))
sns.violinplot(x='DefensePersonnel', y='Yards', data=train_top10_defense, ax=ax)
plt.ylim(-10,20)


# In[27]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_ylim(-10, 100)
ax.set_title('Yards vs Quarter')
sns.boxenplot(x='Quarter',
            y='Yards',
            data=train_df.sample(5000),
            ax=ax)
plt.show()


# In[28]:


train_df['DefendersInTheBox'].value_counts().sort_values().plot(kind='barh')


# In[29]:


fig, ax = plt.subplots(1,1,figsize=(16,4))
sns.boxenplot(x='DefendersInTheBox',
             y='Yards',
             data=train_df.query('DefendersInTheBox > 2')
             )


# In[30]:


fig, ax = plt.subplots(3,2,constrained_layout=True, figsize=(15,10))
ax_id1=0
ax_id2=0
for i in range(4,10):
    this_ax = ax[ax_id1][ax_id2]
    sns.distplot(train_df.query('DefendersInTheBox == @i')['Yards'],
                ax=this_ax, color=color_pal[ax_id1])
    this_ax.set_title(f'{i} Defenders in the box')
    ax_id2 +=1
    if ax_id2 == 2:
        ax_id2=0
        ax_id1+=1


# In[31]:


train_df.query('NflIdRusher == NflId').groupby('DisplayName')['Yards'].agg(['count','mean']).query('count > 100').sort_values(by='mean', ascending=True).tail(10)['mean'].plot(kind='barh', figsize=(15,5), title='Top 10 players with average yards', xlim=(0,6))
plt.show()


train_df.query('NflIdRusher == NflId').groupby('DisplayName')['Yards'].agg(['count','mean']).query('count > 100').sort_values(by='mean', ascending=True).head(10)['mean'].plot(kind='barh', figsize=(15,5), title='Bottom 10 players with average yards', xlim=(0,6))
plt.show()


# In[32]:


# Create the DL-LB combos
train_df['DL_LB'] = train_df['DefensePersonnel']     .str[:10]     .str.replace(' DL, ','-')     .str.replace(' LB','') # Clean up and convert to DL-LB combo
top_5_dl_lb_combos = train_df.groupby('DL_LB').count()['GameId']     .sort_values()     .tail(10).index.tolist()
ax = train_df.loc[train_df['DL_LB'].isin(top_5_dl_lb_combos)]     .groupby('DL_LB').mean()['Yards']     .sort_values(ascending=True)     .plot(kind='bar',
          title='Average Yards Top 5 Defensive DL-LB combos',
          figsize=(15, 5),
          color=color_pal[4])
# for p in ax.patches:
#     ax.annotate(str(round(p.get_height(), 2)),
#                 (p.get_x() * 1.005, p.get_height() * 1.015))

#bars = ax.bar(0.5, 5, width=0.5, align="center")
bars = [p for p in ax.patches]
value_format = "{:0.2f}"
label_bars(ax, bars, value_format, fontweight='bold')
plt.show()


# In[33]:


def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12*2, 6.33*2)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax

import math
def get_dx_dy(angle, dist):
    cartesianAngleRadians = (450-angle)*math.pi/180.0
    dx = dist * math.cos(cartesianAngleRadians)
    dy = dist * math.sin(cartesianAngleRadians)
    return dx, dy


# In[34]:


play_id = train_df.query("DL_LB == '3-4'")['PlayId'].reset_index(drop=True)[500]
fig, ax = create_football_field()
train_df.query("PlayId == @play_id and Team == 'away'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=200, legend='Away')
train_df.query("PlayId == @play_id and Team == 'home'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=200, legend='Home')
train_df.query("PlayId == @play_id and NflIdRusher == NflId")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=200, legend='Rusher')
rusher_row = train_df.query("PlayId == @play_id and NflIdRusher == NflId")
yards_covered = rusher_row["Yards"].values[0]

x = rusher_row["X"].values[0]
y = rusher_row["Y"].values[0]
rusher_dir = rusher_row["Dir"].values[0]
rusher_speed = rusher_row["S"].values[0]
dx, dy = get_dx_dy(rusher_dir, rusher_speed)
yards_gained = train_df.query("PlayId == @play_id")['Yards'].tolist()[0]
ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3)
plt.title(f'Example of a 3-4 Defense - run resulted in {yards_gained} yards gained', fontsize=20)
plt.legend()
plt.show()


# In[35]:


plt.figure(figsize=(12,10))
temp_df = train_df.query("NflIdRusher == NflId")
sns.boxplot(data=temp_df, y="PossessionTeam", x="Yards", showfliers=False, whis=3.0)
plt.ylabel('PossessionTeam', fontsize=12)
plt.xlabel('Yards (Target)', fontsize=12)
plt.title("Possession team Vs Yards (target)", fontsize=20)
plt.show()


# In[36]:


plt.figure(figsize=(16,12))
temp_df = train_df.query("NflIdRusher == NflId")
sns.catplot(data=temp_df, x="Quarter", y="Yards", kind="boxen")
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Yards (Target)', fontsize=12)
plt.title("Quarter Vs Yards (target)", fontsize=20)
plt.show()


# In[ ]:




