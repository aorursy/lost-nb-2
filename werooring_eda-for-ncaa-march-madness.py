#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import pandas as pd
import numpy as np
import re
import random
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('fivethirtyeight')
sns.set(font_scale=2.2)
sns.set(style="whitegrid")
base_color = sns.color_palette()[0]
second_color = sns.color_palette()[1]
third_color = sns.color_palette()[2]


# In[2]:


files = glob.glob('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents*.csv')


# In[3]:


# Load all csv file started with 'MEvents'
data_frames = [pd.read_csv(file) for file in files]

events = pd.concat(data_frames, axis=0, sort=False)
events.head()


# In[4]:


# Debugging: When it is true, load 50,000 rows randomly according to MEvents. if false, load all rows
# Reference: https://stackoverflow.com/questions/22258491/read-a-small-random-sample-from-a-big-csv-file-into-a-python-data-frame

DEBUG = False
if DEBUG:
    sample_size = 50000
else: 
    sample_size = None

def get_skiprows(file, sample_size):
    num_of_records = sum(1 for line in open(file))
    # the 0-indexed header will not be included in the skip list
    if DEBUG:
        skiprows = sorted(random.sample(range(1,num_of_records+1),
                                    num_of_records-sample_size))
    else:
        skiprows=None
    return skiprows


# In[5]:


PATH = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/'
dfs = {'MEvents': [], 'Misc': {}}
for root, dirs, files in os.walk(PATH):
    print(PATH)
    for file in files:
        path_and_file = os.path.join(root, file)
        print(path_and_file)
        if bool(re.search('MEvents', path_and_file)):
            skiprows = get_skiprows(path_and_file, sample_size)
            dfs['MEvents'].append(pd.read_csv(path_and_file, skiprows=skiprows))
            
        elif bool(re.search('.DS_Store', path_and_file)):
            pass
        else:
            file_name_start_index = path_and_file.rfind('/') + 1
            file_name_end_index = re.search('.csv', path_and_file).span()[0]
            if bool(re.search('MTeamSpellings', path_and_file)):
                dfs['Misc'][path_and_file[file_name_start_index:file_name_end_index]] = pd.read_csv(path_and_file, encoding='cp1252')
            else: 
                dfs['Misc'][path_and_file[file_name_start_index:file_name_end_index]] = pd.read_csv(path_and_file)                
            


# In[6]:


dfs['Misc'].keys()


# In[7]:


MTeams = dfs['Misc']['MTeams']
MTeams.head()


# In[8]:


MTeams.shape


# In[9]:


MTeams.sort_values('FirstD1Season', ascending=False).head()


# In[10]:


def plot_bar_chart(df, feature):
    plt.figure(figsize=(12, 5))
    counts = df[feature].value_counts()
    sns.barplot(x=counts.index, y=counts.values, color=base_color)


plot_bar_chart(MTeams, 'FirstD1Season')
plt.title('FirstD1Season Counts', fontsize=18); plt.xlabel('FirstD1Season', fontsize=15); plt.ylabel('Counts', fontsize=15)
plt.xticks(rotation=30);


# In[11]:


rate_of_FirstD1Season = round(MTeams['FirstD1Season'].value_counts().max() / MTeams.shape[0], 2)
rate_of_FirstD1Season = int(rate_of_FirstD1Season*100)
print(f'{rate_of_FirstD1Season}% of the teams got into Division I in 1985. Note that the teams that entered before 1985 are also recorded as 1985')


# In[12]:


plot_bar_chart(MTeams, 'LastD1Season')
plt.title('LastD1Season Counts', fontsize=18); plt.xlabel('LastD1Season', fontsize=15); plt.ylabel('Counts', fontsize=15)
plt.xticks(rotation=30);


# In[13]:


rate_of_LastD1Season = round(MTeams['LastD1Season'].value_counts().max() / MTeams.shape[0], 2)
rate_of_LastD1Season = int(rate_of_LastD1Season*100)
print(f'{rate_of_LastD1Season}% of the teams remain until now')


# In[14]:


MSeasons = dfs['Misc']['MSeasons']
MSeasons.head(5)


# In[15]:


MSeasons.shape


# In[16]:


MNCAATourneySeeds = dfs['Misc']['MNCAATourneySeeds']
MNCAATourneySeeds.head(5)


# In[17]:


MNCAATourneySeeds.shape


# In[18]:


MNCAATourneySeeds['Seed'].value_counts()


# In[19]:


MRegularSeasonCompactResults = dfs['Misc']['MRegularSeasonCompactResults']
MRegularSeasonCompactResults.head()


# In[20]:


MRegularSeasonCompactResults.shape


# In[21]:


MRegularSeasonCompactResults =     MRegularSeasonCompactResults.merge(MTeams[['TeamID', 'TeamName']],
                                      left_on='WTeamID',
                                      right_on='TeamID',
                                      validate='many_to_one').drop('TeamID', axis=1)
MRegularSeasonCompactResults.rename(columns={'TeamName': 'WTeamName'}, inplace=True)

MRegularSeasonCompactResults =     MRegularSeasonCompactResults.merge(MTeams[['TeamID', 'TeamName']],
                                      left_on='LTeamID',
                                      right_on='TeamID',
                                      validate='many_to_one').drop('TeamID', axis=1)
MRegularSeasonCompactResults.rename(columns={'TeamName': 'LTeamName'}, inplace=True)

MRegularSeasonCompactResults.head()


# In[22]:


MRegularSeasonCompactResults['ScoreDiff'] = MRegularSeasonCompactResults['WScore'] - MRegularSeasonCompactResults['LScore']


# In[23]:


MRegularSeasonCompactResults['ScoreDiff'].plot(kind='hist',
                                              bins=90,
                                              figsize=(15,5))
plt.title('Score Difference between WTeam and LTeam', fontsize=18);


# In[24]:


num_winning = MRegularSeasonCompactResults['WTeamName'].value_counts()
num_winning = num_winning.head(20)


# In[25]:


plt.figure(figsize=(12, 5))
counts = MRegularSeasonCompactResults['WTeamName'].value_counts().head(20)
sns.barplot(y=counts.index, x=counts.values, color=second_color, orient='h')

plt.title('Most Winning (pre-season) Teams', fontsize=18); plt.ylabel('Winning Teams', fontsize=15); plt.xlabel('Counts', fontsize=15)
plt.xlim(600, 920);


# In[26]:


plt.figure(figsize=(12, 5))
counts = MRegularSeasonCompactResults['WTeamName'].value_counts().tail(20)
sns.barplot(y=counts.index, x=counts.values, color=base_color, orient='h')

plt.title('Least Winning (pre-season) Teams', fontsize=18); plt.ylabel('Winning Teams', fontsize=15); plt.xlabel('Counts', fontsize=15)
plt.xlim(0, 150);


# In[27]:


plt.figure(figsize=(12, 5))
counts = MRegularSeasonCompactResults['LTeamName'].value_counts().head(20)
sns.barplot(y=counts.index, x=counts.values, color=third_color, orient='h')

plt.title('Most Losing (pre-season) Teams', fontsize=18); plt.ylabel('Losing Teams', fontsize=15); plt.xlabel('Counts', fontsize=15)
plt.xlim(600, 750);


# In[28]:


total_num_of_pre_game = MRegularSeasonCompactResults.shape[0]

MRegularSeasonCompactResults['WLoc'].value_counts() / total_num_of_pre_game


# In[29]:


MRegularSeasonCompactResults['NumOT'].value_counts()


# In[30]:


MRegularSeasonCompactResults.groupby(['Season'])['WScore'].mean().plot()
plt.title('Mean Scores of winning teams by season in regular plays', fontsize=18);


# In[31]:


# Concatenate MEvents2015, 2016, 2017, 2018, 2019 together
MEvents = pd.concat(dfs['MEvents'], axis=0, sort=False)
MEvents.head()


# In[32]:


MEvents.shape


# In[33]:


plot_bar_chart(MEvents, 'EventType')
plt.title('EventType Counts', fontsize=18); plt.xlabel('EventType', fontsize=15); plt.ylabel('Counts', fontsize=15)
plt.xticks(rotation=30);


# In[34]:


area_mapping = {0: np.nan,
                1: 'under basket',
                2: 'in the paint',
                3: 'inside right wing',
                4: 'inside right',
                5: 'inside center',
                6: 'inside left',
                7: 'inside left wing',
                8: 'outside right wing',
                9: 'outside right',
                10: 'outside center',
                11: 'outside left',
                12: 'outside left wing',
                13: 'backcourt'}

MEvents['Area_Name'] = MEvents['Area'].map(area_mapping)


# In[35]:


MEvents['counter'] = 1


# In[36]:


plot_bar_chart(MEvents, 'EventType')
plt.title('EventType Counts', fontsize=18); plt.xlabel('EventType', fontsize=15); plt.ylabel('Counts', fontsize=15)
plt.xticks(rotation=30);


# In[37]:


plt.figure(figsize=(12, 5))
counts = MEvents.groupby('Area_Name')['counter'].sum().sort_values(ascending=False)
sns.barplot(y=counts.index, x=counts.values, color=third_color, orient='h')

plt.title('Events Area Counts', fontsize=18); plt.ylabel('Area Name', fontsize=15); plt.xlabel('Counts', fontsize=15);


# In[38]:


fig, ax = plt.subplots(figsize=(15, 8))
for area, df in MEvents.loc[~MEvents['Area_Name'].isna()].groupby('Area_Name'):
    df.plot(x='X', y='Y', style='.', label=area, ax=ax, title='Visualizing Event Areas')
    ax.legend()
plt.legend(bbox_to_anchor=(1.03,1), loc="upper left")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100);


# In[39]:


MEvents['X_'] = (MEvents['X'] * (94/100))
MEvents['Y_'] = (MEvents['Y'] * (50/100))


# In[40]:


def create_ncaa_full_court(ax=None, three_line='mens', court_color='#edc993',
                           lw=3, lines_color='black', lines_alpha=0.5,
                           paint_fill='blue', paint_alpha=0.4):
    """
    Creates NCAA Basketball
    Dimensions are in feet (Court is 97x50 ft)
    Created by: Rob Mulla / https://github.com/RobMulla

    * Note that this function uses "feet" as the unit of measure.
    * NCAA Data is provided on a x range: 0, 100 and y-range 0 to 100
    * To plot X/Y positions first convert to feet like this:
    ```
    Events['X_'] = (Events['X'] * (94/100))
    Events['Y_'] = (Events['Y'] * (50/100))
    ```

    three_line: 'mens', 'womens' or 'both' defines 3 point line plotted
    court_color : (hex) Color of the court
    lw : line width
    lines_color : Color of the lines
    paint_fill : Color inside the paint
    paint_alpha : transparency of the "paint"
    """
    if ax is None:
        ax = plt.gca()

    # Create Pathes for Court Lines
    center_circle = Circle((94/2, 50/2), 6,
                           linewidth=lw, color=lines_color, lw=lw,
                           fill=False, alpha=lines_alpha)
#     inside_circle = Circle((94/2, 50/2), 2,
#                            linewidth=lw, color=lines_color, lw=lw,
#                            fill=False, alpha=lines_alpha)

    hoop_left = Circle((5.25, 50/2), 1.5 / 2,
                       linewidth=lw, color=lines_color, lw=lw,
                       fill=False, alpha=lines_alpha)
    hoop_right = Circle((94-5.25, 50/2), 1.5 / 2,
                        linewidth=lw, color=lines_color, lw=lw,
                        fill=False, alpha=lines_alpha)

    # Paint - 18 Feet 10 inches which converts to 18.833333 feet - gross!
    left_paint = Rectangle((0, (50/2)-6), 18.833333, 12,
                           fill=paint_fill, alpha=paint_alpha,
                           lw=lw, edgecolor=None)
    right_paint = Rectangle((94-18.83333, (50/2)-6), 18.833333,
                            12, fill=paint_fill, alpha=paint_alpha,
                            lw=lw, edgecolor=None)
    
    left_paint_boarder = Rectangle((0, (50/2)-6), 18.833333, 12,
                           fill=False, alpha=lines_alpha,
                           lw=lw, edgecolor=lines_color)
    right_paint_boarder = Rectangle((94-18.83333, (50/2)-6), 18.833333,
                            12, fill=False, alpha=lines_alpha,
                            lw=lw, edgecolor=lines_color)

    left_arc = Arc((18.833333, 50/2), 12, 12, theta1=-
                   90, theta2=90, color=lines_color, lw=lw,
                   alpha=lines_alpha)
    right_arc = Arc((94-18.833333, 50/2), 12, 12, theta1=90,
                    theta2=-90, color=lines_color, lw=lw,
                    alpha=lines_alpha)
    
    leftblock1 = Rectangle((7, (50/2)-6-0.666), 1, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    leftblock2 = Rectangle((7, (50/2)+6), 1, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(leftblock1)
    ax.add_patch(leftblock2)
    
    left_l1 = Rectangle((11, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    left_l2 = Rectangle((14, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    left_l3 = Rectangle((17, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(left_l1)
    ax.add_patch(left_l2)
    ax.add_patch(left_l3)
    left_l4 = Rectangle((11, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    left_l5 = Rectangle((14, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    left_l6 = Rectangle((17, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(left_l4)
    ax.add_patch(left_l5)
    ax.add_patch(left_l6)
    
    rightblock1 = Rectangle((94-7-1, (50/2)-6-0.666), 1, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    rightblock2 = Rectangle((94-7-1, (50/2)+6), 1, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(rightblock1)
    ax.add_patch(rightblock2)

    right_l1 = Rectangle((94-11, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    right_l2 = Rectangle((94-14, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    right_l3 = Rectangle((94-17, (50/2)-6-0.666), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(right_l1)
    ax.add_patch(right_l2)
    ax.add_patch(right_l3)
    right_l4 = Rectangle((94-11, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    right_l5 = Rectangle((94-14, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    right_l6 = Rectangle((94-17, (50/2)+6), 0.166, 0.666,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(right_l4)
    ax.add_patch(right_l5)
    ax.add_patch(right_l6)
    
    # 3 Point Line
    if (three_line == 'mens') | (three_line == 'both'):
        # 22' 1.75" distance to center of hoop
        three_pt_left = Arc((6.25, 50/2), 44.291, 44.291, theta1=-78,
                            theta2=78, color=lines_color, lw=lw,
                            alpha=lines_alpha)
        three_pt_right = Arc((94-6.25, 50/2), 44.291, 44.291,
                             theta1=180-78, theta2=180+78,
                             color=lines_color, lw=lw, alpha=lines_alpha)

        # 4.25 feet max to sideline for mens
        ax.plot((0, 11.25), (3.34, 3.34),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((0, 11.25), (50-3.34, 50-3.34),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((94-11.25, 94), (3.34, 3.34),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((94-11.25, 94), (50-3.34, 50-3.34),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.add_patch(three_pt_left)
        ax.add_patch(three_pt_right)

    if (three_line == 'womens') | (three_line == 'both'):
        # womens 3
        three_pt_left_w = Arc((6.25, 50/2), 20.75 * 2, 20.75 * 2, theta1=-85,
                              theta2=85, color=lines_color, lw=lw, alpha=lines_alpha)
        three_pt_right_w = Arc((94-6.25, 50/2), 20.75 * 2, 20.75 * 2,
                               theta1=180-85, theta2=180+85,
                               color=lines_color, lw=lw, alpha=lines_alpha)

        # 4.25 inches max to sideline for mens
        ax.plot((0, 8.3), (4.25, 4.25), color=lines_color,
                lw=lw, alpha=lines_alpha)
        ax.plot((0, 8.3), (50-4.25, 50-4.25),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((94-8.3, 94), (4.25, 4.25),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((94-8.3, 94), (50-4.25, 50-4.25),
                color=lines_color, lw=lw, alpha=lines_alpha)

        ax.add_patch(three_pt_left_w)
        ax.add_patch(three_pt_right_w)

    # Add Patches
    ax.add_patch(left_paint)
    ax.add_patch(left_paint_boarder)
    ax.add_patch(right_paint)
    ax.add_patch(right_paint_boarder)
    ax.add_patch(center_circle)
#     ax.add_patch(inside_circle)
    ax.add_patch(hoop_left)
    ax.add_patch(hoop_right)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    # Restricted Area Marker
    restricted_left = Arc((6.25, 50/2), 8, 8, theta1=-90,
                        theta2=90, color=lines_color, lw=lw,
                        alpha=lines_alpha)
    restricted_right = Arc((94-6.25, 50/2), 8, 8,
                         theta1=180-90, theta2=180+90,
                         color=lines_color, lw=lw, alpha=lines_alpha)
    ax.add_patch(restricted_left)
    ax.add_patch(restricted_right)
    
    # Backboards
    ax.plot((4, 4), ((50/2) - 3, (50/2) + 3),
            color=lines_color, lw=lw*1.5, alpha=lines_alpha)
    ax.plot((94-4, 94-4), ((50/2) - 3, (50/2) + 3),
            color=lines_color, lw=lw*1.5, alpha=lines_alpha)
    ax.plot((4, 4.6), (50/2, 50/2), color=lines_color,
            lw=lw, alpha=lines_alpha)
    ax.plot((94-4, 94-4.6), (50/2, 50/2),
            color=lines_color, lw=lw, alpha=lines_alpha)

    # Half Court Line
    ax.axvline(94/2, color=lines_color, lw=lw, alpha=lines_alpha)

    # Boarder
    boarder = Rectangle((0.3,0.3), 94-0.6, 50-0.6, fill=False, lw=3, color='black', alpha=lines_alpha)
    ax.add_patch(boarder)
    
    # Plot Limit
    ax.set_xlim(0, 94)
    ax.set_ylim(0, 50)
    ax.set_facecolor(court_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    return ax


# In[41]:


fig, ax = plt.subplots(figsize=(15, 8.5))
create_ncaa_full_court(ax, three_line='both', paint_alpha=0.2);


# In[42]:


fig, ax = plt.subplots(figsize=(15, 7.5))
marker_size = 10
ax = create_ncaa_full_court(ax, paint_alpha=0.15)
MEvents.loc[MEvents['EventType'] == 'turnover'].plot(x='X_',
                                                    y='Y_',
                                                    style='X',
                                                    c='red',
                                                    alpha=0.3,
                                                    markersize=marker_size,
                                                    ax=ax)
ax.set_title('Turnover Locations', fontsize=18)
ax.set_xlabel('')
ax.get_legend().remove();


# In[43]:


num_of_total_rows = MEvents.loc[MEvents['EventType'] == 'turnover'].shape[0]
num_of_total_rows


# In[44]:


turnover = MEvents.loc[MEvents['EventType'] == 'turnover']
num_of_not_zero_X_rows = turnover.loc[turnover['X_'] != 0].shape[0]
num_of_not_zero_X_rows


# In[45]:


print(f'총 {num_of_total_rows}개의 rows가 있지만 {num_of_not_zero_X_rows}개만이 0이 아닌 X_를 가지고 있어, plot된 점이 적습니다.')


# In[46]:


matplotlib.rcParams['agg.path.chunksize']=100000

if DEBUG:
    alpha = 0.4
    style = 'X'
else:
    alpha = 0.01
    style = 'o'

fig, ax = plt.subplots(figsize=(10, 5.5))
ax = create_ncaa_full_court(ax, paint_alpha=0.2)
MEvents.loc[MEvents['EventType'] == 'made3'].plot(x='X_',
                                                 y='Y_',
                                                 marker=style,
                                                 color='blue',
                                                 alpha=alpha,
                                                 ax=ax)
ax.set_title('3 Pointers Made', fontsize=18)
ax.set_xlabel('')
ax.get_legend().remove();


# In[47]:


fig2, ax2 = plt.subplots(figsize=(10, 5.5))
ax2 = create_ncaa_full_court(ax2, paint_alpha=0.2)
MEvents.loc[MEvents['EventType']=='miss3'].plot(x='X_',
                                                y='Y_',
                                                marker=style,
                                                color='red',
                                                alpha=alpha,
                                                ax=ax2)
ax2.set_title('3 Pointers Missed', fontsize=18)
ax2.set_xlabel('')
ax2.get_legend().remove();


# In[48]:


fig3, ax3 = plt.subplots(figsize=(10, 5.5))
ax3 = create_ncaa_full_court(ax3, paint_alpha=0.2)
MEvents.loc[MEvents['EventType'] == 'made2'].plot(x='X_',
                                                 y='Y_',
                                                 marker=style,
                                                 color='blue',
                                                 alpha=alpha,
                                                 ax=ax3)
ax3.set_title('2 Pointers Made', fontsize=18)
ax3.set_xlabel('')
ax3.get_legend().remove()


# In[49]:


fig, ax = plt.subplots(figsize=(10, 5.5))
ax = create_ncaa_full_court(ax, paint_alpha=0.2)
MEvents.loc[MEvents['EventType'] == 'miss2'].plot(x='X_',
                                                 y='Y_',
                                                 marker=style,
                                                 color='red',
                                                 alpha=alpha,
                                                 ax=ax)
ax.set_title('2 Pointers Missed', fontsize=18)
ax.set_xlabel('')
ax.get_legend().remove();


# In[50]:


# Merge events and MPlayers
MEvents = MEvents.merge(dfs['Misc']['MPlayers'],
             how='left',
             left_on = ['EventTeamID', 'EventPlayerID'],
             right_on = ['TeamID', 'PlayerID'])


# In[51]:


MEvents.drop(['PlayerID', 'TeamID'], axis=1, inplace=True)
MEvents.head()


# In[52]:


MEvents['FullName'] = MEvents['FirstName'] + ' ' + MEvents['LastName']


# In[53]:


marker_size = 15

first_name = 'Mamadi'
last_name = 'Diakite'

fig, ax = plt.subplots(figsize=(15, 8))
ax = create_ncaa_full_court(ax, paint_alpha=0.2)
MEvents.query('FirstName == @first_name and LastName == @last_name and EventType == "made2"')     .plot(x='X_', y='Y_', style='o', label='Made 2',
         markersize=marker_size, ax=ax);

MEvents.query('FirstName == @first_name and LastName == @last_name and EventType == "miss2"')     .plot(x='X_', y='Y_', style='X', label='Missed 2',
         color='red', markersize=marker_size, ax=ax);
plt.legend(loc='lower center')
plt.title('Mamadi Diakite Shots');


# In[54]:


marker_size = 15

first_name = 'Kyle'
last_name = 'Guy'

# first_name = 'Zion'
# last_name = 'Williamson'

fig, ax = plt.subplots(figsize=(15, 8))
ax = create_ncaa_full_court(ax, paint_alpha=0.2)
MEvents.query('FirstName == @first_name and LastName == @last_name and EventType == "made2"')     .plot(x='X_', y='Y_', style='o', label='Made 2',
         markersize=marker_size, ax=ax);

MEvents.query('FirstName == @first_name and LastName == @last_name and EventType == "miss2"')     .plot(x='X_', y='Y_', style='X', label='Missed 2',
         color='red', markersize=marker_size, ax=ax);
plt.legend(loc='lower center')
plt.title('Kyle Guy Shots');


# In[55]:


marker_size = 15

# Best Shooting Guard
first_name = 'Aaron'
last_name = 'Henry'

fig, ax = plt.subplots(figsize=(15, 8))
ax = create_ncaa_full_court(ax, paint_alpha=0.2)
MEvents.query('FirstName == @first_name and LastName == @last_name and EventType == "made3"')     .plot(x='X_', y='Y_', style='o', label='Made 3',
         markersize=marker_size, ax=ax);

MEvents.query('FirstName == @first_name and LastName == @last_name and EventType == "miss3"')     .plot(x='X_', y='Y_', style='X', label='Missed 3',
         color='red', markersize=marker_size, ax=ax);
plt.legend(loc='lower center')
plt.title('Kyle Guy Shots');


# In[56]:


N_bins=100
shot_events = MEvents.loc[MEvents['EventType'].isin(['miss3','made3','miss2','made2']) & (MEvents['X_'] != 0)]
fig, ax = plt.subplots(figsize=(15, 7))
ax = create_ncaa_full_court(ax=ax,
                            paint_alpha=0.0,
                            court_color='black',
                            lines_color='white')
plt.hist2d(shot_events['X_'].values, 
           shot_events['Y_'].values,
           bins=N_bins, norm=matplotlib.colors.LogNorm(),
           cmap='plasma')

# Plot a colorbar with label.
cb = plt.colorbar()
cb.set_label('Numer of Shots')
ax.set_title('Shot Heatmap', fontsize=18);


# In[ ]:




