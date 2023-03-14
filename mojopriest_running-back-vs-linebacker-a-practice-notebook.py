#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<style type="text/css">\n\ndiv.h2 {\n\n    background-color: #159957;\n    background-image: linear-gradient(120deg, #155799, #159957);\n    text-align: left;\n    color: white;              \n    padding:9px;\n    padding-right: 100px; \n    font-size: 20px; \n    max-width: 1500px; \n    margin: auto; \n    margin-top: 40px; \n\n}\n\n                                                                         \nbody {\n\n  font-size: 12px;\n\n}    \n                                     \n\ndiv.h3 {\n\n    color: #159957; \n    font-size: 18px; \n    margin-top: 20px; \n    margin-bottom:4px;\n\n}\n                                      \n\ndiv.h4 {\n\n    color: #159957;\n    font-size: 15px; \n    margin-top: 20px; \n    margin-bottom: 8px;\n\n}\n\n   \nspan.note {\n\n    font-size: 5; \n    color: gray; \n    font-style: italic;\n\n}\n\n  \nhr {\n\n    display: block; \n    color: gray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n\n}\n                                 \n\nhr.light {\n\n    display: block;\n    color: lightgray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n\n}   \n\n   \n                                      \n                                      \n                        \n                                      \n                                      \n                                      \n                                      \n                                                          \n\ntable.dataframe th \n\n{\n\n    border: 1px darkgray solid;\n    color: black;\n    align: left;\n    background-color: white;\n\n}\n\n    \n\n                                      \n\ntable.dataframe td \n                                      \n{\n\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 12px;\n    text-align: center;\n\n} \n                                   \n\ntable.rules th \n\n{\n\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    align: left;\n\n}\n                                   \n\ntable.rules td \n\n{\n\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n\n} \n\n   \ntable.rules tr.best\n\n{\n\n    color: green;\n\n}    \n\n    \n.output { \n\n    align-items: left; \n\n}\n\n        \n.output_png {\n\n    display: table-cell;\n\n    text-align: left;\n\n    margin:auto;\n\n}                                          \n\n                                \n</style>')


# In[2]:


# original css stylesheet: Kaggle, member: TexasTom

#import modules
import pandas as pd
import numpy as np

#load dataset
df = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory = False)

# set maximum number of columns in diplay
pd.set_option('display.max_columns', 36)

#drop possible NaN values
df.dropna(inplace = True)

# switch Position value HB (half back i.e. running back) to RB
# group different linebacker positions (ILB, MLB, OLB, LB) under same label LB 
df["Position"]= df["Position"].replace("HB", "RB")
df["Position"]= df["Position"].replace("ILB", "LB")
df["Position"]= df["Position"].replace("MLB", "LB")
df["Position"]= df["Position"].replace("OLB", "LB")

# select and drop original columns relevant to task at hand
cols = ['Orientation', 'Dir', 'Dis', 'DisplayName', 'JerseyNumber', 'Season', 'Team', 'PossessionTeam', 'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 
       'PlayDirection', 'OffenseFormation', 'PlayerBirthDate', 'PlayerCollegeName', 'TimeHandoff', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'Stadium', 'Location', 'StadiumType', 'Turf',
       'GameWeather', 'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']
df = df.drop(cols, axis=1)

# select only rows with RB or LB as Position value
df = df[df['Position'].isin(['RB', 'LB']) ]

# create a new column storing as string whether the player is RB or LB
df.loc[:,'RbLb'] = df['Position']

# arrange dataframe by column value, in this case PlayId 
# method courtesy of StackOverFlow, member: yogitha jaya reddy gari
df = df.sort_values(['PlayId'],ascending=False).groupby('PlayId',as_index = False).apply(lambda x: x.reset_index(drop = True))
df.reset_index().drop(['level_0','level_1'],axis = 1)

df.head(10)


# In[3]:


# get numeric values for Position column
df = pd.get_dummies(df, columns=['Position'])

# create new columns based on existing data
df['Is3Wr'] = df['OffensePersonnel'].str.contains('3 WR')
df['Is3Wr'] = df['Is3Wr'].map({True: 1, False: 0})

df['Is3Lb'] = df['DefensePersonnel'].str.contains('3 LB')
df['Is3Lb'] = df['Is3Lb'].map({True: 1, False: 0})

df['Is4Lb'] = df['DefensePersonnel'].str.contains('4 LB')
df['Is4Lb'] = df['Is4Lb'].map({True: 1, False: 0})

df['Is4Lb'] = df['DefensePersonnel'].str.contains('4 LB')
df['Is4Lb'] = df['Is4Lb'].map({True: 1, False: 0})

df.loc[:,'X_lb'] = df['X']
df.loc[:,'Y_lb'] = df['Y']
df.loc[:,'X_rb'] = df['X']
df.loc[:,'Y_rb'] = df['Y']

# create separate x/y coordinate values for running backs and linebackers
# these values are taken from original dataset X and Y columns
df['X_lb'] = df['Position_LB'].apply(lambda x: None if x==1 else 0)
df['X_lb'] = df['X_lb'].fillna(df['X'])

df['Y_lb'] = df['Position_LB'].apply(lambda x: None if x==1 else 0)
df['Y_lb'] = df['Y_lb'].fillna(df['Y'])

df['X_rb'] = df['Position_LB'].apply(lambda x: None if x==0 else 0)
df['X_rb'] = df['X_rb'].fillna(df['X'])

df['Y_rb'] = df['Position_LB'].apply(lambda x: None if x==0 else 0)
df['Y_rb'] = df['Y_rb'].fillna(df['Y'])

# replace 0 values with NaN
df.X_lb = df.X_lb.replace(0, np.nan)
df.Y_lb = df.Y_lb.replace(0, np.nan)
df.X_rb = df.X_rb.replace(0, np.nan)
df.Y_rb = df.Y_rb.replace(0, np.nan)

# sort dataframe index and create multi index consisting of Play and Players
df.sort_index(inplace = True) 
df.index.names = ['Play','Players']

# df.head()


# In[4]:


# the average positional X coordinates by Play
a1 = df.groupby('Play')['X_lb'].mean()
b1 = df.groupby('Play')['X_rb'].mean()

# subttract RB average X coordinate values from LB average X values
c1 = (b1 - a1)

# make sure these values are absolute i.e. positive
c1 = np.absolute(c1)
# calculate average
c1 = c1.mean()

# create new temporary column xtr1
# this value is the existing running back X position minus the calculated average 
df['xtr1'] = df['X_rb'] - c1

# on rows where there are no running back X coordinate values, use values stored in the new column
df['X_lb'] = df['X_lb'].fillna(df.xtr1)

## df.head(20)


# In[5]:


# repeat the process above on Y coordinates for running backs
a2 = df.groupby('Play')['Y_lb'].mean()
b2 = df.groupby('Play')['Y_rb'].mean()

c2 = (b2 - a2)
c2 = np.absolute(c2)
c2 = c2.mean()

df['xtr2'] = df['Y_rb'] - c2
df['Y_lb'] = df['Y_lb'].fillna(df.xtr2)

# df.head(20)


# In[6]:


# fill NaN values based on specific multi index. Original code: StackOverFlow, user: piRSquared
df = df.groupby(level='Play').bfill()

# repeat filling missing values with average values
a3 = df.groupby('Play')['X_rb'].mean()
b3 = df.groupby('Play')['X_lb'].mean()

c3 = (b3 - a3)
c3 = np.absolute(c3)
c3 = c3.mean()

df['xtr3'] = df['X_lb'] + c3
df['X_rb'] = df['X_rb'].fillna(df.xtr3)


# repeat filling missing values with average values
a4 = df.groupby('Play')['Y_rb'].mean()
b4 = df.groupby('Play')['Y_lb'].mean()

c4 = (b4 - a4)
c4 = np.absolute(c4)
c4 = c4.mean()

df['xtr4'] = df['Y_lb'] + c4
df['Y_rb'] = df['Y_rb'].fillna(df.xtr4)

# drop unnecessary coordinate columns
xtr_cols = ['X', 'Y', 'xtr1', 'xtr2', 'xtr3', 'xtr4']
df = df.drop(xtr_cols, axis=1)

# df.head(20)


# In[7]:


# import module
import math 

# x and y coordinates to lists
a = df['X_lb'].values.tolist()
b = df['Y_lb'].values.tolist()
c = df['X_rb'].values.tolist()
d = df['Y_rb'].values.tolist()      

# empty list for Euclidean distance
MyList = []

# function to calculate Euclidean distance for LB and RB x,y  values in lists
def distance(x1, y1, x2, y2): 
                    result = [math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) * 1.0) for (x1, y1, x2, y2) in zip(a,b,c,d)] 
                    MyList.append(result)
            
# execute function on list values            
distance (a,b,c,d)

# flatten results list so that it fits the dataframe
MyList = np.array(MyList).flatten()

# round MyList to two digits to fit the dataframe format
MyList = np.round(MyList, 2)

#create new column 'euc' for Euclidean distance
df['euc'] = np.array(MyList)

# df.head(25)


# In[8]:


# create a new column acc_lb for acceleration by LB position
df.loc[:,'acc_lb'] = df['A']

# insert the acceleration value from column A, otherwise 0
df['acc_lb'] = df['Position_LB'].apply(lambda x: None if x==1 else 0)
df['acc_lb'] = df['acc_lb'].fillna(df['A'])

# replace 0 values in column with NaN
df.acc_lb = df.acc_lb.replace(0, np.nan)

# fill NaN values based on multi index Play. Original code: StackOverFlow, user: piRSquared
df = df.groupby(level='Play').bfill()


# there are still NaN values left in acc_lb column
# next the NaN values are replaced with the average LB acceleration


# create average acceleration for LB
acc_1 = df['acc_lb'].mean()

# round acc_1 to two digits
acc_1 = np.round(acc_1, 2)

# replace acc_lb NaN values with average LB acceleration (acc_1) 
df['acc_lb'] = df['acc_lb'].fillna(acc_1)


# next the process above is repeated on RB position


# create a new column acc_rb for acceleration by RB position
df.loc[:,'acc_rb'] = df['A']

# insert the acceleration value from column A, otherwise 0
df['acc_rb'] = df['Position_LB'].apply(lambda x: None if x==0 else 0)
df['acc_rb'] = df['acc_rb'].fillna(df['A'])

# replace 0 values in column with NaN
df.acc_rb = df.acc_rb.replace(0, np.nan)

# fill NaN values based on multi index Play
df = df.groupby(level='Play').bfill()

# for remaining NaN values, create average acceleration for RB
acc_2 = df['acc_rb'].mean()

# round acc_2 to two digits
acc_2 = np.round(acc_2, 2)

# replace acc_rb NaN values with average RB acceleration (acc_2) 
df['acc_rb'] = df['acc_rb'].fillna(acc_2)


# drop original acceleration column A
acc_col = ['A']
df = df.drop(acc_col, axis=1)


# df.head(20)


# In[9]:


# RB and LB acceleration values to two lists
ac1 = df['acc_lb'].values.tolist()
ac2 = df['acc_rb'].values.tolist()

# empty list for relative acceleration
RelAcc = []

# function to calculate relative acceleration using two lists of values
def relative_acc(x1, x2): 
                    result =  [(x2 / x1) for (x1, x2) in zip(ac1,ac2)] 
                    RelAcc.append(result)   
        
# execute function on list values            
relative_acc (ac1,ac2)

# flatten results list so that it fits the dataframe
RelAcc = np.array(RelAcc).flatten()

# round RelAcc to two digits to fit the dataframe format
RelAcc = np.round(RelAcc, 2)

#create new column 'RelAcc' for relative acceleration value
df['RelAcc'] = np.array(RelAcc)
        
# df.head(20)        


# In[10]:


# create a new column spd_lb for speed by LB position
df.loc[:,'spd_lb'] = df['S']

# insert the speed value from column S, otherwise 0
df['spd_lb'] = df['Position_LB'].apply(lambda x: None if x==1 else 0)
df['spd_lb'] = df['spd_lb'].fillna(df['S'])

# replace 0 values in column with NaN
df.spd_lb = df.spd_lb.replace(0, np.nan)

# fill NaN values based on multi index Play. Original code: StackOverFlow, user: piRSquared
df = df.groupby(level='Play').bfill()

# create average speed for LB
spd_1 = df['spd_lb'].mean()

# round spd_1 to two digits
spd_1 = np.round(spd_1, 2)

# replace spd_lb NaN values with average LB speed (spd_1) 
df['spd_lb'] = df['spd_lb'].fillna(spd_1)


# the process above is repeated on RB position


# create a new column spd_rb for speed by RB position
df.loc[:,'spd_rb'] = df['S']

# insert the speed value from column S, otherwise 0
df['spd_rb'] = df['Position_LB'].apply(lambda x: None if x==0 else 0)
df['spd_rb'] = df['spd_rb'].fillna(df['S'])

# replace 0 values in column with NaN
df.spd_rb = df.spd_rb.replace(0, np.nan)

# fill NaN values based on multi index Play
df = df.groupby(level='Play').bfill()

# for remaining NaN values, create average speed for RB
spd_2 = df['spd_rb'].mean()

# round spd_2 to two digits
spd_2 = np.round(spd_2, 2)

# replace spd_rb NaN values with average RB speed (spd_2) 
df['spd_rb'] = df['spd_rb'].fillna(spd_2)



# RB and LB speed values to two lists
sp1 = df['spd_lb'].values.tolist()
sp2 = df['spd_rb'].values.tolist()

# empty list for relative speed
RelSpd = []

# function to calculate relative speed using two lists of values
def relative_spd (x1, x2): 
                    result =  [(x2 / x1) for (x1, x2) in zip(sp1,sp2)] 
                    RelSpd.append(result)   
        
# execute function on list values            
relative_spd (sp1,sp2)

# flatten results list so that it fits the dataframe
RelSpd = np.array(RelSpd).flatten()

# round RelSpd to two digits to fit the dataframe format
RelSpd = np.round(RelSpd, 2)

#create new column RelSpd for relative speed value
df['RelSpd'] = np.array(RelSpd)

# create a new column PlayYards with Yards column values
# this is not necessary but it easily relocates the column
df.loc[:,'PlayYards'] = df['Yards']


# drop original speed column S and Yards 
drp_cols = ['S', 'Yards']
df = df.drop(drp_cols, axis=1)

# df.head(20)


# In[11]:


# the average yards gained in a play in the dataset is 4.18
# create a new column Yds4_18
df.loc[:,'Yds4_18'] = df['PlayYards']

# set new column value 0 if PlayYards are equal or less than 4.18, 1 if more
f = lambda x: 0 if x <= 4.18 else 1
df['Yds4_18'] = df['Yds4_18'].map(f)

#df.head(20)


# In[12]:


# import modules
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# set plot size and font
sns.set(rc={'figure.figsize':(9.7,8.27)})
sns.set(font='sans-serif', palette='colorblind')

# set plot parameters
plot = sns.countplot(x = 'RbLb',
              data = df,
              order = df['RbLb'].value_counts().index)

# set plot title etc.
plot.axes.set_title('Total count of dataset rows divided by position',fontsize=24)
plot.set_xlabel("Position",fontsize=18)
plot.set_ylabel("Total count of rows",fontsize=18)
plot.tick_params(labelsize=14)

# show plot
plt.show()


# In[13]:


# set plot size and font
sns.set(rc={'figure.figsize':(9.7,8.27)})
sns.set(font='sans-serif', palette='colorblind')

# set plot parameters
plot = sns.countplot(x = 'Yds4_18',
              data = df,
              hue = 'Is3Wr',
              order = df['Yds4_18'].value_counts().index)

# set plot title etc.
plot.axes.set_title('4.18 yards threshold divided by 3WR on field',fontsize=24)
plot.set_xlabel("0 = 4.18 yards or less, 1 = more",fontsize=18)
plot.set_ylabel("Total count of rows",fontsize=18)
plot.tick_params(labelsize=14)
plot.legend (loc=1, fontsize = 16, fancybox=True, framealpha=1, shadow=True, borderpad=1, title = '0 = not 3WR, 1 = 3WR')

# show plot
plt.show()


# In[14]:


# set plot parameters
sns.set(font='sans-serif', palette='colorblind', font_scale=1.5) 
sns.lineplot(y='PlayYards', x='acc_rb', data=df, hue='Yds4_18', legend = 'full')


# In[15]:


# set plot parameters
sns.set(font='sans-serif', palette='colorblind', font_scale=1.5) 
sns.lineplot(y='PlayYards', x='acc_lb', data=df, hue='Yds4_18', legend = 'full')


# In[16]:


# define plot
sns.scatterplot(x = "RelAcc", y = "PlayYards", data = df, color = 'lime')

# set plot title etc.
plt.xlabel('Relative acceleration')
plt.ylabel('PlayYards')
plt.title('Relative acceleration and yards gained in play')

# show plot
plt.show()


# In[17]:


# create a new column RelAcc6_177
df.loc[:,'RelAcc6_77'] = df['RelAcc']

# set new column value 0 if RelAcc is equal or less than 6.77, 1 if more
f = lambda x: 0 if x <= 6.77 else 1
df['RelAcc6_77'] = df['RelAcc6_77'].map(f)


# In[18]:


# create variable acc_count with count of different values in column RelAll6_77
acc_count = df['RelAcc6_77'].value_counts()

# print variable
print (acc_count)


# In[19]:


# set plot parameters
sns.set(font='sans-serif', palette='colorblind', font_scale=1.5) 
sns.lineplot(y='PlayYards', x='RelAcc', data=df, hue='RelAcc6_77', legend = 'full')


# In[20]:


# store relacc_mean
relacc_mean = df['RelAcc'].mean()

# round relacc_mean to two digits
relacc_mean = np.round(relacc_mean, 2)

# create new column RelAcc_2 where value is 0 if RelAcc is greater or equal than 6.77
df['RelAcc_2'] = df['RelAcc'].apply(lambda x: None if x <= 6.77 else 0)

# get other column values from RelAcc
df['RelAcc_2'] = df['RelAcc_2'].fillna(df['RelAcc']) 

# replace 0 values with Nan
df.RelAcc_2 = df.RelAcc_2.replace(0, np.nan)

# replace Nan values with average RelAcc value stored in relacc_mean
df['RelAcc_2'] = df['RelAcc_2'].fillna(relacc_mean)

# drop unnecessary columns for relative acceleration
relacc_cols = ['RelAcc', 'RelAcc6_77']
df = df.drop(relacc_cols, axis=1)

#df.head(20)


# In[21]:


# import module
import plotly.express as px

# set plot parameters
fig = px.histogram(df, x="RelAcc_2", nbins = 100, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "Relative acceleration datapoints divided by percentage",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# show plot
fig.show()


# In[22]:


# set plot parameters
sns.set(font='sans-serif', palette='colorblind', font_scale=1.5) 
sns.lineplot(y='PlayYards', x='RelAcc_2', data=df, hue='Yds4_18')


# In[23]:


# create a new column RelAcc2_32
df.loc[:,'RelAcc2_32'] = df['RelAcc_2']

# set new column value 0 if RelAcc_2 is equal or less than 2.32, 1 if more
f = lambda x: 0 if x <= 2.32 else 1
df['RelAcc2_32'] = df['RelAcc2_32'].map(f)

# define plot size, color etc.
sns.set(rc={'figure.figsize':(9.7,7.27)})
sns.set(font='sans-serif', palette='colorblind')

# set plot parameters
plot = sns.countplot(x = 'Yds4_18',
              data = df,
              hue = 'RelAcc2_32',
              order = df['Yds4_18'].value_counts().index)

# set plot title etc.
plot.axes.set_title('4.18 yards threshold divided by relative acceleration (2.32)',fontsize=18)
plot.set_xlabel("0 = 4.18 yards run or less, 1 = more",fontsize=18)
plot.set_ylabel("Total count of rows",fontsize=18)
plot.tick_params(labelsize=14)
plot.legend (loc=1, fontsize = 16, fancybox=True, framealpha=1, shadow=True, borderpad=1, title = '0 = 2.32 or less, 1 = more')

# show plot
plt.show()


# In[24]:


# import module
import plotly.express as px

# set plot parameters
fig = px.histogram(df, x="euc", nbins = 100, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "Euclidean distance datapoints divided by percentage",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# show plot
fig.show()


# In[25]:


# create a new column euc_12
df.loc[:,'euc_12'] = df['euc']

# set new column value 0 if euc is equal or less than 12.0, 1 if more
f = lambda x: 0 if x <= 12.0 else 1
df['euc_12'] = df['euc_12'].map(f)

# print out the number of 0 and 1 values in the new column
euc_count = df['euc_12'].value_counts()
print (euc_count)


# In[26]:


# store euc_mean
euc_mean = df['euc'].mean()

# round euc_mean to two digits
euc_mean = np.round(euc_mean, 2)

# create new column euc_2 where value is 0 if euc is greater or equal than 12.0
df['euc_2'] = df['euc'].apply(lambda x: None if x <= 12.0 else 0)

# get other column values from euc
df['euc_2'] = df['euc_2'].fillna(df['euc']) 

# replace 0 values with Nan
df.euc_2 = df.euc_2.replace(0, np.nan)

# replace Nan values with average euc value stored in euc_mean
df['euc_2'] = df['euc_2'].fillna(euc_mean)

# drop previous columns for Euclidean distance as well as RelAcc2_32
euc_cols = ['euc', 'euc_12', 'RelAcc2_32']
df = df.drop(euc_cols, axis=1)

# df.head(10)


# In[27]:


# set histogram parameters
fig = px.histogram(df, x="euc_2", nbins = 100, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "euc_2 column datapoints divided by percentage",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# show histogram
fig.show()


# In[28]:


# create a new column euc6_46
df.loc[:,'euc6_46'] = df['euc_2']

# set new column value 0 if euc is equal or less than 6.46, 1 if more
f = lambda x: 0 if x <= 6.46 else 1
df['euc6_46'] = df['euc6_46'].map(f)

# set plot size, color etc.
sns.set(rc={'figure.figsize':(9.7,7.27)})
sns.set(font='sans-serif', palette='colorblind')

# set plot parameters
plot = sns.countplot(x = 'Yds4_18',
              data = df,
              hue = 'euc6_46',
              order = df['Yds4_18'].value_counts().index)

# set plot title etc.
plot.axes.set_title('4.18 yards threshold divided by Euclidean distance (6.46)',fontsize=18)
plot.set_xlabel("0 = 4.18 yards run or less, 1 = more",fontsize=18)
plot.set_ylabel("Total count of rows",fontsize=18)
plot.tick_params(labelsize=14)
plot.legend (loc=1, fontsize = 16, fancybox=True, framealpha=1, shadow=True, borderpad=1, title = '0 = 6.46 or less, 1 = more')

# show plot
plt.show()


# In[29]:


# set histogram paramters
fig = px.histogram(df, x="RelSpd", nbins = 200, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "Relative speed datapoints divided by percentage",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# show histogram
fig.show()


# In[30]:


# create a new column spd_12
df.loc[:,'RelSpd_12'] = df['RelSpd']

# set new column value 0 if RelSpd is equal or less than 12.0, 1 if more
f = lambda x: 0 if x <= 12.0 else 1
df['RelSpd_12'] = df['RelSpd_12'].map(f)

# print out the number of 0 and 1 values in the new column
RelSpd_count = df['RelSpd_12'].value_counts()
print (RelSpd_count)


# In[31]:


# store RelSpd_mean
RelSpd_mean = df['RelSpd'].mean()

# round euc_mean to two digits
RelSpd_mean = np.round(RelSpd_mean, 2)

# create new column RelSpd_2 where value is 0 if RelSpd is greater or equal than 12
df['RelSpd_2'] = df['RelSpd'].apply(lambda x: None if x <= 12 else 0)

# get other column values from RelSpd
df['RelSpd_2'] = df['RelSpd_2'].fillna(df['RelSpd']) 

# replace 0 values with Nan
df.RelSpd_2 = df.RelSpd_2.replace(0, np.nan)

# replace Nan values with average RelSpd value stored in relacc_mean
df['RelSpd_2'] = df['RelSpd_2'].fillna(RelSpd_mean)

# drop previous columns for RelSpd
relspd_cols = ['RelSpd', 'RelSpd_12']
df = df.drop(relspd_cols, axis=1)

# df.head(20)


# In[32]:


df.RelSpd_2.describe()


# In[33]:


# create a new column RelSpd2_06
df.loc[:,'RelSpd2_06'] = df['RelSpd_2']

# set new column value 0 if RelSpd is equal or less than RelSpd_mean_2 (2.06), 1 if more
f = lambda x: 0 if x <= 2.06 else 1
df['RelSpd2_06'] = df['RelSpd2_06'].map(f)

# df.head(20)


# In[34]:


# set plot size, color etc.
sns.set(rc={'figure.figsize':(9.7,7.27)})
sns.set(font='sans-serif', palette='colorblind')

# set plot parameters
plot = sns.countplot(x = 'Yds4_18',
                data = df,
                hue = 'RelSpd2_06',
                order = df['Yds4_18'].value_counts().index)

# set plot title etc.
plot.axes.set_title('4.18 yards threshold divided by relative speed (2.06)',fontsize=18)
plot.set_xlabel("0 = 4.18 yards run or less, 1 = more",fontsize=18)
plot.set_ylabel("Total count of rows",fontsize=18)
plot.tick_params(labelsize=14)
plot.legend (loc=1, fontsize = 16, fancybox=True, framealpha=1, shadow=True, borderpad=1, title = '0 = 2.06 or less, 1 = more')

# show plot
plt.show()


# In[35]:


# set histogram parameters
fig = px.histogram(df, x="X_rb", nbins = 100, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "Running back X position datapoints divided by percentage",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# show histogram
fig.show()


# In[36]:


# set histogram parameters
fig = px.histogram(df, x="YardLine", nbins = 100, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "YardLine column datapoints divided by percentage",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# show histogram
fig.show()


# In[37]:


# set histogram parameters
fig = px.histogram(df, x="Down", nbins = 8, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 4
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "Down column datapoints divided by percentage",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# set x ticks
fig.update_xaxes(nticks = 4)

# show histogram
fig.show()


# In[38]:


# set plot size, color etc.
sns.set(rc={'figure.figsize':(9.7,7.27)})
sns.set(font='sans-serif', palette='colorblind')

# set plot parameters
plot = sns.countplot(x = 'Yds4_18',
                data = df,
                hue = 'Down',
                order = df['Yds4_18'].value_counts().index)

# set plot title etc.
plot.axes.set_title('4.18 yards threshold divided by down (1-4)',fontsize=24)
plot.set_xlabel("0 = 4.18 yards run or less, 1 = more",fontsize=18)
plot.set_ylabel("Total count of rows",fontsize=18)
plot.tick_params(labelsize=14)
plot.legend (loc=1, fontsize = 16, fancybox=True, framealpha=1, shadow=True, borderpad=1, title = 'Down (1-4)')

# show plot
plt.show()


# In[39]:


# set plot size, color etc.
sns.set(rc={'figure.figsize':(9.7,7.27)})
sns.set(font='sans-serif', palette='colorblind')

# set plot parameters
plot = sns.countplot(x = 'Yds4_18',
              data = df,
              order = df['Yds4_18'].value_counts().index)

# set plot title and x/y labels
plot.axes.set_title('Total count of dataset rows divided by 4.18+ run plays',fontsize=18)
plot.set_xlabel("0 = 4.18 yards or less, 1 = more",fontsize=18)
plot.set_ylabel("Total count of rows",fontsize=18)
plot.tick_params(labelsize=14)

#show the plot
plt.show()


# In[40]:


run4_18_count = df['Yds4_18'].value_counts()
print (run4_18_count)


# In[41]:


# set histogram parameters
fig = px.histogram(df, x="PlayYards", nbins = 100, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "PlayYards column datapoints divided by percentage",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# show histogram
fig.show()


# In[42]:


# create new dataframe including only PlayYard values between 1-4
df_2 = df[(df['PlayYards']>= 0 ) & (df['PlayYards']<= 4)]

# plot historgram with yards between 1-4
fig = px.histogram(df_2, x="PlayYards", nbins = 16, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 4
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "PlayYards column datapoints divided by yards 0-4",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# set x ticks
fig.update_xaxes(nticks=5)

# show ploe
fig.show()


# In[43]:


# create new dataframe including only PlayYard values between 5-10
df_3 = df[(df['PlayYards']>= 5 ) & (df['PlayYards']<= 10)]

# plot histogram with yards between 5-10
fig = px.histogram(df_3, x="PlayYards", nbins = 16, histnorm = 'percent')
fig.data[0].marker.color = "orange"
fig.data[0].marker.line.width = 4
fig.data[0].marker.line.color = "black"

# set plot title
fig.update_layout(
    title={
        'text': "PlayYards column datapoints divided by yards 5-10",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# show plot
fig.show()


# In[44]:


# import modules 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# show dataframe column names
df.columns


# In[45]:


# select the desired features as X
X = df[['Is3Wr', 'Is3Lb', 'Is4Lb', 'acc_lb', 'acc_rb', 'spd_lb', 
               'spd_rb','RelAcc_2', 'euc_2','RelSpd_2']]

# select the labels as y (in this case 4.18+ yard runs)
y = df['Yds4_18']

# perform train-test split, train data 80%, no random state
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 0)

# print train-test dataset sizes
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[46]:


# import module
from sklearn.preprocessing import StandardScaler

# scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create and fit the Logistic Regression model
model = LogisticRegression(solver='lbfgs', class_weight='balanced')
model.fit(X_train, y_train)

# print the scores
print (model.score(X_train, y_train))
print (model.score(X_test, y_test))


# In[47]:


# print the coefficients
print(model.coef_)


# In[48]:


# print each feature with its respective coefficient value
print(list(zip(['Is3Wr', 'Is3Lb', 'Is4Lb', 'acc_lb', 'acc_rb', 'spd_lb', 
               'spd_rb','RelAcc_2', 'euc_2','RelSpd_2'],model.coef_[0])))


# In[49]:


# import module
from sklearn.metrics import confusion_matrix

# set model prediction
y_pred = model.predict(X_test)

# print prediction
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[50]:


# import module
from sklearn.metrics import classification_report

# prin report
print(classification_report(y_test, y_pred))


# In[51]:


# import module
from sklearn.ensemble import RandomForestClassifier

# select the desired features as X
X = df[['Is3Wr', 'Is3Lb', 'Is4Lb', 'acc_lb', 'acc_rb', 'spd_lb', 
               'spd_rb','RelAcc_2', 'euc_2','RelSpd_2']]

# select the labels as y (in this case 4.18+ yard runs)
y = df['Yds4_18']

# scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# define and fit Random Forest Classifier model
classifier = RandomForestClassifier(random_state=0, max_depth = 8, n_estimators = 100)
classifier.fit(X_train, y_train)

# set model prediction
y_pred = classifier.predict(X_test)

# print report
print(classification_report(y_test, y_pred))


# In[52]:


# import module
from sklearn.metrics import roc_curve

# define function for ROC surve
def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# set predictions        
probs = classifier.predict_proba(X_test)  
probs = probs[:, 1]  
fper, tper, thresholds = roc_curve(y_test, probs) 

# plot ROC curve
plot_roc_curve(fper, tper)


# In[53]:


# import modules
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# select the desired features as X
X = df[['Is3Wr', 'Is3Lb', 'Is4Lb', 'acc_lb', 'acc_rb', 'spd_lb', 
               'spd_rb','RelAcc_2', 'euc_2','RelSpd_2']]

# select the labels as y (in this case 4.18+ yard runs)
y = df['Yds4_18']

# scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# perform train-test split, train data 80%, no random state
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 0)

# define list of learning rates to test
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 1, 2]

# check which learning rate is the best and print the result
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators= 32, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)   
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (test): {0:.3f}".format(gb_clf.score(X_test, y_test)))


# In[54]:


# import modules
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# select the desired features as X
X = df[['Is3Wr', 'Is3Lb', 'Is4Lb', 'acc_lb', 'acc_rb', 'spd_lb', 
               'spd_rb','RelAcc_2', 'euc_2','RelSpd_2']]

# select the labels as y (in this case 4.18+ yard runs)
y = df['Yds4_18']

# scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# perform train-test split, train data 80%, no random state
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 0)

# fit the Gradient Booster model and set predictions
gb_clf2 = GradientBoostingClassifier(n_estimators=32, learning_rate=0.25, max_features=2, max_depth=3, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)

#print results
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report")
print(classification_report(y_test, predictions))


# In[55]:


# import modules
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

# select the desired features as X
X = df[['Is3Wr', 'Is3Lb', 'Is4Lb', 'acc_lb', 'acc_rb', 'spd_lb', 
               'spd_rb','RelAcc_2', 'euc_2','RelSpd_2']]

# select the labels as y (in this case 4.18+ yard runs)
y = df['Yds4_18']

# scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# define and fit the K nearest neighbors model
model = KNeighborsRegressor(n_neighbors=9)
model.fit(X_train, y_train)

# calculate the errors for our training data
mse = mean_squared_error(y_train, model.predict(X_train))
mae = mean_absolute_error(y_train, model.predict(X_train))

# print results
print("mean squared error = ",mse," & mean absolute error = ",mae," & root mean squared error = ", sqrt(mse))


# In[56]:


# calculate the errors for our training data
test_mse = mean_squared_error(y_test, model.predict(X_test))
test_mae = mean_absolute_error(y_test, model.predict(X_test))

# print results
print("mean squared error = ",test_mse," & mean absolute error = ",test_mae," & root mean squared error = ", sqrt(test_mse))


# In[57]:


# import module
from sklearn.neighbors import KNeighborsClassifier

# select the desired features as X
X = df[['Is3Wr', 'Is3Lb', 'Is4Lb', 'acc_lb', 'acc_rb', 'spd_lb', 
               'spd_rb','RelAcc_2', 'euc_2','RelSpd_2']]

# select the labels as y (in this case 4.18+ yard runs)
y = df['Yds4_18']

# scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# define and fit K nearest neighbor classifier model
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)

# set predictions
y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)

# print results
print(confusion_matrix)
print(classification_report(y_test, y_pred))


# In[58]:


# create an empty list for error values
error = []

# function to calculate K value
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# create plot for visualizing the results    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)

# set plot title etc.
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# show plot
plt.show()


# In[59]:


# import modules
import eli5
from pdpbox import pdp, get_dataset, info_plots
from eli5.sklearn import PermutationImportance
import joblib


# select the desired features as X
X = df[['Is3Wr', 'Is3Lb', 'Is4Lb', 'acc_lb', 'acc_rb', 'spd_lb', 
               'spd_rb','RelAcc_2', 'euc_2','RelSpd_2']]

# select the labels as y (in this case 4.18+ yard runs)
y = df['Yds4_18']

# scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# perform train-test split, train data 80%, no random state
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 0)

# define Random Forest Classifier model
model = RandomForestClassifier(n_estimators=32, random_state=0).fit(X_train, y_train)

# fit permutation importance and show the results 
perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[60]:


# define column used in graph
feature_name = 'acc_rb'

# Create the data that we will plot
my_pdp = pdp.pdp_isolate(model=model, dataset=X_train, model_features=X_train.columns, feature=feature_name)

# set the plot
pdp.pdp_plot(my_pdp, feature_name)

# show plot
plt.show()

