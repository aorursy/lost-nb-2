#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns
plt.style.use('seaborn-dark-palette')
mypal = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Grab the color pal
import os
import gc

MENS_DIR = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament'
WOMENS_DIR = '../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament'


# In[2]:


def logloss(true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -np.log(p)
    return -np.log(1 - p)


# In[3]:


print(f'Confident Wrong Prediction: \t\t {logloss(1, 0.01):0.4f}')
print(f'Confident Correct Prediction: \t\t {logloss(0, 0.01):0.4f}')
print(f'Non-Confident Wrong Prediction: \t {logloss(1, 0.49):0.4f}')
print(f'Non-Confident Correct Prediction: \t {logloss(0, 0.49):0.4f}')


# In[4]:


Mss = pd.read_csv(f'{MENS_DIR}/MSampleSubmissionStage1_2020.csv')
Wss = pd.read_csv(f'{WOMENS_DIR}/WSampleSubmissionStage1_2020.csv')
Mss.head()


# In[5]:


# Womens' data does not contain years joined :(
WTeams = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WTeams.csv')
WTeams.head()


# In[6]:


len(WTeams) # number of teams in total


# In[7]:


WSeasons = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WSeasons.csv')
WSeasons.head()

# Day Zero : first day of the season
# Regions = to identify the four regions


# In[8]:


WSeasons


# In[9]:


WNCAATourneySeeds = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WNCAATourneySeeds.csv')


# In[10]:


WNCAATourneySeeds.head()


# In[11]:


# lets get the seeds for 2019
# teams selected for the March Madness
march_2019 = WNCAATourneySeeds[WNCAATourneySeeds['Season'] == 2019]
march_2019


# In[12]:


# let's join this with the teams data to see some of the past matchups

teams = WNCAATourneySeeds.merge(WTeams, validate='many_to_one')


# In[13]:


teams


# In[14]:


len(teams['TeamID'].unique()) # teams selected for the NCAA


# In[15]:


count = teams.groupby('TeamName').count() # to see old and pretty young teams
count = count.sort_values('TeamID', ascending = False)
old_teams = count[count['Season']>10]


# In[16]:


plt.figure(figsize = (40,39))
plt.barh(count.index[:30], count['TeamID'][:30])


# In[17]:


WRegularSeasonCompactResults = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')


# In[18]:


# We have the team the won, lost and the score.
WRegularSeasonCompactResults.head(5)


# In[19]:


# Lets Add the winning and losing team names to the results

WRegularSeasonCompactResults =     WRegularSeasonCompactResults     .merge(WTeams[['TeamName', 'TeamID']],
           left_on='WTeamID',
           right_on='TeamID',
           validate='many_to_one') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'WTeamName'}) \
    .merge(WTeams[['TeamName', 'TeamID']],
           left_on='LTeamID',
           right_on='TeamID') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'LTeamName'})


# In[20]:


WRegularSeasonCompactResults


# In[21]:


results_post_2015 = WRegularSeasonCompactResults[WRegularSeasonCompactResults['Season']>2014]


# In[22]:


victories = WRegularSeasonCompactResults.groupby(['Season', 'WTeamName']).count()


# In[23]:


victories =victories.reset_index()


# In[ ]:





# In[24]:


for i in [2015,2016,2017,2018,2019]: 
    plt.style.use('fivethirtyeight')
    data =  victories[victories['Season']==i] 
    data = data.sort_values('DayNum', ascending = False)
    plt.figure(figsize = (15,12))
    a = 'Season '+str(i)
    plt.title(a)
    plt.barh(data['WTeamName'][:20], data['DayNum'][:20])


# In[25]:


WRegularSeasonCompactResults.head()


# In[26]:


# score difference 
WRegularSeasonCompactResults['Score_Diff'] = WRegularSeasonCompactResults['WScore'] - WRegularSeasonCompactResults['LScore']


# In[27]:


results_2009 = WRegularSeasonCompactResults[WRegularSeasonCompactResults['Season']>2009]


# In[28]:


plt.style.use('fivethirtyeight')
WRegularSeasonCompactResults['counter'] = 1
WRegularSeasonCompactResults.groupby('WTeamName')['counter']     .count()     .sort_values()     .tail(20)     .plot(kind='barh',
          title='Most Winning (Regular Season) Womens Teams',
          figsize=(15, 8),
          xlim=(400, 680),
          color=mypal[0])
plt.show()


# In[29]:


# after 2009

plt.style.use('fivethirtyeight')
results_2009['counter'] = 1
results_2009.groupby('WTeamName')['counter']     .count()     .sort_values()     .tail(20)     .plot(kind='barh',
          title='Most Winning (Regular Season) Teams after 2009',
          figsize=(15, 8),
          xlim=(10, 350),
          color=mypal[1])
plt.show()


# In[30]:


teams_2019 = WRegularSeasonCompactResults[WRegularSeasonCompactResults['Season']==2019]


# In[31]:


len(teams_2019['WTeamID'].unique())


# In[ ]:





# In[32]:


WRegularTourneyCompactResults = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')


# In[33]:


WRegularTourneyCompactResults =     WRegularTourneyCompactResults     .merge(WTeams[['TeamName', 'TeamID']],
           left_on='WTeamID',
           right_on='TeamID',
           validate='many_to_one') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'WTeamName'}) \
    .merge(WTeams[['TeamName', 'TeamID']],
           left_on='LTeamID',
           right_on='TeamID') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'LTeamName'})


# In[34]:


WRegularTourneyCompactResults['Score_Diff'] = WRegularTourneyCompactResults['WScore'] - WRegularTourneyCompactResults['LScore']


# In[35]:


WRegularTourneyCompactResults


# In[36]:


WNCAATourneySeeds


# In[37]:


WRegularTourneyCompactResults =     WRegularTourneyCompactResults     .merge(WNCAATourneySeeds[['Seed', 'TeamID', 'Season']],
           left_on=['WTeamID', 'Season'],
           right_on=['TeamID','Season'],
           validate='many_to_one') \
    .drop('TeamID', axis=1) \
    .rename(columns={'Seed': 'WSeed'}) \
    .merge(WNCAATourneySeeds[['Seed', 'TeamID', 'Season']],
           left_on=['LTeamID', 'Season'],
           right_on=['TeamID','Season'],
           validate='many_to_one') \
    .drop('TeamID', axis=1) \
    .rename(columns={'Seed': 'LSeed'})


# In[38]:


WRegularTourneyCompactResults


# In[39]:


WRegularTourneyCompactResults = WRegularTourneyCompactResults.sort_values(['Season', 'DayNum'])


# In[40]:


WRegularTourneyCompactResults


# In[41]:


WRegularTourneyCompactResults= WRegularTourneyCompactResults.reset_index()


# In[42]:


WRegularTourneyCompactResults['index']=WRegularTourneyCompactResults.index


# In[43]:


WRegularTourneyCompactResults['Round'] = 0 
#WRegularTourneyCompactResults = WRegularTourneyCompactResults.reset_index()
for i in WRegularTourneyCompactResults.index: 
    
    if WRegularTourneyCompactResults['Season'][i]<2003 :
        #print(WRegularTourneyCompactResults['Season'][i])
        #print(WRegularTourneyCompactResults['DayNum'][i])
        if WRegularTourneyCompactResults['DayNum'][i] == 137 :
            WRegularTourneyCompactResults['Round'][i]= 1 
        elif WRegularTourneyCompactResults['DayNum'][i] == 138: 
            WRegularTourneyCompactResults['Round'][i]= 1 
            
        elif WRegularTourneyCompactResults['DayNum'][i] == 139 :
            WRegularTourneyCompactResults['Round'][i]= 2 
        elif WRegularTourneyCompactResults['DayNum'] [i] == 140 :
            WRegularTourneyCompactResults['Round'][i]= 2 
            
        elif WRegularTourneyCompactResults['DayNum'][i] ==145 :
            WRegularTourneyCompactResults['Round'][i]= 3 
        elif WRegularTourneyCompactResults['DayNum'][i] ==147 :
            WRegularTourneyCompactResults['Round'][i]= 4 
        elif WRegularTourneyCompactResults['DayNum'][i] ==151: 
            WRegularTourneyCompactResults['Round'][i]= 5
        else: #WRegularTourneyCompactResults['DayNum'][i]==153:
            WRegularTourneyCompactResults['Round'][i]= 6
                

    else :   
        WRegularTourneyCompactResults['Round'][i] = 0 
        if WRegularTourneyCompactResults['Season'][i]<2015 : 
            if WRegularTourneyCompactResults['DayNum'][i] ==138 :
                WRegularTourneyCompactResults['Round'][i]= 1 
            elif WRegularTourneyCompactResults['DayNum'][i] ==139: 
                WRegularTourneyCompactResults['Round'][i]= 1
            elif WRegularTourneyCompactResults['DayNum'][i] == 140 :
                WRegularTourneyCompactResults['Round'][i]= 2
            elif WRegularTourneyCompactResults['DayNum'][i] ==141:
                WRegularTourneyCompactResults['Round'][i]= 2 
            elif WRegularTourneyCompactResults['DayNum'][i] ==145 :
                WRegularTourneyCompactResults['Round'][i]= 3 
            elif WRegularTourneyCompactResults['DayNum'][i] ==146:
                WRegularTourneyCompactResults['Round'][i]= 3 
            elif WRegularTourneyCompactResults['DayNum'][i] ==147:
                WRegularTourneyCompactResults['Round'][i]= 4
            elif WRegularTourneyCompactResults['DayNum'][i] ==148:
                WRegularTourneyCompactResults['Round'][i]= 4 
            elif WRegularTourneyCompactResults['DayNum'][i] ==153: 
                WRegularTourneyCompactResults['Round'][i]= 5
            else: #WRegularTourneyCompactResults['DayNum'][i]==155:
                WRegularTourneyCompactResults['Round'][i]= 6
    
        else :  
            if WRegularTourneyCompactResults['Season'][i]<2017 : 

                if WRegularTourneyCompactResults['DayNum'][i] ==137:
                    WRegularTourneyCompactResults['Round'][i]= 1
                elif WRegularTourneyCompactResults['DayNum'][i] ==138:
                    WRegularTourneyCompactResults['Round'][i]= 1 
                elif WRegularTourneyCompactResults['DayNum'][i] ==139 or WRegularTourneyCompactResults['DayNum'][i] ==140:
                    WRegularTourneyCompactResults['Round'][i]= 2 
                elif WRegularTourneyCompactResults['DayNum'][i] ==144 or WRegularTourneyCompactResults['DayNum'][i] ==145:
                    WRegularTourneyCompactResults['Round'][i]= 3 
                elif WRegularTourneyCompactResults['DayNum'][i] ==146 or WRegularTourneyCompactResults['DayNum'][i] ==147:
                    WRegularTourneyCompactResults['Round'][i]= 4 
                elif WRegularTourneyCompactResults['DayNum'][i] ==153: 
                    WRegularTourneyCompactResults['Round'][i]= 5
                else: # WRegularTourneyCompactResults['DayNum'][i]==155:
                    WRegularTourneyCompactResults['Round'][i]= 6

            else : 
                if WRegularTourneyCompactResults['DayNum'][i] ==137 or WRegularTourneyCompactResults['DayNum'][i] ==138:
                    WRegularTourneyCompactResults['Round'][i]= 1 
                elif WRegularTourneyCompactResults['DayNum'][i] ==139 or WRegularTourneyCompactResults['DayNum'][i] ==140:
                    WRegularTourneyCompactResults['Round'][i]= 2 
                elif WRegularTourneyCompactResults['DayNum'][i] ==144 or WRegularTourneyCompactResults['DayNum'][i] ==145:
                    WRegularTourneyCompactResults['Round'][i]= 3 
                elif WRegularTourneyCompactResults['DayNum'][i] ==146 or WRegularTourneyCompactResults['DayNum'][i] ==147:
                    WRegularTourneyCompactResults['Round'][i]= 4 
                elif WRegularTourneyCompactResults['DayNum'][i] ==151: 
                    WRegularTourneyCompactResults['Round'][i]= 5
                else: # WRegularTourneyCompactResults['DayNum'][i] ==153:
                    WRegularTourneyCompactResults['Round'][i]= 6  
            


# In[44]:



WRegularTourneyCompactResults['Region']=''
for i in WRegularTourneyCompactResults.index:
    if WRegularTourneyCompactResults['LSeed'][i][0] == WRegularTourneyCompactResults['WSeed'][i][0]: 
        WRegularTourneyCompactResults['Region'][i]=WRegularTourneyCompactResults['LSeed'][i][0]
    else : 
        WRegularTourneyCompactResults['Region'][i]=WRegularTourneyCompactResults['WSeed'][i][0] + WRegularTourneyCompactResults['LSeed'][i][0]
        
        


# In[45]:


WRegularTourneyCompactResults['Seeds']=''
for i in WRegularTourneyCompactResults.index: 
    WRegularTourneyCompactResults['Seeds'][i] = str(WRegularTourneyCompactResults['WSeed'][i][1:]) + '-' + str(int(WRegularTourneyCompactResults['LSeed'][i][1:]))


# In[46]:


cinderella= pd.DataFrame()
same_seed = pd.DataFrame()
predicted =pd.DataFrame()

for i in WRegularTourneyCompactResults.index: 
    if int(WRegularTourneyCompactResults['WSeed'][i][1:])>int(WRegularTourneyCompactResults['LSeed'][i][1:]):
        #print((WRegularTourneyCompactResults['WSeed'][i][1:], WRegularTourneyCompactResults['LSeed'][i][1:]))
        cinderella = pd.concat([cinderella, pd.DataFrame(WRegularTourneyCompactResults[WRegularTourneyCompactResults['index']==i])])
    elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==int(WRegularTourneyCompactResults['LSeed'][i][1:]):
        same_seed = pd.concat([same_seed, pd.DataFrame(WRegularTourneyCompactResults[WRegularTourneyCompactResults['index']==i])])
    
    else : 
        predicted = pd.concat([predicted, pd.DataFrame(WRegularTourneyCompactResults[WRegularTourneyCompactResults['index']==i])])


# In[47]:


# surprises 
cinderella= cinderella.reset_index()

# same seed games 
same_seed =same_seed.reset_index()

# predicted wins with seeds 
predicted =predicted.reset_index()


# In[48]:


round_6 = WRegularTourneyCompactResults[WRegularTourneyCompactResults['Round']==6]
round_1 = WRegularTourneyCompactResults[WRegularTourneyCompactResults['Round']==1]
round_5 = WRegularTourneyCompactResults[WRegularTourneyCompactResults['Round']==5]


# In[49]:


# Surprises at Round 1 
plt.figure(figsize =(30,30))
test = round_1.groupby('Seeds').count()
test = test.sort_values('Season')
plt.bar(test.index, test['Season'])
plt.title('Surprises according to seeds')


# In[50]:


# Surprised in Semis (Round 5)

plt.figure(figsize =(30,30))
test = round_5.groupby('Seeds').count()
test = test.sort_values('Season')
plt.bar(test.index, test['Season'])
plt.title('Surprises according to seeds')


# In[51]:


# Surprises in Final (round 6)

plt.figure(figsize =(30,30))
test = round_6.groupby('Seeds').count()
test = test.sort_values('Season')
plt.bar(test.index, test['Season'])
plt.title('Surprises according to seeds')


# In[ ]:





# In[52]:


(len(same_seed)/len(WRegularTourneyCompactResults))*100


# In[53]:


cinderella_2019 = cinderella[cinderella['Season']==2019]
march_2019 = WRegularTourneyCompactResults[WRegularTourneyCompactResults['Season']==2019]
march_2019 = march_2019.sort_values('DayNum')


# In[54]:


(len(cinderella)/len(WRegularTourneyCompactResults))*100


# In[55]:


# in 2019
len(cinderella_2019)/len(march_2019)


# In[56]:


# get the ones with more than 1 seed difference 
# for ex match between W01 and W02 wont count 

big_cinderella = pd.DataFrame()
little_cinderella = pd.DataFrame()
for i in cinderella.index: 
    if int(cinderella['WSeed'][i][1:]) - int(cinderella['LSeed'][i][1:])>1 :
        big_cinderella = pd.concat([big_cinderella, pd.DataFrame(cinderella[WRegularTourneyCompactResults['index']==i])])
    else : 
        little_cinderella = pd.concat([little_cinderella, pd.DataFrame(cinderella[WRegularTourneyCompactResults['index']==i])])
        


# In[57]:


# Big Cinderellas 


# In[58]:


(len(big_cinderella)/len(WRegularTourneyCompactResults))*100


# In[59]:


big_cinderella.groupby('Seeds').count()['Season'].sort_values().plot(kind = 'barh',
          title='Seeds for big cinderellas',
          figsize=(15, 8),
          color=mypal[0])
plt.show()


# In[60]:


(len(little_cinderella)/len(WRegularTourneyCompactResults))*100


# In[61]:


little_cinderella.groupby('Seeds').count()['Season'].sort_values().plot(kind = 'barh',
          title='Seeds for Little cinderellas',
          figsize=(15, 8),
          color=mypal[0])
plt.show()


# In[62]:


# number of surprises over the years
plt.style.use('fivethirtyeight')
test = cinderella.groupby('Season').count()
plt.bar(test.index, test['index'])
plt.title('Cinderellas over the years')


# In[63]:


# number of big surprises over the years
plt.style.use('fivethirtyeight')
test = big_cinderella.groupby('Season').count()
plt.bar(test.index, big_cinderella.groupby('Season').count()['index'])
plt.title('Big Cinderellas over the years')


# In[64]:


# predict surprises according to rounds

test = cinderella.groupby('Round').count()
plt.bar(test.index, test['Season'])
plt.title('Surprises over the years according to rounds')


# In[65]:


# in 2019 

test = cinderella_2019.groupby('Round').count()
plt.bar(test.index, test['Season'])
plt.title('Surprises in each round in season 2019')


# In[66]:


test = cinderella.groupby('Region').count()
plt.bar(test.index, test['Season'])
plt.title('Surprises IN REGIONS')


# In[67]:


for i in cinderella['Season'].unique():
    cind =cinderella[cinderella['Season']==i ]
    test = cind.groupby('Region').count()
    plt.figure()
    plt.bar(test.index, test['Season'])
    plt.title(i)


# In[68]:


# code to get surprises per year per region per round

'''for i in cinderella['Season'].unique()[20:]:
    cind =cinderella[cinderella['Season']==i ]
    
    for j in cind['Region'].unique():
        cind2 = cind[cind['Region']==j]
        test = cind2.groupby('Round').count()
        
        plt.figure(figsize =(20,20))
        plt.bar(test.index, test['Season'])
        hello = str(i)+' in region ' + str(j)
        plt.title(hello)'''


# In[69]:


cinderella.groupby('Seeds').count()['Season'].sort_values().plot(kind = 'barh',
          title='Seeds for All cinderellas',
          figsize=(15, 8),
          color=mypal[0])
plt.show()


# In[70]:


# what seeds are involved mostly 
for i in cinderella['Season'].unique():
    cind.groupby('Seeds').count()['Season'].sort_values().plot(kind = 'barh',
              title='Seeds for All cinderellas in '+str(i) ,
              figsize=(15, 8),
              color=mypal[0])
    plt.show()


# In[71]:


for i in big_cinderella['Season'].unique():
    cind =big_cinderella[big_cinderella['Season']==i ]
    cind.groupby('Region').count()['Season'].sort_values().plot(kind = 'barh',
              title='Big Cinderellas in Season ' +str(i) ,
              figsize=(15, 8),
              color=mypal[0])
    plt.show()


# In[72]:



cinderella_post_2010 = cinderella[cinderella['Season']>2009]
plt.figure(figsize =(30,30))
test = cinderella_post_2010.groupby('Seeds').count()
test = test.sort_values('Season')
plt.bar(test.index, test['Season'])
plt.title('Surprises according to seeds')


# In[73]:


# to get the round of each seed combination
tryt = cinderella[cinderella['Seeds']=='07-2']
tryt


# In[74]:


big_cinderella_2019 = big_cinderella[big_cinderella['Season']==2019]


# In[75]:


big_cinderella_2019


# In[76]:


(len(big_cinderella)/len(WRegularTourneyCompactResults))*100


# In[77]:


(len(little_cinderella)/len(WRegularTourneyCompactResults))*100


# In[78]:


# find the round for each game but NOT WORKING VERSIONA

'''
WRegularTourneyCompactResults['Round']= 0
for i in WRegularTourneyCompactResults.index :
    for k in range(len(L)):     
        if int(WRegularTourneyCompactResults['WSeed'][i][1:])==L[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== M[k]:
            WRegularTourneyCompactResults['Round'][i]= 1                                                                      
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==M[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== L[k]:
            WRegularTourneyCompactResults['Round'][i]= 1                                                                         
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==L[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== M_2[k]:
            WRegularTourneyCompactResults['Round'][i]= 2                                                                           
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==M_2[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== L[k]:
            WRegularTourneyCompactResults['Round'][i]= 2
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==L_2[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== M[k]:
            WRegularTourneyCompactResults['Round'][i]= 2
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==M[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== L_2[k]:
            WRegularTourneyCompactResults['Round'][i]= 2
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==L[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== L_2[k]:
            WRegularTourneyCompactResults['Round'][i]= 2
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==L_2[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== L[k]:
            WRegularTourneyCompactResults['Round'][i]= 2
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==M[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== M_2[k]:
            WRegularTourneyCompactResults['Round'][i]= 2
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==M_2[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== M[k]:
            WRegularTourneyCompactResults['Round'][i]= 2
            

        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==L[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== L[k-4]:
            WRegularTourneyCompactResults['Round'][i]= 3
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==L[k-4] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== L[k]:
            WRegularTourneyCompactResults['Round'][i]= 3
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==M[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== M[k-4]:
            WRegularTourneyCompactResults['Round'][i]= 3
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==M[k-4] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== M[k]:
            WRegularTourneyCompactResults['Round'][i]= 3
            

        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==L[k] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== M[k-4]:
            WRegularTourneyCompactResults['Round'][i]= 3
        elif int(WRegularTourneyCompactResults['WSeed'][i][1:])==M[k-4] and int(WRegularTourneyCompactResults['LSeed'][i][1:])== L[k]:
            WRegularTourneyCompactResults['Round'][i]= 3

'''


# In[79]:


#2017 season through 2020 season:
#Round 1 = days 137/138 (Fri/Sat)
#Round 2 = days 139/140 (Sun/Mon)
#Round 3 = days 144/145 (Sweet Sixteen, Fri/Sat)
#Round 4 = days 146/147 (Elite Eight, Sun/Mon)
#National Seminfinal = day 151 (Fri)
#National Final = day 153 (Sun)

#2015 season and 2016 season:
#Round 1 = days 137/138 (Fri/Sat)
#Round 2 = days 139/140 (Sun/Mon)
#Round 3 = days 144/145 (Sweet Sixteen, Fri/Sat)
#Round 4 = days 146/147 (Elite Eight, Sun/Mon)
#National Seminfinal = day 153 (Sun)
#National Final = day 155 (Tue)

#2003 season through 2014 season:
#Round 1 = days 138/139 (Sat/Sun)
#Round 2 = days 140/141 (Mon/Tue)
#Round 3 = days 145/146 (Sweet Sixteen, Sat/Sun)
#Round 4 = days 147/148 (Elite Eight, Mon/Tue)
#National Seminfinal = day 153 (Sun)
#National Final = day 155 (Tue)

#1998 season through 2002 season:
#Round 1 = days 137/138 (Fri/Sat)
#Round 2 = days 139/140 (Sun/Mon)
#Round 3 = day 145 only (Sweet Sixteen, Sat)
#Round 4 = day 147 only (Elite Eight, Mon)
#National Seminfinal = day 151 (Fri)
#National Final = day 153 (Sun)


# In[80]:


WPlayers = pd.read_csv(f'{WOMENS_DIR}/WPlayers.csv')


# In[81]:


WPlayers


# In[82]:


womens_events = []
for year in [2015, 2016, 2017, 2018, 2019]:
    womens_events.append(pd.read_csv(f'{WOMENS_DIR}/WEvents{year}.csv'))
WEvents = pd.concat(womens_events)
print(WEvents.shape)


# In[83]:


WEvents.head()


# In[84]:


del womens_events
gc.collect()


# In[85]:


# Merge Player name onto events

WEvents = WEvents.merge(WPlayers,
              how='left',
              left_on='EventPlayerID',
              right_on='PlayerID')


# In[86]:


WEvents


# In[87]:


# Event Types
plt.style.use('fivethirtyeight')
WEvents['counter'] = 1
WEvents.groupby('EventType')['counter']     .sum()     .sort_values(ascending=False)     .plot(kind='bar',
          figsize=(15, 5),
         color=mypal[3],
         title='Event Type Frequency (Womens)')
plt.xticks(rotation=0)
plt.show()


# In[ ]:





# In[88]:


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

WEvents['Area_Name'] = WEvents['Area'].map(area_mapping)


# In[89]:


WEvents.groupby('Area_Name')['counter'].sum()     .sort_values()     .plot(kind='barh',
          figsize=(15, 8),
          title='Frequency of Event Area')
plt.show()


# In[90]:


fig, ax = plt.subplots(figsize=(15, 8))
for i, d in WEvents.loc[~WEvents['Area_Name'].isna()].groupby('Area_Name'):
    d.plot(x='X', y='Y', style='.', label=i, ax=ax, title='Visualizing Event Areas')
    ax.legend()
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.show()


# In[91]:


# Normalize X, Y positions for court dimentions
# Court is 50 feet wide and 94 feet end to end.

WEvents['X_'] = (WEvents['X'] * (94/100))
WEvents['Y_'] = (WEvents['Y'] * (50/100))


# In[92]:


def create_ncaa_full_court(ax=None, three_line='mens', court_color='#dfbb85',
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


fig, ax = plt.subplots(figsize=(15, 8.5))
create_ncaa_full_court(ax, three_line='both', paint_alpha=0.4)
plt.show()


# In[93]:


fig, ax = plt.subplots(figsize=(15, 7.8))
ms = 10
ax = create_ncaa_full_court(ax, paint_alpha=0.1)
WEvents.query('EventType == "turnover"')     .plot(x='X_', y='Y_', style='X',
          title='Turnover Locations (Mens)',
          c='red',
          alpha=0.3,
         figsize=(15, 9),
         label='Steals',
         ms=ms,
         ax=ax)
ax.set_xlabel('')
ax.get_legend().remove()
plt.show()


# In[94]:


COURT_COLOR = '#dfbb85'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
# Where are 3 pointers made from? (This is really cool)
WEvents.query('EventType == "made3"')     .plot(x='X_', y='Y_', style='.',
          color='blue',
          title='3 Pointers Made (Womens)',
          alpha=0.01, ax=ax1)
ax1 = create_ncaa_full_court(ax1, lw=0.5, three_line='womens', paint_alpha=0.1)
ax1.set_facecolor(COURT_COLOR)
WEvents.query('EventType == "miss3"')     .plot(x='X_', y='Y_', style='.',
          title='3 Pointers Missed (Womens)',
          color='red',
          alpha=0.01, ax=ax2)
ax2.set_facecolor(COURT_COLOR)
ax2 = create_ncaa_full_court(ax2, lw=0.5, three_line='womens', paint_alpha=0.1)
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_xlabel('')
ax2.set_xlabel('')
plt.show()


# In[95]:


COURT_COLOR = '#dfbb85'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
# Where are 3 pointers made from? (This is really cool)
WEvents.query('EventType == "made2"')     .plot(x='X_', y='Y_', style='.',
          color='blue',
          title='2 Pointers Made (Womens)',
          alpha=0.01, ax=ax1)
ax1.set_facecolor(COURT_COLOR)
ax1 = create_ncaa_full_court(ax1, lw=0.5, three_line='womens', paint_alpha=0.1)
WEvents.query('EventType == "miss2"')     .plot(x='X_', y='Y_', style='.',
          title='2 Pointers Missed (Womens)',
          color='red',
          alpha=0.01, ax=ax2)
ax2.set_facecolor(COURT_COLOR)
ax2 = create_ncaa_full_court(ax2, lw=0.5, three_line='womens', paint_alpha=0.1)
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_xlabel('')
ax2.set_xlabel('')
plt.show()


# In[96]:


WPlayers = pd.read_csv(f'{WOMENS_DIR}/WPlayers.csv')


# In[97]:


WPlayers.head()


# In[ ]:





# In[ ]:





# In[98]:


ms = 10 # Marker Size
FirstName = 'Katie Lou'
LastName = 'Samuelson'
fig, ax = plt.subplots(figsize=(15, 8))
ax = create_ncaa_full_court(ax, three_line='womens')
WEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "made2"')     .plot(x='X_', y='Y_', style='o',
          title='Shots (Katie Lou Samuelson)',
          alpha=0.5,
         figsize=(15, 8),
         label='Made 2',
         ms=ms,
         ax=ax)
plt.legend()
WEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "miss2"')     .plot(x='X_', y='Y_', style='X',
          alpha=0.5, ax=ax,
         label='Missed 2',
         ms=ms)
plt.legend()
WEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "made3"')     .plot(x='X_', y='Y_', style='o',
          c='brown',
          alpha=0.5,
         figsize=(15, 8),
         label='Made 3', ax=ax,
         ms=ms)
plt.legend()
WEvents.query('FirstName == @FirstName and LastName == @LastName and EventType == "miss3"')     .plot(x='X_', y='Y_', style='X',
          c='green',
          alpha=0.5, ax=ax,
         label='Missed 3',
         ms=ms)
ax.set_xlabel('')
plt.legend()
plt.show()


# In[99]:


N_bins = 100
shot_events = WEvents.loc[WEvents['EventType'].isin(['miss3','made3','miss2','made2']) & (WEvents['X_'] != 0)]
fig, ax = plt.subplots(figsize=(15, 7))
ax = create_ncaa_full_court(ax,
                            paint_alpha=0.0,
                            three_line='mens',
                            court_color='black',
                            lines_color='white')
_ = plt.hist2d(shot_events['X_'].values + np.random.normal(0, 0.1, shot_events['X_'].shape), # Add Jitter to values for plotting
           shot_events['Y_'].values + np.random.normal(0, 0.1, shot_events['Y_'].shape),
           bins=N_bins, norm=mpl.colors.LogNorm(),
               cmap='plasma')

# Plot a colorbar with label.
cb = plt.colorbar()
cb.set_label('Number of shots')

ax.set_title('Shot Heatmap (Mens)')
plt.show()


# In[100]:


N_bins = 100
shot_events = WEvents.loc[WEvents['EventType'].isin(['miss3','made3','miss2','made2']) & (WEvents['X_'] != 0)]
fig, ax = plt.subplots(figsize=(15, 7))
ax = create_ncaa_full_court(ax, three_line='womens', paint_alpha=0.0,
                            court_color='black',
                            lines_color='white')
_ = plt.hist2d(shot_events['X_'].values + np.random.normal(0, 0.2, shot_events['X_'].shape),
           shot_events['Y_'].values + np.random.normal(0, 0.2, shot_events['Y_'].shape),
           bins=N_bins, norm=mpl.colors.LogNorm(),
               cmap='plasma')

# Plot a colorbar with label.
cb = plt.colorbar()
cb.set_label('Number of shots')

ax.set_title('Shot Heatmap (Womens)')
plt.show()


# In[101]:


MEvents['PointsScored'] =  0
MEvents.loc[MEvents['EventType'] == 'made2', 'PointsScored'] = 2
MEvents.loc[MEvents['EventType'] == 'made3', 'PointsScored'] = 3
MEvents.loc[MEvents['EventType'] == 'missed2', 'PointsScored'] = 0
MEvents.loc[MEvents['EventType'] == 'missed3', 'PointsScored'] = 0


# In[102]:


# # Average Points Scored per xy coord
# avg_pnt_xy = MEvents.loc[MEvents['EventType'].isin(['miss3','made3','miss2','made2']) & (MEvents['X_'] != 0)] \
#     .groupby(['X_','Y_'])['PointsScored'].mean().reset_index()

# # .plot(x='X_',y='Y_', style='.')
# fig, ax = plt.subplots(figsize=(15, 8))
# ax = sns.scatterplot(data=avg_pnt_xy, x='X_', y='Y_', hue='PointsScored', cmap='coolwarm')
# ax = create_ncaa_full_court(ax)
# plt.show()


# In[103]:


# avg_made_xy.sort_values('Made')


# In[104]:


# avg_made_xy['Made'] / avg_made_xy['Missed']


# In[105]:


# MEvents['Made'] = False
# MEvents['Made'] = False
# MEvents.loc[MEvents['EventType'] == 'made2', 'Made'] = True
# MEvents.loc[MEvents['EventType'] == 'made3', 'Made'] = True
# MEvents.loc[MEvents['EventType'] == 'missed2', 'Made'] = False
# MEvents.loc[MEvents['EventType'] == 'missed3', 'Made'] = False
# MEvents.loc[MEvents['EventType'] == 'made2', 'Missed'] = False
# MEvents.loc[MEvents['EventType'] == 'made3', 'Missed'] = False
# MEvents.loc[MEvents['EventType'] == 'missed2', 'Missed'] = True
# MEvents.loc[MEvents['EventType'] == 'missed3', 'Missed'] = True

# # Average Pct Made per xy coord
# avg_made_xy = MEvents.loc[MEvents['EventType'].isin(['miss3','made3','miss2','made2']) & (MEvents['X_'] != 0)] \
#     .groupby(['X_','Y_'])['Made','Missed'].sum().reset_index()

# # .plot(x='X_',y='Y_', style='.')
# fig, ax = plt.subplots(figsize=(15, 8))
# cmap = sns.cubehelix_palette(as_cmap=True)
# ax = sns.scatterplot(data=avg_made_xy, x='X_', y='Y_', size='Made', cmap='plasma')
# ax = create_ncaa_full_court(ax, paint_alpha=0)
# ax.set_title('Number of Shots Made')
# plt.show()

