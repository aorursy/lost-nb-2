#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import math
import scipy.stats as st

class Season():
    def __init__(self, year):
        self.year = year
        seasonResults = pd.read_csv("../input/WRegularSeasonCompactResults.csv")
        yearMask = seasonResults.loc[:,'Season'] == self.year
        self.games = seasonResults.loc[yearMask,:].copy()
        self.__setTeams()
        self.expectedScore = (self.games.loc[:,"WScore"].mean() + self.games.loc[:,"LScore"].mean()) / 2
        self.__calcErrors()
        
    def __setTeams(self, filename = "../input/WNCAATourneySeeds.csv"):
        """
        Pulls all of the teams due to compete in the tournement in the coming year
        """
        temp = pd.read_csv(filename)
        temp = temp.query('Season == {}'.format(self.year)).copy()
        teamset = set()
        teamset.update(temp["TeamID"].unique())
        self.tournTeams = list(teamset)
        self.tournTeams.sort()
        
        allteamset = set(self.games.loc[:,"WTeamID"].unique())
        allteamset.update(self.games.loc[:,"LTeamID"].unique())
        self.allTeams = list(allteamset)
        self.allTeams.sort()
        
    def __calcErrors(self):
        scores = self.games.loc[:,"WScore"].copy().values
        scores = np.append(self.games.loc[:,"LScore"],scores)
        scores = np.std(scores)
        self.stdScore = scores
        
    def getTeamRecords(self,teamId):
        return self.games.query('(WTeamID == {} | LTeamID == {})'.format(teamId,teamId))
class Team():
    def __init__(self, teamId, season):
        self.teamId = teamId
        self.record = season.getTeamRecords(self.teamId)
        self.__setOffense()
        
    def play(self, opponent, gameData ):
        return 0
        
    def __str__(self):
        print(self.teamId)
        print("Team Scoring: ", self.teamStats)
        print("Opponent Scoring: ", self.oppStats)
        return ""
        
    def __setOffense(self):
        Wmask = self.record['WTeamID'] == self.teamId
        Lmask = self.record['LTeamID'] == self.teamId        
        
        teamScoreSeries = pd.Series([])
        teamScoreSeries = teamScoreSeries.append([self.record[Wmask]['WScore'].copy(),
                                self.record[Lmask]['LScore'].copy()], ignore_index=True)
        
        oppScoreSeries = pd.Series()
        oppScoreSeries = oppScoreSeries.append([self.record.loc[Wmask,'LScore'].copy(),
                                                self.record.loc[Lmask,'WScore'].copy()], ignore_index=True)
        
        self.teamStats = {"PPG":teamScoreSeries.mean(), "STD":teamScoreSeries.std()}
        self.oppStats = {"PPG":oppScoreSeries.mean(), "STD":oppScoreSeries.std()}

class Matchup(object):
    def __init__(self,team1,team2,season):
        self.season = season.year
        self.team1 = team1.teamId
        self.team2 = team2.teamId
        self.PTeam1Win = EstimateProb(team1,team2)
        
    def tostr(self):
        return "{}_{}_{},{}\n".format(
                self.season,self.team1,self.team2,self.PTeam1Win)


# In[10]:


years = [1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017] # an array of years from 1998 to 2018

teamData = pd.DataFrame({'Year':[], 'TeamId':[],'AverageScore':[], 'ScoreSTD':[], 'OpponentAverageScore':[], 
                         'OpponentScoreSTD':[]})
for year in years:
    season = Season(year)
    teamList = [Team(ID,season) for ID in season.allTeams]

    for team in teamList: 
        temp = pd.DataFrame({'Year': [year], 'TeamId': [team.teamId], 
                         'AverageScore': [team.teamStats['PPG']], 'ScoreSTD': [team.teamStats['STD']],
                         'OpponentAverageScore': [team.oppStats['PPG']], 
                         'OpponentScoreSTD': [team.oppStats['STD']]})
        teamData = pd.concat([teamData,temp])
    print(year, 'Done')
teamData['Year'] = teamData['Year'].astype('int', copy=False)
teamData['TeamId'] = teamData['TeamId'].astype('int', copy=False)
teamData['UID'] = teamData['Year'].astype(str) + '_' + teamData['TeamId'].astype(str)
teamData = teamData.set_index('UID')
teamData.head()


# In[3]:


tourney = pd.read_csv("../input/WNCAATourneyCompactResults.csv")
tourney.head()


# In[4]:


tourney['WUID'] = tourney['Season'].astype(str) + '_' + tourney['WTeamID'].astype(str)
tourney['LUID'] = tourney['Season'].astype(str) + '_' + tourney['LTeamID'].astype(str)
tourney.head()


# In[26]:


def EstimateProb(team1PPG, team1OPPG, team1STD, team1OSTD, team2PPG, team2OPPG, team2STD, team2OSTD):
    expVal = team1PPG - team2PPG
    expVal -= team1OPPG - team2OPPG
    
    variance = team1STD**2 + team1OSTD**2 
    variance += team2STD**2 + team2OSTD**2
    
    std =np.sqrt(variance)
    zScore = expVal / std
    np.nan_to_num(zScore)
    return 1 - st.norm.cdf(-1 * zScore)


# In[53]:


# The Requisite team1 statistics
team1PPG = tourney['WUID'].map(teamData['AverageScore']).values
team1OPPG = tourney['WUID'].map(teamData['OpponentAverageScore']).values
team1STD = tourney['WUID'].map(teamData['ScoreSTD']).values
team1OSTD = tourney['WUID'].map(teamData['OpponentScoreSTD']).values

# The Requisite team2 statisitics
team2PPG = tourney['LUID'].map(teamData['AverageScore']).values
team2OPPG = tourney['LUID'].map(teamData['OpponentAverageScore']).values
team2STD = tourney['LUID'].map(teamData['ScoreSTD']).values
team2OSTD = tourney['LUID'].map(teamData['OpponentScoreSTD']).values

tourney['Prediction'] = EstimateProb(team1PPG, team1OPPG, team1STD, team1OSTD, team2PPG, team2OPPG, team2STD, team2OSTD)
tourney['Result'] = 1
tourney.head()


# In[54]:


correct = tourney.loc[tourney['Prediction'] > .5, :]['Prediction'].size
incorrect = tourney.loc[tourney['Prediction'] <= .5, :]['Prediction'].size
print("{:.4f}".format(correct / (correct + incorrect)))


# In[57]:


from sklearn.metrics import log_loss
print("{:.4f}".format(log_loss(tourney['Result'].values, tourney['Prediction'].values, labels = [0,1])))


# In[ ]:


A fairly high log-loss on its own, however a valueable piece of information none the less.

