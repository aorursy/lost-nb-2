#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# this is the base directory so you can switch between local, Kaggle Notebook, etc.
data_directory = "../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1"


# In[2]:


# this is a template for the core function
# predict the probability that in a given season (year) lteam beats gteam 
def predict_dummy(season, lteam_id, gteam_id):
    return(0.5)

predict_dummy(2019, 1115, 1118)


# In[3]:


tourney_results = pd.read_csv(os.path.join(data_directory,'MNCAATourneyCompactResults.csv'))

tourney_results = tourney_results.loc[(tourney_results["DayNum"] != 134) & (tourney_results["DayNum"] != 135)]

tourney_results.index = range(len(tourney_results.index))

tourney_results


# In[4]:


tourney_results["Truth"] = (tourney_results["WScore"] > tourney_results["LScore"]).astype("float")

tourney_results["Pred_Dummy"] =     tourney_results.apply(lambda row: 
                          predict_dummy(row['Season'], row['WTeamID'], row["LTeamID"]), 
                          axis=1)

tourney_results


# In[5]:


seeds = pd.read_csv(os.path.join(data_directory,'MNCAATourneySeeds.csv'))

seeds['SeedInt'] = [int(x[1:3]) for x in seeds['Seed']]

# Merge the seeds file with itself on Season.  This creates every combination of two teams by season.
seed_diff = seeds.merge(seeds, how='inner', on='Season')

seed_diff['ID'] = seed_diff['Season'].astype(str) + '_'               + seed_diff['TeamID_x'].astype(str) + '_'               + seed_diff['TeamID_y'].astype(str)

# formula found on Kaggle for getting probabilities out of seed differences
seed_diff['Pred'] = 0.5 + 0.030975*(seed_diff['SeedInt_y'] - seed_diff['SeedInt_x'])

seed_diff


# In[6]:


def predict_seed(season, lteam_id, gteam_id):
    return(seed_diff.loc[(seed_diff.TeamID_x == lteam_id) &
                 (seed_diff.TeamID_y == gteam_id) &
                 (seed_diff.Season == season)].Pred.mean())

predict_seed(1985, 1207, 1210)


# In[7]:


tourney_results["Pred_Seed"] =     tourney_results.apply(lambda row: 
                          predict_seed(row['Season'], row['WTeamID'], row["LTeamID"]), 
                          axis=1)

tourney_results


# In[8]:


# based on https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings

# a parameter for elo changes
elo_k = 20
# the home advantage
elo_home = 100

# functions taken for Elo calculation
# this function calculates a wind probability from the elo ratings
def elo_pred(elo1, elo2):
    return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))

# this calculates an expexted score difference for a given elo difference
def expected_margin(elo_diff):
    return((7.5 + 0.006 * elo_diff))

# this calculates the update (and the prediction) taking into account home advantage
def elo_update(w_elo, l_elo, margin, wloc):
    if wloc == "H":
        w_elo += elo_home
    elif wloc == "A":
        l_elo += elo_home
    elo_diff = w_elo - l_elo
    pred = elo_pred(w_elo, l_elo)
    mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
    update = elo_k * mult * (1 - pred)
    return(pred, update)


# In[9]:


season_results = pd.read_csv(os.path.join(data_directory,'MRegularSeasonCompactResults.csv'))

season_results["ScoreDiff"] = season_results.WScore - season_results.LScore
season_results["WElo"] = 0
season_results["LElo"] = 0

season_results


# In[10]:


# We initialise the ELOs to 1500 before the season
team_ids = set(season_results.WTeamID).union(set(season_results.LTeamID))

# cache for the current elo of all teams
current_elo = dict(zip(list(team_ids), [1500] * len(team_ids)))

for index, row in season_results.iterrows():
    pred, update = elo_update(current_elo[row.WTeamID],
                              current_elo[row.LTeamID],
                              row.ScoreDiff,
                              row.WLoc)
    current_elo[row.WTeamID] += update
    current_elo[row.LTeamID] -= update
    season_results.loc[index, "WElo"] = current_elo[row.WTeamID]
    season_results.loc[index, "LElo"] = current_elo[row.LTeamID]

season_results


# In[11]:


elo_summary = season_results[["Season", "DayNum", "WTeamID", "WElo"]].copy()
elo_summary.rename(columns={"WTeamID" : "LTeamID", "WElo" : "LElo"}, inplace=True)
elo_summary = pd.concat((elo_summary, season_results[["Season", "DayNum", "LTeamID", "LElo"]]), axis=0, ignore_index=True)
elo_summary.rename(columns = {"LTeamID" : "TeamID", "LElo" : "Elo"}, inplace=True)

idx = elo_summary[(elo_summary.Season == 2015) & (elo_summary.TeamID == 1433)]["DayNum"].idxmax()
elo_summary.iloc[idx].Elo


# In[12]:


def predict_elo(season, lteam_id, gteam_id):
    idx_l = elo_summary[(elo_summary.Season == season) & (elo_summary.TeamID == lteam_id)]["DayNum"].idxmax()
    idx_g = elo_summary[(elo_summary.Season == season) & (elo_summary.TeamID == gteam_id)]["DayNum"].idxmax()

    return(elo_pred(elo_summary.iloc[idx_l].Elo, elo_summary.iloc[idx_g].Elo))

predict_elo(1985, 1207, 1210)


# In[13]:


tourney_results["Pred_Elo"] =     tourney_results.apply(lambda row: 
                          predict_elo(row['Season'], row['WTeamID'], row["LTeamID"]), 
                          axis=1)

tourney_results


# In[14]:


def kaggle_clip_log(x):
    '''
    Calculates the natural logarithm, but with the argument clipped within [1e-15, 1 - 1e-15]
    '''
    return np.log(np.clip(x,1.0e-15, 1.0 - 1.0e-15))

def kaggle_log_loss(pred, result):
    '''
    Calculates the kaggle log loss for prediction pred given result result
    '''
    return -(result*kaggle_clip_log(pred) + (1-result)*kaggle_clip_log(1.0 - pred))


# In[15]:


tourney_results["LogLoss_Dummy"] = kaggle_log_loss(tourney_results["Pred_Dummy"], tourney_results["Truth"])
tourney_results["LogLoss_Seed"] = kaggle_log_loss(tourney_results["Pred_Seed"], tourney_results["Truth"])
tourney_results["LogLoss_Elo"] = kaggle_log_loss(tourney_results["Pred_Elo"], tourney_results["Truth"])

tourney_results.groupby('Season').agg({'LogLoss_Dummy' : 'mean', 'LogLoss_Seed' : 'mean', 
                                       "LogLoss_Elo" : "mean"})


# In[16]:


df_plt = tourney_results.groupby('Season').agg({'LogLoss_Dummy' : 'mean', 'LogLoss_Seed' : 'mean', 
                                                "LogLoss_Elo" : "mean"})

plt.plot(df_plt.LogLoss_Seed, label="Seed")
plt.plot(df_plt.LogLoss_Elo, label="Elo")
plt.axhline(df_plt[["LogLoss_Seed", "LogLoss_Elo"]].values.mean())
plt.legend()
plt.show()


# In[17]:


submission = pd.read_csv(os.path.join(data_directory,'../MSampleSubmissionStage1_2020.csv'))

# parse year and the two teams out of it
submission['season'] = [int(x[0:4]) for x in submission['ID']]
submission['lteam'] = [int(x[5:9]) for x in submission['ID']]
submission['gteam'] = [int(x[10:14]) for x in submission['ID']]

submission["Pred"] =     submission.apply(lambda row: 
                          predict_elo(row['season'], row['lteam'], row["gteam"]), 
                          axis=1)

submission


# In[18]:


submission[["ID", "Pred"]].to_csv(os.path.join(data_directory, "/kaggle/working/my_submission.csv"), index=False)

