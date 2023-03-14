#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
# This kaggle kernel uses an NCAA predictions file plus seed and slot data for a tournament
# year and simulates the tournament multiple times. In a simulated match-up, the prediction
# for the match-up is used to randomly pick the team to advance.
#
# The proportion of time each team advances out of a given round is stored in a csv file after
# all the simulations are complete. Advancing out of round 2 means the team makes the Sweet 16, 
# advancing out of round 4 means the team makes the Final Four, and advancing out of round 6 means 
# the team has won the championship. For the Men's tournament, Round 0 is the First Four.
#
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

###
### Configure the run.
### DATA_DIR should be the directory containing the competition data files. (Make sure it is Stage 2)
###    if you are doing the current year). 
### 
### PREDICTIONS_FILE should be the path to your predictions, in the competition prediction format
###
FILE_PREFIX = "W"  # Empty for Men, 'W' for Women
SEASON = 2019
ITERATIONS = 1000  # Number of times to simulate the tournament
DATA_DIR = os.path.join("..", "input", "stage2" + FILE_PREFIX.lower() + "datafiles")
PREDICTIONS_FILE = os.path.join("..", "input", FILE_PREFIX + "SampleSubmissionStage2.csv")
###
###
###

print(os.listdir("../input"))
print(os.listdir(DATA_DIR))
print()

# Verify the input files are available
SLOTS_FILE = os.path.join(DATA_DIR, FILE_PREFIX + "NCAATourneySlots.csv")
SEEDS_FILE = os.path.join(DATA_DIR, FILE_PREFIX + "NCAATourneySeeds.csv")
TEAMS_FILE = os.path.join(DATA_DIR, FILE_PREFIX + "Teams.csv")

# Make sure is file is present
for fname in [PREDICTIONS_FILE, SLOTS_FILE, SEEDS_FILE, TEAMS_FILE]:
    if os.path.isfile(fname):
        print("File found: {0}".format(fname))
    else:
        print("MISSING FILE: {0}".format(fname))


# In[2]:


# Our predictions
pred_df = pd.read_csv(PREDICTIONS_FILE)

# Turn the ID into something that can be used as in index to 
# directly look up a prediction
pred_df['ID'] = pred_df['ID'].apply(lambda x: x[5:])
pred_df.set_index('ID', inplace=True)

# Teams, indexed by TeamID
teams_df = pd.read_csv(TEAMS_FILE, index_col=0)

# Limit the seeds file to this season
seeds_df = pd.read_csv(SEEDS_FILE).query('Season == @SEASON')[['Seed', 'TeamID']].copy()
seeds_df.set_index('Seed', inplace=True)

# Men and women slots files are a little different
if FILE_PREFIX == "W":
    slots_df = pd.read_csv(SLOTS_FILE, index_col=0)
else:
    slots_df = pd.read_csv(SLOTS_FILE).query('Season == @SEASON')[['Slot', 'StrongSeed', 'WeakSeed']].copy()
    slots_df.set_index('Slot', inplace=True)
    
# For each slot, figure out what round it is. (First Four = Round 0)
slots_df['Round'] = slots_df.apply(lambda r: int(r.name[1:2]) if r.name.startswith("R") else 0, axis=1)

# Make space for a simulated winner
slots_df['WinnerTeamID'] = 0

print("Found {0} teams, {1} games, and {2} predictions.".format(seeds_df.shape[0], slots_df.shape[0], pred_df.shape[0]))


# In[3]:


# Create a dictionary per round to hold the number of times each team wins a game in that round
round_winners = {r: {} for r in slots_df.Round.unique()}
    
# Helper function to add a win for a team in a given round
def note_round_winner(the_round, TeamID):
    round_winners[the_round][TeamID] = 1 if TeamID not in round_winners[the_round]         else round_winners[the_round][TeamID] + 1
    
print("Expecting {0} rounds of games.".format(len(round_winners)))


# In[4]:


# Helper function to figure out the team that should be used
# for a particular matchup
def get_team_by_slot_reference(slotname_or_seed, seeds_df, tourney_df):
    
    if slotname_or_seed in tourney_df.index:
        # This refers to a game previously played; return the winner
        return tourney_df.loc[slotname_or_seed, "WinnerTeamID"]
    else:
        # If this is not another tournament slot, it must be the seed
        return seeds_df.loc[slotname_or_seed, "TeamID"]
    
start = time.time()

# Simulate the tournament many times
for iteration in range(ITERATIONS):
    
    # Make a random results variable for each game
    slots_df['RandomTrial'] = np.random.random(slots_df.shape[0])
    
    for row in slots_df.itertuples():
        # Get the team IDs.
        SID = get_team_by_slot_reference(row.StrongSeed, seeds_df, slots_df)
        WID = get_team_by_slot_reference(row.WeakSeed, seeds_df, slots_df)
        
        # Team A is always the lesser ID
        AID, BID = (SID, WID) if SID < WID else (WID, SID)
        
        # Get the prediction for the match-up and predict a winner
        pred_index = "{0}_{1}".format(AID, BID)
        A_likelihood = pred_df.loc[pred_index, 'Pred']
        winner = AID if row.RandomTrial < A_likelihood else BID
        slots_df.loc[row.Index, "WinnerTeamID"] = winner

        # Tabulate
        note_round_winner(row.Round, winner)
        
    if (iteration+1) % 100 == 0:
        print("+++ Iteration {0} complete after {1:0.1f} seconds.".format(iteration+1, time.time() - start))
        
print("Completed {0} simulations in {1:0.1f} seconds.".format(ITERATIONS, time.time() - start))


# In[5]:


def print_round_winners(round_num, max_num=68, title=None):
    
    this_round = round_winners[round_num]
    
    # Sort the round winners from highest to lowest by counts
    sorted_teams = sorted(this_round, key=lambda k: this_round[k], reverse=True)
    
    if title is None:
        print("Round {0} advance likelihood".format(round_num))
    else:
        print(title.format(round_num))
    
    count = 0
    for team in sorted_teams:    
        print("{0:16s} {1:0.3f}".format(teams_df.loc[team, 'TeamName'], this_round[team]/ITERATIONS)) 
        count += 1
        if count >= max_num:
            break

print_round_winners(6,8, title="Likelihood of winning the tournament:")
print()
print_round_winners(4,8, title="Likelihood of making the Final Four;")
print()
print_round_winners(2,16, title="Likelihood of making the Sweet 16:")


# In[6]:


# Convert results to a dataframe saved as csv
saved_filename = FILE_PREFIX + "{0}_Simul_{1}.csv".format(SEASON, ITERATIONS)
rwin_df = pd.DataFrame(round_winners) / ITERATIONS
rwin_df = rwin_df.join(teams_df['TeamName'])
rwin_df.index.name = 'TeamID'
rwin_df.to_csv(saved_filename)
print("Results saved to:", saved_filename)

