#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import matplotlib.pylab as plt
plt.style.use('seaborn-dark-palette')
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

DIR = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/'


# In[2]:


#Team Data
MTeams = pd.read_csv(f'{DIR}/MDataFiles_Stage1/MTeams.csv')
print(MTeams.shape)
print(MTeams.isnull().sum())
MTeams.head()


# In[3]:


#Season info
MSeasons = pd.read_csv(f'{DIR}/MDataFiles_Stage1/MSeasons.csv')
print(MSeasons.shape)
print(MSeasons.isnull().sum())
MSeasons.head()


# In[4]:


#Seeds Info
#separate the seeds and the conferences
MNCAATourneySeed = pd.read_csv(f'{DIR}/MDataFiles_Stage1/MNCAATourneySeeds.csv')
print(MNCAATourneySeed.shape)
print(MNCAATourneySeed.isnull().sum())
MNCAATourneySeeds = MNCAATourneySeed.merge(MTeams, how = 'left', left_on='TeamID', right_on='TeamID')
MNCAATourneySeeds['SeedConference'] = 'Region'+MNCAATourneySeeds['Seed'].str.slice(stop=1)
MNCAATourneySeeds['SeedOrder'] = MNCAATourneySeeds['Seed'].str.slice(start=1, stop=3).astype(int)
MNCAATourneySeeds.head(5)


# In[5]:


#double check to see if the seeds are in the rage of 1 and 16.
MNCAATourneySeeds['SeedOrder'].value_counts()


# In[6]:


#Regular Season --Team Year by Year
MRegularSeasonCompactResult = pd.read_csv(f'{DIR}/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
MRegularSeasonCompactResults = MRegularSeasonCompactResult.merge(MTeams[['TeamName','TeamID']], how='left', left_on='WTeamID', right_on='TeamID')                                .drop('TeamID', axis=1)                                .rename(columns={"TeamName":"WTeamName"})                                .merge(MTeams[['TeamName','TeamID']], how='left', left_on='LTeamID', right_on='TeamID')                                .drop('TeamID', axis=1)                                .rename(columns={"TeamName":"LTeamName"})
freq_win_yr = MRegularSeasonCompactResults.groupby(['Season','WTeamID','WTeamName'])['WTeamID'].count().sort_values(ascending=False)
freq_lose_yr = MRegularSeasonCompactResults.groupby(['Season','LTeamID','LTeamName'])['LTeamID'].count().sort_values(ascending=False)
MRegularSeasonTeamResultsYr = pd.concat([freq_win_yr, freq_lose_yr], axis=1)
print(MRegularSeasonTeamResultsYr.shape)
MRegularSeasonTeamResultsYr.fillna(0,inplace=True)
MRegularSeasonTeamResultsYr.index.set_names(['Season','TeamID','TeamName'],inplace=True)
MRegularSeasonTeamResultsYr.rename(columns={'WTeamID':'win','LTeamID':'loss'}, inplace=True)
MRegularSeasonTeamResultsYr['compact']=MRegularSeasonTeamResultsYr['win'] + MRegularSeasonTeamResultsYr['loss']
MRegularSeasonTeamResultsYr['WinRate'] = MRegularSeasonTeamResultsYr['win']/MRegularSeasonTeamResultsYr['compact']
MRegularSeasonTeamResultsYr.reset_index(inplace=True)
MRegularSeasonTeamResultsYr.head(10)


# In[7]:


MRegularSeasonTeamResultsYr.isnull().sum().to_frame(name = 'missing').T


# In[8]:


#Regular Season -- Game Details
MRegularSeasonDetail = pd.read_csv(f'{DIR}/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')
print(MRegularSeasonDetail.shape)
MRegularSeasonDetail.head()
print(MRegularSeasonDetail.columns)
wcol = [col for col in MRegularSeasonDetail if (col.startswith('W')) & (col !='WLoc') or (col=='Season')or (col=='DayNum') ]
print(wcol)
lcol = [col for col in MRegularSeasonDetail if col.startswith('L') or (col=='Season')or (col=='DayNum')]
print(lcol)


# In[9]:


rename = [w[1:] for w in wcol if w.startswith('W')]

wteam = MRegularSeasonDetail[wcol].copy()
wteam.columns = ['Season','DayNum']+rename
wteam['result']='W'
wteam['LScore']=MRegularSeasonDetail['LScore']
print(len(wteam))

lteam = MRegularSeasonDetail[lcol].copy()
lteam.columns = ['Season','DayNum']+rename
lteam['result']='L'
lteam['LScore']=MRegularSeasonDetail['WScore']
print(len(lteam))

MRegularSeasonDetails = pd.concat([wteam, lteam])
print(len(MRegularSeasonDetails))

MRegularSeasonDetails['FG_avg'] = MRegularSeasonDetails.FGM/MRegularSeasonDetails.FGA
MRegularSeasonDetails['FG3_avg'] = MRegularSeasonDetails.FGM3/MRegularSeasonDetails.FGA3
MRegularSeasonDetails['FGM2'] = MRegularSeasonDetails.FGM-MRegularSeasonDetails.FGM3
MRegularSeasonDetails['FGA2'] = MRegularSeasonDetails.FGA-MRegularSeasonDetails.FGA3
MRegularSeasonDetails['FG2_avg'] = MRegularSeasonDetails.FGM2/MRegularSeasonDetails.FGA2
MRegularSeasonDetails['FT_avg'] = MRegularSeasonDetails.FTM/MRegularSeasonDetails.FTA
MRegularSeasonDetails['TR'] = MRegularSeasonDetails.OR + MRegularSeasonDetails.DR
MRegularSeasonDetails.head(10)


# In[10]:


MRegularSeasonTeamBox = MRegularSeasonDetails.groupby(['Season','TeamID']).mean().reset_index()

MRegularSeasonTeamBox.head()


# In[11]:


#Public Rating
MMasseyOrdinals = pd.read_csv(f'{DIR}/MDataFiles_Stage1/MMasseyOrdinals.csv')
MMasseyOrdinals.sort_values(by=['Season', 'TeamID','SystemName','RankingDayNum'], inplace=True)
MMasseyOrdinals.head()


# In[12]:


prior_tourney = MMasseyOrdinals.query("RankingDayNum <= 133")
comb=prior_tourney.groupby(['Season','TeamID','SystemName']).size().reset_index().rename(columns={0:'count'})
comb.shape
max_rankingdaynum = prior_tourney.groupby(['Season','TeamID','SystemName']).agg({'RankingDayNum':'max'}).reset_index()
max_rankingdaynum.head()


# In[13]:


MMasseyOrdinalsPriorTourney = prior_tourney.merge(max_rankingdaynum, how='inner', left_on=['Season','TeamID','SystemName','RankingDayNum'], right_on=['Season','TeamID','SystemName','RankingDayNum'])
print(MMasseyOrdinalsPriorTourney.shape) #has to have 307393 combos
MMasseyOrdinalsPriorTourney.query('TeamID==1102 & Season==2003')


# In[14]:


MMasseyOrdinalsMedian = MMasseyOrdinalsPriorTourney.groupby(['Season','TeamID'])['OrdinalRank'].median().reset_index()
print(MMasseyOrdinalsMedian.shape)
MMasseyOrdinalsMedian.head()


# In[15]:


#Tournaments
MNCAATourneyCompactResult = pd.read_csv(f'{DIR}/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
print(MNCAATourneyCompactResult.shape)
print(MNCAATourneyCompactResult.isnull().sum())
MNCAATourneyCompactResults = MNCAATourneyCompactResult.merge(MTeams[['TeamName','TeamID']], how='left', left_on='WTeamID', right_on='TeamID')                                .drop('TeamID', axis=1)                                .rename(columns={"TeamName":"WTeamName"})                                .merge(MTeams[['TeamName','TeamID']], how='left', left_on='LTeamID', right_on='TeamID')                                .drop('TeamID', axis=1)                                .rename(columns={"TeamName":"LTeamName"})
MNCAATourneyCompactResults['Diff_Score'] = MNCAATourneyCompactResults['WScore'] - MNCAATourneyCompactResults['LScore']
MNCAATourneyCompactResults.sort_values(by='Diff_Score', ascending=False, inplace=True)
MNCAATourneyCompactResults.head()


# In[16]:


# decide the underdogs and collect the fields
MNCAATourneyCompactResults.sort_values(by=['Season','DayNum'], inplace=True)

MNCAATourney_ = MNCAATourneyCompactResults.merge(MMasseyOrdinalsMedian, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
MNCAATourney = MNCAATourney_.merge(MMasseyOrdinalsMedian, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
MNCAATourney.rename(columns={'OrdinalRank_x':'WTeamRank','OrdinalRank_y':'LTeamRank', 'TeamID_x':'T1','TeamID_y':'T2'}, inplace=True)
MNCAATourney.loc[MNCAATourney['WTeamRank'] < MNCAATourney['LTeamRank'], 'T1']=MNCAATourney['LTeamID']
MNCAATourney.loc[MNCAATourney['WTeamRank'] < MNCAATourney['LTeamRank'], 'T2']=MNCAATourney['WTeamID']
MNCAATourney['label'] = np.where(MNCAATourney['T1']==MNCAATourney['WTeamID'], 1,0)
print(MNCAATourney.shape)
MNCAATourney.tail()


# In[17]:


MNCAATourney[MNCAATourney.Season>=2003].head()


# In[18]:


def gen_TeamBoxDict(tag):
    TeamBoxDict = { k:tag+v for (k,v) in zip(MRegularSeasonTeamBox.columns, MRegularSeasonTeamBox.columns) if (k != 'Season' and k != 'DayNum')}  
    return(TeamBoxDict)


# In[19]:


def grab_col(tag,dataframe):
    df = dataframe.merge(MMasseyOrdinalsMedian, how='left', left_on=['Season', tag], right_on=['Season', 'TeamID'])        .rename(columns={'OrdinalRank':tag+'OrdinalRank'})        .merge(MNCAATourneySeeds[['Season', 'TeamID','SeedConference','SeedOrder']], how='left', left_on=['Season', tag], right_on=['Season', 'TeamID'])        .rename(columns={'SeedConference':tag+'SeedConference','SeedOrder':tag+'SeedOrder'})        .merge(MRegularSeasonTeamResultsYr[['Season','TeamID','WinRate']], how='left', left_on=['Season',tag], right_on=['Season','TeamID'])        .rename(columns={'WinRate':tag+'SeasonWinRate'})        .merge(MRegularSeasonTeamBox, how='left', left_on=['Season', tag], right_on=['Season', 'TeamID'])        .rename(columns=gen_TeamBoxDict(tag))        .rename(columns={'DayNum_x':'DayNum'})
    df.drop(columns=[col for col in df if col.startswith('TeamID')], inplace=True)
    df.drop(columns='DayNum_y', inplace=True)
    return df


# In[20]:


df_ = MNCAATourney.query('Season >= 2003')[['label','Season','DayNum','T1','T2']]
df1 = grab_col('T1',df_)
df2 = grab_col('T2',df_)
df = df1.merge(df2, how='inner',left_on=['label','Season','DayNum','T1','T2'],right_on=['label','Season','DayNum','T1','T2'])
df['OrdinalRankDiff']=df['T2OrdinalRank'] - df['T1OrdinalRank']
df['SeedOrderDiff']=df['T2SeedOrder'] - df['T1SeedOrder']
df.drop(columns={'T1SeedConference','T2SeedConference'}, inplace=True)
df.head()


# In[21]:


pd.set_option('display.max_rows', df.shape[0]+1)
df.isnull().sum().to_frame(name = 'missing')
pd.set_option('display.max_rows', 5)


# In[22]:


# separate traning and test sets
sc = StandardScaler()
df_training_ = df.loc[df.Season < 2015, df.columns != 'label']
label_training = df.loc[df.Season < 2015,'label']
df_training_.set_index(['Season','DayNum','T1','T2'], inplace=True)
df_training=pd.DataFrame(sc.fit_transform(df_training_),columns = df_training_.columns)
df_training.head()
df_training.describe()


# In[23]:


df_test_ = df.loc[df.Season >= 2015, df.columns != 'label']
label_test = df.loc[df.Season >= 2015,'label']
df_test_.set_index(['Season','DayNum','T1','T2'], inplace=True)
df_test = pd.DataFrame(sc.transform (df_test_),columns = df_test_.columns)
df_test.head()


# In[24]:


#check the defeat rate in the training data
label_training.value_counts(normalize=True)


# In[25]:


import sklearn
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import xgboost as xgb


# In[26]:


# Some useful parameters which will come in handy later on
ntrain = df_training.shape[0]
ntest = df_test.shape[0]
SEED = 20200325 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=100, shuffle=True)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)


# In[27]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    for i, (train_index, test_index) in enumerate(kf.split(oof_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[28]:


# Put in our parameters for said classifiers
# Random Forest parameters

# Logistic Regression 
lr_params = {'C': 1}

# Support Vector 
svc_params = {
    'kernel' : 'linear',
    'C' : 1
    }

# Random Forest
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': False, 
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}


# In[29]:


# Create 6 objects
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params=lr_params)


# In[30]:


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = label_training.ravel()
x_train = df_training.values # Creates an array of the train data
x_test = df_test.values # Creats an array of the test data


# In[31]:


# Create our OOF train and test predictions. These base results will be used as new features
lr_oof_train, lr_oof_test = get_oof(lr,x_train, y_train, x_test) # Logistic Regression Classifier
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

print("Training is complete")


# In[32]:


base_predictions_train = pd.DataFrame( {
    'LR': lr_oof_train.ravel(),
    'SVM': svc_oof_train.ravel(),
    'RandomForest': rf_oof_train.ravel(),
    'ExtraTrees': et_oof_train.ravel(),
    'AdaBoost': ada_oof_train.ravel(),
    'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()


# In[33]:


#form 2nd stage training/test sets
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train, lr_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test, lr_oof_test), axis=1)

#conduct 2nd level learning model via XGBoost
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
     n_estimators= 2000,
     max_depth= 4,
     min_child_weight= 2,
     #gamma=1,
     gamma=0.9,                        
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread= -1,
     scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# In[34]:


gbm.score(x_test, label_test)

