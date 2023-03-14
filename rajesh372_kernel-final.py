#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
from kaggle.competitions import nflrush
import tqdm
import re
from string import punctuation
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold
import matplotlib.patches as patches
import math
import seaborn as sns
from multiprocessing import Pool
import gc
from matplotlib import image
from sklearn.model_selection import train_test_split
from multiprocessing import Manager

import warnings
warnings.filterwarnings('ignore')


# In[2]:


env = nflrush.make_env()


# In[3]:


train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object',
                                       'Quarter': 'object',
                                       'Down': 'object'})
train['DefendersInTheBox_vs_Distance'] = train['DefendersInTheBox'] / train['Distance']


# In[4]:


convert_dict = {'GameId':int,'PlayId':int,'Team':object,'X':float,'Y':float,'S':float,'A':float,'Dis':float,'Orientation':float,
'Dir':float,'NflId':int,'DisplayName':object,'JerseyNumber':int,'Season':int,'YardLine':int,'Quarter':object,
'GameClock':object,'PossessionTeam':object,'Down':object,'Distance':int,'FieldPosition':object,'HomeScoreBeforePlay':int,
'VisitorScoreBeforePlay':int,'NflIdRusher':int,'OffenseFormation':object,'OffensePersonnel':object,'DefendersInTheBox':float,
'DefensePersonnel':object,'PlayDirection':object,'TimeHandoff':object,'TimeSnap':object,'PlayerHeight':object,'PlayerWeight':int,
'PlayerBirthDate':object,'PlayerCollegeName':object,'Position':object,'HomeTeamAbbr':object,'VisitorTeamAbbr':object,'Week':int,
'Stadium':object,'Location':object,'StadiumType':object,'Turf':object,'GameWeather':object,'Temperature':float,
'Humidity':float,'WindSpeed':object,'WindDirection':object}


convert_dict2 = {'GameId':int,'PlayId':int,'Team':object,'X':float,'Y':float,'S':float,'A':float,'Dis':float,'Orientation':float,
'Dir':float,'NflId':int,'DisplayName':object,'JerseyNumber':int,'Season':int,'YardLine':int,'Quarter':int,
'GameClock':object,'PossessionTeam':object,'Down':int,'Distance':int,'FieldPosition':object,'HomeScoreBeforePlay':int,
'VisitorScoreBeforePlay':int,'NflIdRusher':int,'OffenseFormation':object,'OffensePersonnel':object,'DefendersInTheBox':float,
'DefensePersonnel':object,'PlayDirection':object,'TimeHandoff':object,'TimeSnap':object,'PlayerHeight':object,'PlayerWeight':int,
'PlayerBirthDate':object,'PlayerCollegeName':object,'Position':object,'HomeTeamAbbr':object,'VisitorTeamAbbr':object,'Week':int,
'Stadium':object,'Location':object,'StadiumType':object,'Turf':object,'GameWeather':object,'Temperature':float,
'Humidity':float,'WindSpeed':object,'WindDirection':object}


# In[5]:


def clean_StadiumType(txt):
    if pd.isnull(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub(' +', ' ', txt)
    txt = txt.strip()
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('retractable', 'rtr.')
    return txt


# In[6]:


train['StadiumType'] = train['StadiumType'].apply(clean_StadiumType)


# In[7]:


Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 


# In[8]:


map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in train['PossessionTeam'].unique():
    map_abbr[abb] = abb


# In[9]:


train['PossessionTeam'] = train['PossessionTeam'].map(map_abbr)
train['HomeTeamAbbr'] = train['HomeTeamAbbr'].map(map_abbr)
train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].map(map_abbr)


# In[10]:


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans


# In[11]:


train['GameClock'] = train['GameClock'].apply(strtoseconds)


# In[12]:


train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))


# In[13]:


train['PlayerBMI'] = 703*(train['PlayerWeight']/(train['PlayerHeight'])**2)


# In[14]:


train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))


# In[15]:


train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)


# In[16]:


train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))


# In[17]:


seconds_in_year = 60*60*24*365.25
train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-
                                              row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)


# In[18]:


train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isnull(x) else x)


# In[19]:


train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isnull(x) and '-' in x else x)
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isnull(x) and type(x)!=float and 'gusts up to' in x else x)


# In[20]:


def str_to_float(txt):
    try:
        return float(txt)
    except:
        return -1


# In[21]:


train['WindSpeed'] = train['WindSpeed'].apply(str_to_float)


# In[22]:


def clean_WindDirection(txt):
    if pd.isnull(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = txt.replace('from', '')
    txt = txt.replace(' ', '')
    txt = txt.replace('north', 'n')
    txt = txt.replace('south', 's')
    txt = txt.replace('west', 'w')
    txt = txt.replace('east', 'e')
    return txt


# In[23]:


train['WindDirection'] = train['WindDirection'].apply(clean_WindDirection)


# In[24]:


def transform_WindDirection(txt):
    if pd.isnull(txt):
        return np.nan
    
    if txt=='n':
        return 0
    if txt=='nne' or txt=='nen':
        return 1/8
    if txt=='ne':
        return 2/8
    if txt=='ene' or txt=='nee':
        return 3/8
    if txt=='e':
        return 4/8
    if txt=='ese' or txt=='see':
        return 5/8
    if txt=='se':
        return 6/8
    if txt=='ses' or txt=='sse':
        return 7/8
    if txt=='s':
        return 8/8
    if txt=='ssw' or txt=='sws':
        return 9/8
    if txt=='sw':
        return 10/8
    if txt=='sww' or txt=='wsw':
        return 11/8
    if txt=='w':
        return 12/8
    if txt=='wnw' or txt=='nww':
        return 13/8
    if txt=='nw':
        return 14/8
    if txt=='nwn' or txt=='nnw':
        return 15/8
    return np.nan


# In[25]:


def clean_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub(' +', ' ', txt)
    txt = txt.strip()
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('retractable', 'retr.')
    txt = txt.replace('retr. roof - closed', 'retr roof closed')
    txt = txt.replace('retr. roof closed', 'retr roof closed')
    txt = txt.replace('outdoor retr roof-Open', 'retr roof open')
    txt = txt.replace('retr. roof - open', 'retr roof open')
    txt = txt.replace('retr. roof-open', 'retr roof open')
    txt = txt.replace('cloudy', 'outdoor')
    txt = txt.replace('dome', 'domed')
    return txt


# In[26]:


train['WindDirection'] = train['WindDirection'].apply(transform_WindDirection)


# In[27]:


train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x.strip() == 'right')


# In[28]:


train['GameWeather'] = train['GameWeather'].str.lower()
indoor = "indoor"
train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isnull(x) and indoor in x else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isnull(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isnull(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isnull(x) else x)


# In[29]:


diff_abbr = []
for x,y  in zip(sorted(train['HomeTeamAbbr'].unique()), sorted(train['PossessionTeam'].unique())):
    if x!=y:
        print(x + " " + y)


# In[30]:


map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in train['PossessionTeam'].unique():
    map_abbr[abb] = abb


# In[31]:


train['PossessionTeam'] = train['PossessionTeam'].map(map_abbr)
train['HomeTeamAbbr'] = train['HomeTeamAbbr'].map(map_abbr)
train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].map(map_abbr)


# In[32]:


train['HomePossesion'] = train['PossessionTeam'] == train['HomeTeamAbbr']


# In[33]:


train['Field_eq_Possession'] = train['FieldPosition'] == train['PossessionTeam']
train['HomeField'] = train['FieldPosition'] == train['HomeTeamAbbr']


# In[34]:


off_form = train['OffenseFormation'].unique()


# In[35]:


train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')

train['GameWeather'] = train['GameWeather'].str.lower()
indoor = "indoor"
train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)

train.loc[(train.loc[:,'FieldPosition'] == train.loc[:,'PossessionTeam']),'YardsFromOwnGoal'] = train.loc[(train.loc[:,'FieldPosition'] == train.loc[:,'PossessionTeam']),'YardLine'] 
train.loc[(train.loc[:,'FieldPosition'] != train.loc[:,'PossessionTeam']),'YardsFromOwnGoal'] = 50+ (50-train.loc[(train.loc[:,'FieldPosition'] != train.loc[:,'PossessionTeam']),'YardLine'] )
train.loc[(train.loc[:,'YardLine'] == 50),'YardsFromOwnGoal'] = 50

train.loc[(train.loc[:,'PlayDirection'] == True),'X_std'] = train.loc[(train.loc[:,'PlayDirection'] == True),'X'] - 10
train.loc[(train.loc[:,'PlayDirection'] == False),'X_std'] = 120 - train.loc[(train.loc[:,'PlayDirection'] == False),'X'] -10
train.loc[(train.loc[:,'PlayDirection'] == True),'Y_std'] = train.loc[(train.loc[:,'PlayDirection'] == True),'Y']
train.loc[(train.loc[:,'PlayDirection'] == False),'Y_std'] = round(160/3,2) - train.loc[(train.loc[:,'PlayDirection'] == False),'Y']

train.loc[((train.loc[:,'PlayDirection'] == False) & (train.loc[:,'Dir'] < 90)),'Dir_std_1'] = 360+ train.loc[((train.loc[:,'PlayDirection'] == False) & (train.loc[:,'Dir'] < 90)),'Dir']
train.loc[-(((train.loc[:,'PlayDirection'] == False) & (train.loc[:,'Dir'] < 90))),'Dir_std_1'] = train.loc[-(((train.loc[:,'PlayDirection'] == False) & (train.loc[:,'Dir'] < 90))),'Dir']
train.loc[((train.loc[:,'PlayDirection'] == True) & (train.loc[:,'Dir'] > 270)),'Dir_std_1'] = train.loc[((train.loc[:,'PlayDirection'] == True) & (train.loc[:,'Dir'] > 270)),'Dir'] - 360
train.loc[(train.loc[:,'PlayDirection'] == False),'Dir_std_2'] = train.loc[(train.loc[:,'PlayDirection'] == False),'Dir_std_1'] - 180
train.loc[-(train.loc[:,'PlayDirection'] == False),'Dir_std_2'] = train.loc[-(train.loc[:,'PlayDirection'] == False),'Dir_std_1']

train.loc[:,'X_std_end'] = train.loc[:,'S']*np.cos((90-train.loc[:,'Dir_std_2'])*np.pi/180)+train.loc[:,'X_std']
train.loc[:,'Y_std_end'] = train.loc[:,'S']*np.sin((90-train.loc[:,'Dir_std_2'])*np.pi/180)+train.loc[:,'Y_std']

train['HomeField'] = train['FieldPosition'] == train['HomeTeamAbbr']
train['YardsLeft'] = train.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
train['YardsLeft'] = train.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)

train.drop(train.index[(train['YardsLeft']<train['Yards']) | (train['YardsLeft']-100>train['Yards'])], inplace=True)

train.loc[:,'X_Force'] = round(train.loc[:,'PlayerWeight']*train.loc[:,'A']*np.cos((90-train.loc[:,'Dir_std_2'])*np.pi/180),2)
train.loc[:,'Y_Force'] = round(train.loc[:,'PlayerWeight']*train.loc[:,'A']*np.sin((90-train.loc[:,'Dir_std_2'])*np.pi/180),2)
train.loc[:,'X_Momentum'] = round(train.loc[:,'PlayerWeight']*train.loc[:,'S']*np.cos((90-train.loc[:,'Dir_std_2'])*np.pi/180),2)
train.loc[:,'Y_Momentum'] = round(train.loc[:,'PlayerWeight']*train.loc[:,'S']*np.sin((90-train.loc[:,'Dir_std_2'])*np.pi/180),2)
train.loc[:,'Y_Ang_Momentum'] = round((train.loc[:,'PlayerHeight']*.5)*train.loc[:,'PlayerWeight']*train.loc[:,'S']*np.sin((90-train.loc[:,'Dir_std_2'])*np.pi/180),2)
train.loc[:,'X_Ang_Momentum'] = round((train.loc[:,'PlayerHeight']*.5)*train.loc[:,'PlayerWeight']*train.loc[:,'S']*np.cos((90-train.loc[:,'Dir_std_2'])*np.pi/180),2)
train.loc[:,'X_A'] = round(train.loc[:,'A']*np.cos((90-train.loc[:,'Dir_std_2'])*np.pi/180),4)
train.loc[:,'Y_A'] = round(train.loc[:,'A']*np.sin((90-train.loc[:,'Dir_std_2'])*np.pi/180),4)
train.loc[:,'X_S'] = round(train.loc[:,'S']*np.cos((90-train.loc[:,'Dir_std_2'])*np.pi/180),4)
train.loc[:,'Y_S'] = round(train.loc[:,'S']*np.sin((90-train.loc[:,'Dir_std_2'])*np.pi/180),4)
train.loc[:,'Cos_O'] = round(np.cos((90-train.loc[:,'Orientation'])*np.pi/180),4)
train.loc[:,'Sin_O'] = round(np.sin((90-train.loc[:,'Orientation'])*np.pi/180),4)


# In[36]:


train_x = train[['PlayId','Team','PlayerHeight','PlayerWeight','PlayerAge','PlayerBMI',
'X_std', 'Y_std', 'Dir_std_2', 'X_std_end', 'Y_std_end','X_Force', 'Y_Force', 'X_Momentum', 'Y_Momentum',
       'Y_Ang_Momentum', 'X_Ang_Momentum', 'X_A', 'Y_A', 'X_S', 'Y_S', 'Cos_O',
       'Sin_O']].round(4)


# In[37]:


scale_cols = ['PlayerHeight','PlayerWeight','PlayerAge','PlayerBMI',
'X_std', 'Y_std', 'Dir_std_2', 'X_std_end', 'Y_std_end','X_Force', 'Y_Force', 'X_Momentum', 'Y_Momentum',
       'Y_Ang_Momentum', 'X_Ang_Momentum', 'X_A', 'Y_A', 'X_S', 'Y_S', 'Cos_O',
       'Sin_O']


# In[38]:


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(train_x.loc[:,scale_cols])
train_x.loc[:,scale_cols] = imp.transform(train_x.loc[:,scale_cols])


# In[39]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

train_x.loc[:,scale_cols] = ss.fit_transform(train_x.loc[:,scale_cols])


# In[40]:


train_x.loc[:,scale_cols] = train_x.loc[:,scale_cols].round(4)
train_l_home = train_x.loc[(train_x.loc[:,'Team'] == True),:].groupby(['PlayId','Team']).agg(lambda x: list(x))
train_l_away = train_x.loc[(train_x.loc[:,'Team'] == False),:].groupby(['PlayId','Team']).agg(lambda x: list(x))


# In[41]:


final_cats = ['PlayId','PossessionTeam','Down','Distance','FieldPosition','OffenseFormation','OffensePersonnel','DefensePersonnel',
       'HomeTeamAbbr','VisitorTeamAbbr','Week',
       'Stadium','Location','StadiumType','Turf','GameWeather']
final_cats2 = ['PossessionTeam','Down','Distance','FieldPosition','OffenseFormation','OffensePersonnel','DefensePersonnel',
       'HomeTeamAbbr','VisitorTeamAbbr','Week',
       'Stadium','Location','StadiumType','Turf','GameWeather']


# In[42]:


X_train_categorical = train.loc[:,final_cats].drop_duplicates()


# In[43]:


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans

def strtofloat(x):
    try:
        return float(x)
    except:
        return -1

def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0

def OffensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def DefensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x/15))
    except:
        return "nan"
    
    
def preprocess(df):
    
    train = df[['PlayId','GameId','WindSpeed','GameWeather','Turf','OffenseFormation','OffensePersonnel','DefensePersonnel','HomeScoreBeforePlay', 'VisitorScoreBeforePlay']]
    ## WindSpeed
    train['WindSpeed_ob'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

    ## Weather
    train['GameWeather_process'] = train['GameWeather'].str.lower()
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    #train['GameWeather_dense'] = train['GameWeather_process'].apply(map_weather)

    ## Turf
    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 
    train['Turf'] = train['Turf'].map(Turf)

    ## OffensePersonnel
    temp = train["OffensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(OffensePersonnelSplit(x)))
    temp.columns = ["Offense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on = "PlayId")

    ## DefensePersonnel
    temp = train["DefensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(DefensePersonnelSplit(x)))
    temp.columns = ["Defense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on = "PlayId")
    train = train.drop_duplicates()
    
    ## diff Score
    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]

    
    return train


# In[44]:


def create_features(df, deploy=False):
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)

    def new_orientation(angle, play_direction):
        if play_direction == 'left':
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return new_angle
        else:
            return angle

    def euclidean_distance(x1,y1,x2,y2):
        x_diff = (x1-x2)**2
        y_diff = (y1-y2)**2

        return np.sqrt(x_diff + y_diff)

    def back_direction(orientation):
        if orientation > 180.0:
            return 1
        else:
            return 0

    def update_yardline(df):
        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
        new_yardline = new_yardline[['GameId','PlayId','YardLine']]

        return new_yardline

    def update_orientation(df, yardline):
        df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
        df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
        df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

        df = df.drop('YardLine', axis=1)
        df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

        return df

    def back_features(df):
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X':'back_X',
                                            'Y':'back_Y'})
        carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']]

        return carriers

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])                                         .agg({'dist_to_back':['min','max','mean','std']})                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']

        return player_distance

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        defense = defense.groupby(['GameId','PlayId'])                         .agg({'def_dist_to_back':['min','max','mean','std']})                         .reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

        return defense
    
    def offensive_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        offense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        offense = offense[offense['Team'] == offense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        offense['off_dist_to_back'] = offense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        offense = offense.groupby(['GameId','PlayId'])                         .agg({'off_dist_to_back':['min','max','mean','std']})                         .reset_index()
        offense.columns = ['GameId','PlayId','off_min_dist','off_max_dist','off_mean_dist','off_std_dist']

        return offense

    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox']].drop_duplicates()
        static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))

        return static_features
    
    def split_personnel(s):
        splits = s.split(',')
        for i in range(len(splits)):
            splits[i] = splits[i].strip()

        return splits

    def defense_formation(l):
        dl = 0
        lb = 0
        db = 0
        other = 0

        for position in l:
            sub_string = position.split(' ')
            if sub_string[1] == 'DL':
                dl += int(sub_string[0])
            elif sub_string[1] in ['LB','OL']:
                lb += int(sub_string[0])
            else:
                db += int(sub_string[0])

        counts = (dl,lb,db,other)

        return counts

    def offense_formation(l):
        qb = 0
        rb = 0
        wr = 0
        te = 0
        ol = 0

        sub_total = 0
        qb_listed = False
        for position in l:
            sub_string = position.split(' ')
            pos = sub_string[1]
            cnt = int(sub_string[0])

            if pos == 'QB':
                qb += cnt
                sub_total += cnt
                qb_listed = True
            # Assuming LB is a line backer lined up as full back
            elif pos in ['RB','LB']:
                rb += cnt
                sub_total += cnt
            # Assuming DB is a defensive back and lined up as WR
            elif pos in ['WR','DB']:
                wr += cnt
                sub_total += cnt
            elif pos == 'TE':
                te += cnt
                sub_total += cnt
            # Assuming DL is a defensive lineman lined up as an additional line man
            else:
                ol += cnt
                sub_total += cnt

        # If not all 11 players were noted at given positions we need to make some assumptions
        # I will assume if a QB is not listed then there was 1 QB on the play
        # If a QB is listed then I'm going to assume the rest of the positions are at OL
        # This might be flawed but it looks like RB, TE and WR are always listed in the personnel
        if sub_total < 11:
            diff = 11 - sub_total
            if not qb_listed:
                qb += 1
                diff -= 1
            ol += diff

        counts = (qb,rb,wr,te,ol)

        return counts
    
    def combine_features(relative_to_back, defense, offense, static, prep, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,offense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,prep,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    prep = preprocess(df)
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    off_feats = offensive_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, off_feats, static_feats, prep, deploy=deploy)
    
    return basetable


# In[45]:


new_continuos =  ['PlayId',
 'back_from_scrimmage',
 'back_oriented_down_field',
 'back_moving_down_field',
 'min_dist',
 'max_dist',
 'mean_dist',
 'std_dist',
 'def_min_dist',
 'def_max_dist',
 'def_mean_dist',
 'def_std_dist',
 'off_min_dist',
 'off_max_dist',
 'off_mean_dist',
 'off_std_dist',
 'OffenseDB',
 'OffenseDL',
 'OffenseLB',
 'OffenseOL',
 'OffenseQB',
 'OffenseRB',
 'OffenseTE',
 'OffenseWR',
 'DefenseDB',
 'DefenseDL',
 'DefenseLB',
 'DefenseOL',
 'diffScoreBeforePlay']


# In[46]:


train2 = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
outcomes = train2[['GameId','PlayId','Yards']].drop_duplicates()
train_basetable = create_features(train2, False)


# In[47]:


new_continuos_df = train_basetable[new_continuos].drop_duplicates()


# In[48]:


final_continuous = ['PlayId','YardsFromOwnGoal','TimeDelta','DefendersInTheBox_vs_Distance',
'Temperature','Humidity','WindSpeed','WindDirection','DefendersInTheBox',
'Quarter','GameClock','HomeScoreBeforePlay','VisitorScoreBeforePlay']
X_train_continuous = train[final_continuous].drop_duplicates()


# In[49]:


X_train_continuous = X_train_continuous.merge(new_continuos_df, on='PlayId')


# In[50]:


scale_cols2 = ['YardsFromOwnGoal','TimeDelta','DefendersInTheBox_vs_Distance',
'Temperature','Humidity','WindSpeed','WindDirection','DefendersInTheBox',
'Quarter','GameClock','HomeScoreBeforePlay','VisitorScoreBeforePlay',
'back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
'min_dist','max_dist','mean_dist','std_dist','def_min_dist','def_max_dist',
'def_mean_dist','def_std_dist','off_min_dist','off_max_dist','off_mean_dist','off_std_dist',
'OffenseDB','OffenseDL','OffenseLB','OffenseOL','OffenseQB','OffenseRB','OffenseTE','OffenseWR',
'DefenseDB','DefenseDL','DefenseLB','DefenseOL','diffScoreBeforePlay']

# scale_cols2 = ['YardsFromOwnGoal','TimeDelta','DefendersInTheBox_vs_Distance',
# 'Temperature','Humidity','WindSpeed','WindDirection','DefendersInTheBox',
# 'Quarter','GameClock','HomeScoreBeforePlay','VisitorScoreBeforePlay']

X_train_continuous.loc[:,scale_cols2] = ss.fit_transform(X_train_continuous.loc[:,scale_cols2])


# In[51]:


y_train = train.loc[:,['PlayId','Yards']].drop_duplicates()


# In[52]:


final_train = y_train.merge(X_train_continuous, on='PlayId')
final_train = final_train.merge(X_train_categorical, on='PlayId')
final_train = final_train.merge(train_l_home, on='PlayId')
final_train = final_train.merge(train_l_away, on='PlayId')


# In[53]:


series_cols = ['PlayerHeight_x', 'PlayerWeight_x', 'PlayerAge_x',
       'PlayerBMI_x', 'X_std_x', 'Y_std_x', 'Dir_std_2_x', 'X_std_end_x',
       'Y_std_end_x', 'X_Force_x', 'Y_Force_x', 'X_Momentum_x', 'Y_Momentum_x',
       'Y_Ang_Momentum_x', 'X_Ang_Momentum_x', 'X_A_x', 'Y_A_x', 'X_S_x',
       'Y_S_x', 'Cos_O_x', 'Sin_O_x', 'PlayerHeight_y', 'PlayerWeight_y',
       'PlayerAge_y', 'PlayerBMI_y', 'X_std_y', 'Y_std_y', 'Dir_std_2_y',
       'X_std_end_y', 'Y_std_end_y', 'X_Force_y', 'Y_Force_y', 'X_Momentum_y',
       'Y_Momentum_y', 'Y_Ang_Momentum_y', 'X_Ang_Momentum_y', 'X_A_y',
       'Y_A_y', 'X_S_y', 'Y_S_y', 'Cos_O_y', 'Sin_O_y']


# In[54]:


X_train_continuous = final_train[scale_cols2]
X_train_categorical = final_train[final_cats2]
X_train_series = final_train[series_cols]
y_train = final_train['Yards']
y_train = pd.DataFrame(y_train)


# In[55]:


X_train_series = X_train_series.values.tolist()
X_train_series = np.array(X_train_series)


# In[56]:


Y_train = np.zeros(shape=(y_train.shape[0], 199))
for i,yard in enumerate(y_train['Yards'][::1]):
    Y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))


# In[57]:


class EmbeddingMapping():
    """
    Helper class for handling categorical variables
    
    An instance of this class should be defined for each categorical variable we want to use.
    """
    def __init__(self, series):
        # get a list of unique values
        values = series.unique().tolist()
        
        # Set a dictionary mapping from values to integer value
        # In our example this will be {'Mercaz': 1, 'Old North': 2, 'Florentine': 3}
        self.embedding_dict = {value: int_value+1 for int_value, value in enumerate(values)}
        
        # The num_values will be used as the input_dim when defining the embedding layer. 
        # It will also be returned for unseen values 
        self.num_values = len(values) + 1

    def get_mapping(self, value):
        # If the value was seen in the training set, return its integer mapping
        if value in self.embedding_dict:
            return self.embedding_dict[value]
        
        # Else, return the same integer for unseen values
        else:
            return self.num_values


# In[58]:


PossessionTeam = EmbeddingMapping(X_train_categorical['PossessionTeam'])
Down = EmbeddingMapping(X_train_categorical['Down'])
Distance = EmbeddingMapping(X_train_categorical['Distance'])
FieldPosition = EmbeddingMapping(X_train_categorical['FieldPosition'])
OffenseFormation = EmbeddingMapping(X_train_categorical['OffenseFormation'])
OffensePersonnel = EmbeddingMapping(X_train_categorical['OffensePersonnel'])
DefensePersonnel = EmbeddingMapping(X_train_categorical['DefensePersonnel'])
#PlayerCollegeName = EmbeddingMapping(X_train_categorical['PlayerCollegeName'])
#Position = EmbeddingMapping(X_train_categorical['0_Position'])
HomeTeamAbbr = EmbeddingMapping(X_train_categorical['HomeTeamAbbr'])
VisitorTeamAbbr = EmbeddingMapping(X_train_categorical['VisitorTeamAbbr'])
Week = EmbeddingMapping(X_train_categorical['Week'])
Stadium = EmbeddingMapping(X_train_categorical['Stadium'])
Location = EmbeddingMapping(X_train_categorical['Location'])
StadiumType = EmbeddingMapping(X_train_categorical['StadiumType'])
Turf = EmbeddingMapping(X_train_categorical['Turf'])
GameWeather = EmbeddingMapping(X_train_categorical['GameWeather'])
#HomePossesion = EmbeddingMapping(X_train_categorical['0_HomePossesion'])
#HomeField = EmbeddingMapping(X_train_categorical['0_HomeField'])


# In[59]:


X_train_cat_mapping = pd.DataFrame()
X_train_cat_mapping = X_train_cat_mapping.assign(#PossessionTeam = X_train_categorical['PossessionTeam'].apply(PossessionTeam.get_mapping),
                                                Down = X_train_categorical['Down'].apply(Down.get_mapping),
                                                Distance = X_train_categorical['Distance'].apply(Distance.get_mapping),
                                                FieldPosition = X_train_categorical['FieldPosition'].apply(FieldPosition.get_mapping),
                                                OffenseFormation = X_train_categorical['OffenseFormation'].apply(OffenseFormation.get_mapping),
                                                OffensePersonnel = X_train_categorical['OffensePersonnel'].apply(OffensePersonnel.get_mapping),
                                                DefensePersonnel = X_train_categorical['DefensePersonnel'].apply(DefensePersonnel.get_mapping),
                                                #PlayerCollegeName = X_train_categorical['0_PlayerCollegeName'].apply(PlayerCollegeName.get_mapping),
                                                #Position = X_train_categorical['0_Position'].apply(Position.get_mapping),
                                                HomeTeamAbbr = X_train_categorical['HomeTeamAbbr'].apply(HomeTeamAbbr.get_mapping),
                                                VisitorTeamAbbr = X_train_categorical['VisitorTeamAbbr'].apply(VisitorTeamAbbr.get_mapping)
                                                #Week = X_train_categorical['Week'].apply(Week.get_mapping),
                                                #Stadium = X_train_categorical['Stadium'].apply(Stadium.get_mapping),
                                                #Location = X_train_categorical['Location'].apply(Location.get_mapping),
                                                #StadiumType = X_train_categorical['StadiumType'].apply(StadiumType.get_mapping),
                                                #Turf = X_train_categorical['Turf'].apply(Turf.get_mapping),
                                                #GameWeather = X_train_categorical['GameWeather'].apply(GameWeather.get_mapping)
                                                #HomePossesion = X_train_categorical['0_HomePossesion'].apply(HomePossesion.get_mapping),
                                                #HomeField = X_train_categorical['0_HomeField'].apply(HomeField.get_mapping)
                                                )
X_train_cat_mapping = np.array(X_train_cat_mapping)


# In[60]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train_continuous)
X_train_continuous = imp.transform(X_train_continuous)


# In[61]:


def crps(y_true, y_pred):
    return K.mean(K.square(y_true - K.cumsum(y_pred, axis=1)), axis=1)


# In[62]:


p = .1

numeric_inputs = tf.keras.layers.Input((40,), name='numeric_inputs')
cat_inputs = tf.keras.layers.Input((8,), name='cat_inputs')
series_inputs = tf.keras.layers.Input((42,11), name='series_inputs')
lstm_inputs = tf.keras.layers.Input((42,11), name='lstm_inputs')
#img_inputs = tf.keras.layers.Input(shape= (100, 150, 3), name='img_inputs')

#sx = tf.keras.layers.Dropout(.6)(series_inputs)
sx = tf.keras.layers.Conv1D(42, 1)(series_inputs)
#sx = tf.keras.layers.PReLU()(sx)
sx = tf.keras.layers.LeakyReLU()(sx)


global_ave = tf.keras.layers.GlobalAveragePooling1D()(sx)
global_max = tf.keras.layers.GlobalMaxPool1D()(sx)
sx = tf.keras.layers.Concatenate()([global_ave, global_max])
sx = tf.keras.layers.BatchNormalization()(sx)
sx = tf.keras.layers.Flatten()(sx)


sxlstm = tf.keras.layers.Dropout(.1)(lstm_inputs)
sxlstm = tf.keras.layers.LSTM(64,return_sequences=True)(sxlstm)
sxlstm = tf.keras.layers.LSTM(64,return_sequences=True)(sxlstm)
# sxlstm = tf.keras.layers.LSTM(128,return_sequences=True)(sxlstm)
# sxlstm = tf.keras.layers.LSTM(128,return_sequences=True)(sxlstm)
sxlstm = tf.keras.layers.LSTM(64)(sxlstm)
#sxlstm = tf.keras.layers.PReLU()(sxlstm)
sxlstm = tf.keras.layers.LeakyReLU()(sxlstm)
sxlstm = tf.keras.layers.BatchNormalization()(sxlstm)
sxlstm = tf.keras.layers.Flatten()(sxlstm)



embedding_layer = tf.keras.layers.Embedding(137,8,input_length=8)
cats = embedding_layer(cat_inputs)
cats = tf.keras.layers.LSTM(64,return_sequences=True)(cats)
cats = tf.keras.layers.LSTM(64,return_sequences=True)(cats)
cats = tf.keras.layers.LSTM(64)(cats)
cats = tf.keras.layers.LeakyReLU()(cats)
cats = tf.keras.layers.BatchNormalization()(cats)
cats = tf.keras.layers.Flatten()(cats)

embedding_layer2 = tf.keras.layers.Embedding(137,8,input_length=8)
cats2 = embedding_layer(cat_inputs)
cats2 = tf.keras.layers.Flatten()(cats2)


# ix = tf.keras.layers.Conv2D(8, (3, 3),padding= 'same',activation='relu',data_format='channels_last')(img_inputs)
# #ix = tf.keras.layers.PReLU()(ix)
# global_ave_ix = tf.keras.layers.AveragePooling2D((2,2))(ix)
# global_max_ix = tf.keras.layers.MaxPooling2D((2,2))(ix)
# ix = tf.keras.layers.Concatenate()([global_ave_ix, global_max_ix])
# ix = tf.keras.layers.BatchNormalization()(ix)

# ix = tf.keras.layers.Conv2D(16, (3, 3),activation='relu')(ix)
# #ix = tf.keras.layers.PReLU()(ix)
# global_ave_ix2 = tf.keras.layers.AveragePooling2D((2,2))(ix)
# global_max_ix2 = tf.keras.layers.MaxPooling2D((2,2))(ix)
# ix = tf.keras.layers.Concatenate()([global_ave_ix2, global_max_ix2])
# ix = tf.keras.layers.BatchNormalization()(ix)

# ix = tf.keras.layers.Conv2D(16, (3, 3),activation='relu')(ix)
# #ix = tf.keras.layers.PReLU()(ix)
# global_ave_ix2 = tf.keras.layers.AveragePooling2D((2,2))(ix)
# global_max_ix2 = tf.keras.layers.MaxPooling2D((2,2))(ix)
# ix = tf.keras.layers.Concatenate()([global_ave_ix2, global_max_ix2])
# ix = tf.keras.layers.BatchNormalization()(ix)

# ix = tf.keras.layers.Conv2D(16, (3, 3),activation='relu')(ix)
# #ix = tf.keras.layers.PReLU()(ix)
# global_ave_ix2 = tf.keras.layers.AveragePooling2D((2,2))(ix)
# global_max_ix2 = tf.keras.layers.MaxPooling2D((2,2))(ix)
# ix = tf.keras.layers.Concatenate()([global_ave_ix2, global_max_ix2])
# ix = tf.keras.layers.BatchNormalization()(ix)

# ix = tf.keras.layers.Conv2D(16, (3, 3),activation='relu')(ix)
# #ix = tf.keras.layers.PReLU()(ix)
# global_ave_ix2 = tf.keras.layers.AveragePooling2D((2,2))(ix)
# global_max_ix2 = tf.keras.layers.MaxPooling2D((2,2))(ix)
# ix = tf.keras.layers.Concatenate()([global_ave_ix2, global_max_ix2])
# ix = tf.keras.layers.BatchNormalization()(ix)

# ix = tf.keras.layers.Flatten()(ix)


#x = tf.keras.layers.Concatenate()([cats, numeric_inputs,sx,ix,sxlstm])
x = tf.keras.layers.Concatenate()([cats,cats2,numeric_inputs,sxlstm,sx]) #cats sx
#x = tf.keras.layers.LeakyReLU()(x)
#x = tf.keras.layers.Dropout(.2)(x)
x = tf.keras.layers.LeakyReLU()(x)
#x = tf.keras.layers.PReLU()(x)
#x = tf.keras.layers.GaussianNoise(0.15)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(400)(x) #, activation='relu'

#x = tf.keras.layers.Concatenate()([cats, numeric_inputs,sx,ix,sxlstm])
#x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dropout(.3)(x)
x = tf.keras.layers.LeakyReLU()(x)
#x = tf.keras.layers.PReLU()(x)
#x = tf.keras.layers.GaussianNoise(0.15)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(400)(x) #, activation='relu'

# x = tf.keras.layers.Concatenate()([cats, numeric_inputs,sx,ix])
# #x = tf.keras.layers.LeakyReLU()(x)
# x = tf.keras.layers.Dropout(.2)(x)
# x = tf.keras.layers.PReLU()(x)
# #x = tf.keras.layers.GaussianNoise(0.15)(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dense(100)(x) #, activation='relu'

# x = tf.keras.layers.Concatenate()([cats, numeric_inputs,sx,ix])
# #x = tf.keras.layers.LeakyReLU()(x)
# x = tf.keras.layers.Dropout(.4)(x)
# x = tf.keras.layers.PReLU()(x)
# #x = tf.keras.layers.GaussianNoise(0.15)(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dense(100)(x) #, activation='relu'

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
#x = tf.keras.layers.PReLU()(x)
#x = tf.keras.activations.selu(x)
x = tf.keras.layers.Dropout(.3)(x)
#x = tf.keras.layers.GaussianNoise(0.12)(x)
x = tf.keras.layers.Dense(400)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
#x = tf.keras.layers.PReLU()(x)
#x = tf.keras.activations.selu(x)
x = tf.keras.layers.Dropout(.3)(x)
#x = tf.keras.layers.GaussianNoise(0.12)(x)
x = tf.keras.layers.Dense(400)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
#x = tf.keras.layers.PReLU()(x)
x = tf.keras.layers.Dropout(.2)(x)
#x = tf.keras.layers.GaussianNoise(0.6)(x)
out = tf.keras.layers.Dense(199, activation='softmax', name='output')(x)


#model = tf.keras.models.Model(inputs=[numeric_inputs, cat_inputs,series_inputs,lstm_inputs,img_inputs], outputs=out)
model = tf.keras.models.Model(inputs=[numeric_inputs, cat_inputs,series_inputs,lstm_inputs], outputs=out) #series_inputs
#sgd = tf.keras.optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.1, nesterov=False)
#rms = tf.keras.optimizers.RMSprop(learning_rate=0.02, rho=0.9)
#adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer= 'Adam',loss= crps ,metrics=['accuracy'])


# In[63]:


playid_array = np.array(final_train['PlayId'])


# In[64]:


xtc_train, xtc_test,xtca_train, xtca_test,xts_train, xts_test,xti_train, xti_test, y_train, y_test = train_test_split(X_train_continuous,X_train_cat_mapping,X_train_series,playid_array,Y_train, test_size=0.1, random_state=42)


# In[65]:


batch_size=320
def bootstrap_sample_generator(batch_size):
    while True:
        batch_idx = np.random.choice(xtc_train.shape[0], batch_size)
        yield ({'numeric_inputs': xtc_train[batch_idx],
        'cat_inputs': xtca_train[batch_idx],
        'series_inputs':xts_train[batch_idx],
       'lstm_inputs':xts_train[batch_idx]
        #'img_inputs': img_arry(xti_train[batch_idx])
               }, 
               {'output': y_train[batch_idx]})


# In[66]:


def bootstrap_sample_generator_test(batch_size):
    while True:
        batch_idx = np.random.choice(xtc_test.shape[0], batch_size)
        yield ({'numeric_inputs': xtc_test[batch_idx],
        'cat_inputs': xtca_test[batch_idx],
       'series_inputs':xts_test[batch_idx],
       'lstm_inputs':xts_test[batch_idx]
        #'img_inputs': img_arry(xti_test[batch_idx])
               }, 
               {'output': y_test[batch_idx]})


# In[67]:


er = EarlyStopping(patience=10, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
model.fit_generator(
    bootstrap_sample_generator(batch_size),
    steps_per_epoch= 70,
    epochs=50,use_multiprocessing = True,callbacks=[er],verbose=1
    ,validation_data = bootstrap_sample_generator_test(batch_size),validation_steps = 7 
)


# In[68]:


def make_pred(test, sample,env,model): 
    try:
        test2 = test
        test = test.astype(convert_dict) 
        test2 = test2.astype(convert_dict2)
        test['DefendersInTheBox_vs_Distance'] = test['DefendersInTheBox'] / test['Distance']
        test['StadiumType'] = test['StadiumType'].apply(clean_StadiumType)
        map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
        for abb in test['PossessionTeam'].unique():
            map_abbr[abb] = abb
        test['PossessionTeam'] = test['PossessionTeam'].map(map_abbr)
        test['HomeTeamAbbr'] = test['HomeTeamAbbr'].map(map_abbr)
        test['VisitorTeamAbbr'] = test['VisitorTeamAbbr'].map(map_abbr)
        test['GameClock'] = test['GameClock'].apply(strtoseconds)
        test['PlayerHeight'] = test['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
        test['PlayerBMI'] = 703*(test['PlayerWeight']/(test['PlayerHeight'])**2)
        test['TimeHandoff'] = test['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        test['TimeSnap'] = test['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        test['TimeDelta'] = test.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        test['PlayerBirthDate'] = test['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
        test['PlayerAge'] = test.apply(lambda row: (row['TimeHandoff']-
                                                  row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
        test['WindSpeed'] = test['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isnull(x) else x)
        test['WindSpeed'] = test['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isnull(x) and '-' in x else x)
        test['WindSpeed'] = test['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isnull(x) and type(x)!=float and 'gusts up to' in x else x)
        test['WindSpeed'] = test['WindSpeed'].apply(str_to_float)
        test['WindDirection'] = test['WindDirection'].apply(clean_WindDirection)
        test['WindDirection'] = test['WindDirection'].apply(transform_WindDirection)
        test['PlayDirection'] = test['PlayDirection'].apply(lambda x: x.strip() == 'right')
        test['GameWeather'] = test['GameWeather'].str.lower()
        indoor = "indoor"
        test['GameWeather'] = test['GameWeather'].apply(lambda x: indoor if not pd.isnull(x) and indoor in x else x)
        test['GameWeather'] = test['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isnull(x) else x)
        test['GameWeather'] = test['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isnull(x) else x)
        test['GameWeather'] = test['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isnull(x) else x)
        map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
        for abb in test['PossessionTeam'].unique():
            map_abbr[abb] = abb
        test['PossessionTeam'] = test['PossessionTeam'].map(map_abbr)
        test['HomeTeamAbbr'] = test['HomeTeamAbbr'].map(map_abbr)
        test['VisitorTeamAbbr'] = test['VisitorTeamAbbr'].map(map_abbr)
        test['HomePossesion'] = test['PossessionTeam'] == test['HomeTeamAbbr']
        test['Field_eq_Possession'] = test['FieldPosition'] == test['PossessionTeam']
        test['HomeField'] = test['FieldPosition'] == test['HomeTeamAbbr']
        test['Team'] = test['Team'].apply(lambda x: x.strip()=='home')

        test['GameWeather'] = test['GameWeather'].str.lower()
        indoor = "indoor"
        test['GameWeather'] = test['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
        test['GameWeather'] = test['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
        test['GameWeather'] = test['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        test['GameWeather'] = test['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)

        test.loc[(test.loc[:,'FieldPosition'] == test.loc[:,'PossessionTeam']),'YardsFromOwnGoal'] = test.loc[(test.loc[:,'FieldPosition'] == test.loc[:,'PossessionTeam']),'YardLine'] 
        test.loc[(test.loc[:,'FieldPosition'] != test.loc[:,'PossessionTeam']),'YardsFromOwnGoal'] = 50+ (50-test.loc[(test.loc[:,'FieldPosition'] != test.loc[:,'PossessionTeam']),'YardLine'] )
        test.loc[(test.loc[:,'YardLine'] == 50),'YardsFromOwnGoal'] = 50

        test.loc[(test.loc[:,'PlayDirection'] == True),'X_std'] = test.loc[(test.loc[:,'PlayDirection'] == True),'X'] - 10
        test.loc[(test.loc[:,'PlayDirection'] == False),'X_std'] = 120 - test.loc[(test.loc[:,'PlayDirection'] == False),'X'] -10
        test.loc[(test.loc[:,'PlayDirection'] == True),'Y_std'] = test.loc[(test.loc[:,'PlayDirection'] == True),'Y']
        test.loc[(test.loc[:,'PlayDirection'] == False),'Y_std'] = round(160/3,2) - test.loc[(test.loc[:,'PlayDirection'] == False),'Y']

        test.loc[((test.loc[:,'PlayDirection'] == False) & (test.loc[:,'Dir'] < 90)),'Dir_std_1'] = 360+ test.loc[((test.loc[:,'PlayDirection'] == False) & (test.loc[:,'Dir'] < 90)),'Dir']
        test.loc[-(((test.loc[:,'PlayDirection'] == False) & (test.loc[:,'Dir'] < 90))),'Dir_std_1'] = test.loc[-(((test.loc[:,'PlayDirection'] == False) & (test.loc[:,'Dir'] < 90))),'Dir']
        test.loc[((test.loc[:,'PlayDirection'] == True) & (test.loc[:,'Dir'] > 270)),'Dir_std_1'] = test.loc[((test.loc[:,'PlayDirection'] == True) & (test.loc[:,'Dir'] > 270)),'Dir'] - 360
        test.loc[(test.loc[:,'PlayDirection'] == False),'Dir_std_2'] = test.loc[(test.loc[:,'PlayDirection'] == False),'Dir_std_1'] - 180
        test.loc[-(test.loc[:,'PlayDirection'] == False),'Dir_std_2'] = test.loc[-(test.loc[:,'PlayDirection'] == False),'Dir_std_1']

        test.loc[:,'X_std_end'] = test.loc[:,'S']*np.cos((90-test.loc[:,'Dir_std_2'])*np.pi/180)+test.loc[:,'X_std']
        test.loc[:,'Y_std_end'] = test.loc[:,'S']*np.sin((90-test.loc[:,'Dir_std_2'])*np.pi/180)+test.loc[:,'Y_std']

        test['HomeField'] = test['FieldPosition'] == test['HomeTeamAbbr']
        test['YardsLeft'] = test.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
        test['YardsLeft'] = test.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)

        #test.drop(test.index[(test['YardsLeft']<test['Yards']) | (test['YardsLeft']-100>test['Yards'])], inplace=True)

        test.loc[:,'X_Force'] = round(test.loc[:,'PlayerWeight']*test.loc[:,'A']*np.cos((90-test.loc[:,'Dir_std_2'])*np.pi/180),2)
        test.loc[:,'Y_Force'] = round(test.loc[:,'PlayerWeight']*test.loc[:,'A']*np.sin((90-test.loc[:,'Dir_std_2'])*np.pi/180),2)
        test.loc[:,'X_Momentum'] = round(test.loc[:,'PlayerWeight']*test.loc[:,'S']*np.cos((90-test.loc[:,'Dir_std_2'])*np.pi/180),2)
        test.loc[:,'Y_Momentum'] = round(test.loc[:,'PlayerWeight']*test.loc[:,'S']*np.sin((90-test.loc[:,'Dir_std_2'])*np.pi/180),2)
        test.loc[:,'Y_Ang_Momentum'] = round((test.loc[:,'PlayerHeight']*.5)*test.loc[:,'PlayerWeight']*test.loc[:,'S']*np.sin((90-test.loc[:,'Dir_std_2'])*np.pi/180),2)
        test.loc[:,'X_Ang_Momentum'] = round((test.loc[:,'PlayerHeight']*.5)*test.loc[:,'PlayerWeight']*test.loc[:,'S']*np.cos((90-test.loc[:,'Dir_std_2'])*np.pi/180),2)
        test.loc[:,'X_A'] = round(test.loc[:,'A']*np.cos((90-test.loc[:,'Dir_std_2'])*np.pi/180),4)
        test.loc[:,'Y_A'] = round(test.loc[:,'A']*np.sin((90-test.loc[:,'Dir_std_2'])*np.pi/180),4)
        test.loc[:,'X_S'] = round(test.loc[:,'S']*np.cos((90-test.loc[:,'Dir_std_2'])*np.pi/180),4)
        test.loc[:,'Y_S'] = round(test.loc[:,'S']*np.sin((90-test.loc[:,'Dir_std_2'])*np.pi/180),4)
        test.loc[:,'Cos_O'] = round(np.cos((90-test.loc[:,'Orientation'])*np.pi/180),4)
        test.loc[:,'Sin_O'] = round(np.sin((90-test.loc[:,'Orientation'])*np.pi/180),4)
        test_x = test[['PlayId','Team','PlayerHeight','PlayerWeight','PlayerAge','PlayerBMI',
        'X_std', 'Y_std', 'Dir_std_2', 'X_std_end', 'Y_std_end','X_Force', 'Y_Force', 'X_Momentum', 'Y_Momentum',
           'Y_Ang_Momentum', 'X_Ang_Momentum', 'X_A', 'Y_A', 'X_S', 'Y_S', 'Cos_O',
           'Sin_O']].round(4)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(test_x.loc[:,scale_cols])
        test_x.loc[:,scale_cols] = imp.transform(test_x.loc[:,scale_cols])
        ss = StandardScaler()
        test_x.loc[:,scale_cols] = ss.fit_transform(test_x.loc[:,scale_cols])
        test_x.loc[:,scale_cols] = test_x.loc[:,scale_cols].round(4)
        test_l_home = test_x.loc[(test_x.loc[:,'Team'] == True),:].groupby(['PlayId','Team']).agg(lambda x: list(x))
        test_l_away = test_x.loc[(test_x.loc[:,'Team'] == False),:].groupby(['PlayId','Team']).agg(lambda x: list(x))
        X_test_categorical = test.loc[:,final_cats].drop_duplicates()
        test_basetable = create_features(test2, True)
        new_continuos_df = test_basetable[new_continuos].drop_duplicates()
        X_test_continuous = test[final_continuous].drop_duplicates()
        X_test_continuous = X_test_continuous.merge(new_continuos_df, on='PlayId')
        X_test_continuous.loc[:,scale_cols2] = ss.fit_transform(X_test_continuous.loc[:,scale_cols2])
        #y_test = test.loc[:,['PlayId','Yards']].drop_duplicates()
        #final_test = y_test.merge(X_test_continuous, on='PlayId')
        final_test = X_test_continuous.merge(X_test_categorical, on='PlayId')
        final_test = final_test.merge(test_l_home, on='PlayId')
        final_test = final_test.merge(test_l_away, on='PlayId')
        X_test_continuous = final_test[scale_cols2]
        X_test_categorical = final_test[final_cats2]
        X_test_series = final_test[series_cols]
        X_test_series = X_test_series.values.tolist()
        X_test_series = np.array(X_test_series)

        PossessionTeam = EmbeddingMapping(X_test_categorical['PossessionTeam'])
        Down = EmbeddingMapping(X_test_categorical['Down'])
        Distance = EmbeddingMapping(X_test_categorical['Distance'])
        FieldPosition = EmbeddingMapping(X_test_categorical['FieldPosition'])
        OffenseFormation = EmbeddingMapping(X_test_categorical['OffenseFormation'])
        OffensePersonnel = EmbeddingMapping(X_test_categorical['OffensePersonnel'])
        DefensePersonnel = EmbeddingMapping(X_test_categorical['DefensePersonnel'])
        #PlayerCollegeName = EmbeddingMapping(X_test_categorical['PlayerCollegeName'])
        #Position = EmbeddingMapping(X_test_categorical['0_Position'])
        HomeTeamAbbr = EmbeddingMapping(X_test_categorical['HomeTeamAbbr'])
        VisitorTeamAbbr = EmbeddingMapping(X_test_categorical['VisitorTeamAbbr'])
        Week = EmbeddingMapping(X_test_categorical['Week'])
        Stadium = EmbeddingMapping(X_test_categorical['Stadium'])
        Location = EmbeddingMapping(X_test_categorical['Location'])
        StadiumType = EmbeddingMapping(X_test_categorical['StadiumType'])
        Turf = EmbeddingMapping(X_test_categorical['Turf'])
        GameWeather = EmbeddingMapping(X_test_categorical['GameWeather'])
        #HomePossesion = EmbeddingMapping(X_test_categorical['0_HomePossesion'])
        #HomeField = EmbeddingMapping(X_test_categorical['0_HomeField'])

        X_test_cat_mapping = pd.DataFrame()
        X_test_cat_mapping = X_test_cat_mapping.assign(#PossessionTeam = X_test_categorical['PossessionTeam'].apply(PossessionTeam.get_mapping),
                                                    Down = X_test_categorical['Down'].apply(Down.get_mapping),
                                                    Distance = X_test_categorical['Distance'].apply(Distance.get_mapping),
                                                    FieldPosition = X_test_categorical['FieldPosition'].apply(FieldPosition.get_mapping),
                                                    OffenseFormation = X_test_categorical['OffenseFormation'].apply(OffenseFormation.get_mapping),
                                                    OffensePersonnel = X_test_categorical['OffensePersonnel'].apply(OffensePersonnel.get_mapping),
                                                    DefensePersonnel = X_test_categorical['DefensePersonnel'].apply(DefensePersonnel.get_mapping),
                                                    #PlayerCollegeName = X_test_categorical['0_PlayerCollegeName'].apply(PlayerCollegeName.get_mapping),
                                                    #Position = X_test_categorical['0_Position'].apply(Position.get_mapping),
                                                    HomeTeamAbbr = X_test_categorical['HomeTeamAbbr'].apply(HomeTeamAbbr.get_mapping),
                                                    VisitorTeamAbbr = X_test_categorical['VisitorTeamAbbr'].apply(VisitorTeamAbbr.get_mapping)
                                                    #Week = X_test_categorical['Week'].apply(Week.get_mapping),
                                                    #Stadium = X_test_categorical['Stadium'].apply(Stadium.get_mapping),
                                                    #Location = X_test_categorical['Location'].apply(Location.get_mapping),
                                                    #StadiumType = X_test_categorical['StadiumType'].apply(StadiumType.get_mapping),
                                                    #Turf = X_test_categorical['Turf'].apply(Turf.get_mapping),
                                                    #GameWeather = X_test_categorical['GameWeather'].apply(GameWeather.get_mapping)
                                                    #HomePossesion = X_test_categorical['0_HomePossesion'].apply(HomePossesion.get_mapping),
                                                    #HomeField = X_test_categorical['0_HomeField'].apply(HomeField.get_mapping)
                                                    )
        X_test_cat_mapping = np.array(X_test_cat_mapping)

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X_test_continuous)
        X_test_continuous = imp.transform(X_test_continuous)

        testdata = {'numeric_inputs': X_test_continuous,
            'cat_inputs': X_test_cat_mapping,
           'series_inputs':X_test_series,
           'lstm_inputs':X_test_series}


        y_pred = model.predict(testdata)
        yardsleft = np.array(test['YardsLeft'][::22])

        for i in range(len(yardsleft)):
            y_pred[i, :yardsleft[i]-1] = 0
            y_pred[i, yardsleft[i]+100:] = 1
        env.predict(pd.DataFrame(data=y_pred.clip(0,1),columns=sample.columns))
        
    except:
        env.predict(sample)
        
    return 0


# In[69]:


for test, sample in tqdm.tqdm(env.iter_test()):
    make_pred(test, sample, env, model)


# In[70]:


env.write_submission_file()

