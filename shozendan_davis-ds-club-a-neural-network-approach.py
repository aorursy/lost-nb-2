#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
from kaggle.competitions import nflrush
import tqdm
import re
from string import punctuation
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import keras
import keras.backend as K


# In[2]:


env = nflrush.make_env()


# In[3]:


train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", dtype={'WindSpeed': 'object'})


# In[4]:


class DataParser:

    def __init__(self, data, predict=False, encoders={}):
        self.data = data
        self.predict = predict
        self.encoders = encoders

    def cleanWindSpeed(self, x):
        x = str(x)
        x = x.lower()
        if '-' in x:
            x = (int(x.split('-')[0]) + int(x.split('-')[1])) / 2
        elif ' gusts up to 25 ' in x:
            x = (int(x.split(' gusts up tp 25 ')))
        try:
            return float(x)
        except:
            return -1

    def cleanGameWeather(self, x):
        x = str(x).lower()
        if 'sunny' in x or 'clear' in x or 'fair' in x:
            return 'sunny'
        elif 'cloud' in x or 'coudy' in x or 'clouidy' in x or 'hazy' in x or 'sun & clouds' in x or 'overcast' in x:
            return 'cloudy'
        elif 'rain' in x or 'shower' in x or 'rainy' in x:
            return 'rainy'
        elif 'controlled climate' in x or 'indoor' in x:
            return 'indoor'
        elif 'snow' in x:
            return 'snowy'
        return 'missing'
        
    def mapGameWeather(self, txt):
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

    def cleanStadiumType(self, txt):  # Fixes the typo
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
        txt = txt.replace('retractable', 'rtr.')
        return txt

    # Focuses only on the words: outdoor, indoor, closed and open.
    def cleanStadiumType2(self, txt):
        if pd.isna(txt):
            return np.nan
        if 'outdoor' in txt or 'open' in txt:
            return 1
        if 'indoor' in txt or 'closed' in txt:
            return 0
        return np.nan

    def cleanDefencePersonnel(self):
        arr = [[int(s[0]) for s in t.split(', ')]
               for t in self.data['DefensePersonnel']]
        self.data['DL'] = pd.Series([int(a[0]) for a in arr])
        self.data['LB'] = pd.Series([int(a[1]) for a in arr])
        self.data['DB'] = pd.Series([int(a[2]) for a in arr])
        self.data = self.data.drop(labels=["DefensePersonnel"], axis=1)

    def cleanOffencePersonnel(self):
        arr = [[int(s[0]) for s in t.split(", ")]
               for t in self.data["OffensePersonnel"]]
        self.data["RB"] = pd.Series([int(a[0]) for a in arr])
        self.data["TE"] = pd.Series([int(a[1]) for a in arr])
        self.data["WR"] = pd.Series([int(a[2]) for a in arr])
        self.data = self.data.drop(labels=["OffensePersonnel"], axis=1)

    def cleanOffenseFormation(self):
        self.data['OffenseFormation'].fillna('missing', inplace=True)
        if(not self.predict):
            le = LabelEncoder()
            le.fit(self.data['OffenseFormation'])
            self.encoders['OffenseFormation'] = le
        self.data['OffenseFormation'] = self.encoders['OffenseFormation'].transform(self.data['OffenseFormation'])
        
    def cleanHeight(self):
        """
        Parses the PlayerHeight column and converts height into inches
        """
        self.data['PlayerHeight'] = self.data['PlayerHeight'].apply(
            lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    def cleanTimeHandoff(self):
        self.data['TimeHandoff'] = self.data['TimeHandoff'].apply(
            lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    def cleanTimeSnap(self):
        self.data['TimeSnap'] = self.data['TimeSnap'].apply(
            lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    def cleanGameClock(self):
        arr = [[int(s[0]) for s in t.split(":")]
               for t in self.data["GameClock"]]
        self.data["GameHour"] = [int(a[0]) for a in arr]
        self.data["GameMinute"] = [int(a[1]) for a in arr]
        self.data = self.data.drop(labels=['GameClock'], axis=1)

    def cleanTurf(self):
        # from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087
        Turf = {'Field Turf': 'Artificial', 'A-Turf Titan': 'Artificial', 'Grass': 'Natural', 'UBU Sports Speed S5-M': 'Artificial',
                'Artificial': 'Artificial', 'DD GrassMaster': 'Artificial', 'Natural Grass': 'Natural',
                'UBU Speed Series-S5-M': 'Artificial', 'FieldTurf': 'Artificial', 'FieldTurf 360': 'Artificial', 'Natural grass': 'Natural', 'grass': 'Natural',
                'Natural': 'Natural', 'Artifical': 'Artificial', 'FieldTurf360': 'Artificial', 'Naturall Grass': 'Natural', 'Field turf': 'Artificial',
                'SISGrass': 'Artificial', 'Twenty-Four/Seven Turf': 'Artificial', 'natural grass': 'Natural'}

        self.data['Turf'] = self.data['Turf'].map(Turf)
        self.data['Turf'] = self.data['Turf'] == 'Natural'

    def cleanPossessionTeam(self):  # fixes problem in team name encoding
        map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
        for abb in self.data['PossessionTeam'].unique():
            map_abbr[abb] = abb
        self.data['PossessionTeam'] = self.data['PossessionTeam'].map(
            map_abbr)
        self.data['HomeTeamAbbr'] = self.data['HomeTeamAbbr'].map(map_abbr)
        self.data['VisitorTeamAbbr'] = self.data['VisitorTeamAbbr'].map(
            map_abbr)

    def cleanPlayerBirthDate(self):
        self.data['PlayerBirthDate'] = self.data['PlayerBirthDate'].apply(
            lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    def cleanWindDirection(self, txt):
        if pd.isna(txt):
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
    
    def mapWindDirection(self, txt):
        windDirectionMap = {
            'n': 0,'nne': 1/8,'nen': 1/8,'ne': 2/8,
            'ene': 3/8,'nee': 3/8,'e': 4/8,'ese': 5/8,
            'see': 5/8,'se': 6/8,'ses': 7/8,'sse': 7/8,
            's': 1,'ssw': 9/8,'sws': 9/8,'sw': 10/8,
            'sww': 11/8,'wsw': 11/8,'w': 12/8,'wnw': 13/8,
            'nw': 14/8,'nwn': 15/8,'nnw': 15/8
        }
        try:
            return windDirectionMap[txt]
        except:
            return np.nan

    def cleanPlayDirection(self):
        """
        1 if play direction if right, 0 if play direction is left.
        """
        self.data['PlayDirection'] = self.data['PlayDirection'].apply(
            lambda x: x.strip() == 'right')

    def cleanTeam(self):
        """
        1 if home team, 0 if away team
        """
        self.data['Team'] = self.data['Team'].apply(
            lambda x: x.strip() == 'home')
        
    def isRusher(self):
        self.data['isRusher'] = self.data['NflId'] == self.data['NflIdRusher']
        temp = self.data[self.data['isRusher']][['Team', 'PlayId']].rename(columns={'Team':'RusherTeam'})
        self.data = self.data.merge(temp, on = 'PlayId')
        self.data['isRusherTeam'] = self.data['Team'] == self.data['RusherTeam']
        self.data.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
        
    def parse(self):
        self.data['WindSpeed'] = self.data['WindSpeed'].apply(self.cleanWindSpeed)
        self.data['GameWeather'] = self.data['GameWeather'].apply(self.cleanGameWeather)
        self.data['GameWeather'] = self.data['GameWeather'].apply(self.mapGameWeather)
        self.data['StadiumType'] = self.data['StadiumType'].apply(self.cleanStadiumType)
        self.data['StadiumType'] = self.data['StadiumType'].apply(self.cleanStadiumType2)
        self.data['WindDirection'] = self.data['WindDirection'].apply(self.cleanWindDirection)
        self.data['WindDirection'] = self.data['WindDirection'].apply(self.mapWindDirection)
        self.cleanOffenseFormation()
        self.cleanOffencePersonnel()
        self.cleanDefencePersonnel()
        self.cleanHeight()
        self.cleanTimeHandoff()
        self.cleanTimeSnap()
        self.cleanTurf()
        self.cleanPossessionTeam()
        self.cleanPlayerBirthDate()
        self.cleanPlayDirection()
        self.cleanTeam()
        self.isRusher()
        
        if(not self.predict):
            return self.data, self.encoders
        return self.data


# In[5]:


parser = DataParser(train)
train, encoders = parser.parse()


# In[6]:


class FeatureEngine:

    def __init__(self, data, exclude=[], deploy=False):
        self.data = data  # Clean data from the parser
        self.exclude = exclude  # Pass a list of processes to exclude
        self.include = ['isRusher',
                        'HorizontalSandA',
                        'HomeField',
                        'FieldEqPossession',
                        'PlayerAge',
                        'HandSnapDelta',
                        'YardsLeft',
                        'BMI',
                        'DefendersInTheBox_vs_Distance']
        self.deploy = deploy

    ### Helper Functions ###
    def normalizeX(self, x_coordinate, play_direction):
        if play_direction == 1:
            return 120 - x_coordinate
        else:
            return x_coordinate

    def normalizeOrientation(self, angle, play_direction):
        if play_direction == 0:
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return new_angle
        else:
            return angle
    
    def newLine(self, rush_team, field_position, yardline):
        if rush_team == field_position:
            return 10.0 + yardline
        else:
            return 60.0 + (50 - yardline)
        
    def euclideanDistance(self, x1, y1, x2, y2):
        x_diff = (x1-x2)**2
        y_diff = (y1-y2)**2
        return np.sqrt(x_diff + y_diff)
    
    def backDirection(self, orientation):
        if orientation > 180.0:
            return 1
        else:
            return 0
    
    def updateYardline(self, df):
        new_yardline = df[df['isRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: self.newLine(x[0],x[1],x[2]), axis=1)
        new_yardline = new_yardline[['GameId', 'PlayId', 'YardLine']]
        return new_yardline
    
    def updateOrientation(self, df, yardline):
        df['X'] = df[['X','PlayDirection']].apply(lambda x: self.normalizeX(x[0],x[1]), axis=1)
        df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: self.normalizeOrientation(x[0],x[1]), axis=1)
        df = df.drop('YardLine', axis=1)
        df = pd.merge(df, yardline, on=['GameId', 'PlayId'], how='inner')
        return df
    
    def backFeatures(self, df):
        carriers = df[df['isRusher']][['GameId','PlayId','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: self.backDirection(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: self.backDirection(x))
        carriers = carriers.rename(columns={'X':'back_X',
                                            'Y':'back_Y'})
        carriers = carriers[['GameId','PlayId','back_X','back_Y', 'back_from_scrimmage', 'back_oriented_down_field', 'back_moving_down_field']]
        return carriers
    
    def featuresRelativeToBack(self, df, carriers):
        player_distance = df[['GameId','PlayId','X','Y','isRusher']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['isRusher'] == 0]
        player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: self.euclideanDistance(x[0],x[1],x[2],x[3]), axis=1)
        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])                                         .agg({'dist_to_back':['min','max','mean','std']})                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']
        return player_distance
    
    def defenseFeatures(self, df):
        rusher = df[df['isRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['isRusherTeam'] != True][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: self.euclideanDistance(x[0],x[1],x[2],x[3]), axis=1)
        defense = defense.groupby(['GameId','PlayId'])                         .agg({'def_dist_to_back':['min','max','mean','std']})                         .reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']
        return defense
    
    def rusherFeatures(self, df): 
        rusher = df[df['isRusher']][['GameId','PlayId','Dir', 'S', 'A', 'X', 'Y']]
        rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY']
    
        radian_angle = (90 - rusher['RusherDir']) * np.pi / 180.0
        v_horizontal = np.abs(rusher['RusherS'] * np.cos(radian_angle))
        v_vertical = np.abs(rusher['RusherS'] * np.sin(radian_angle)) 
    
        rusher['v_horizontal'] = v_horizontal
        rusher['v_vertical'] = v_vertical
        
        rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS','RusherA','RusherX', 'RusherY','v_horizontal', 'v_vertical']
        
        return rusher
    
    ### Engineering Functions ###
    def engineerFieldEqPossession(self):
        self.data['FieldEqPossession'] = self.data['FieldPosition'] == self.data['PossessionTeam']

    def engineerHomeField(self):
        self.data['HomeField'] = self.data['FieldPosition'] == self.data['HomeTeamAbbr']
    
    def engineerHandoffSnapDelta(self):
        self.data['TimeDelta'] = self.data.apply(lambda row: (
            row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        self.data = self.data.drop(['TimeHandoff', 'TimeSnap'], axis=1)

    def engineerYardsLeft(self):
        """
        Computes yards left from end-zone

        Note
        ----
        Requires variable HomeField (must execute engineerHomeField before execution)
        """
        self.data['YardsLeft'] = self.data.apply(
            lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
        self.data['YardsLeft'] = self.data.apply(
            lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)
        
    def engineerBMI(self):
        """
        Computes the BMI of a player from height and weight
        """
        self.data['PlayerBMI'] = 703 *             (self.data['PlayerWeight']/(self.data['PlayerHeight'])**2)

    def engineerPlayerAge(self):
        """
        Computes the age of the player from TimeHandoff
        """
        seconds_in_year = 60*60*24*365.25
        self.data['PlayerAge'] = self.data.apply(lambda row: (
            row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
        self.data = self.data.drop(['PlayerBirthDate'], axis=1)

    def engineerDefendersInTheBox_vs_Distance(self):
        dfInBox_mode = self.data['DefendersInTheBox'].mode()
        self.data['DefendersInTheBox'].fillna(
            dfInBox_mode.iloc[0], inplace=True)
        self.data['DefendersInTheBox_vs_Distance'] = self.data['DefendersInTheBox'] /             self.data['Distance']
        
    def combineFeatures(self, relative_to_back, defense,rushing, static):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,rushing,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        return df

    ### Outputs clean and engineered DataFrame ###
    def engineer(self):
        yardline = self.updateYardline(self.data)
        self.data = self.updateOrientation(self.data, yardline)
        back_feats = self.backFeatures(self.data)
        rel_back = self.featuresRelativeToBack(self.data, back_feats)
        def_feats = self.defenseFeatures(self.data)
        rush_feats = self.rusherFeatures(self.data)
        
        for c in self.include:

            if c in self.exclude:
                continue

            elif c == 'FieldEqPossession':
                self.engineerFieldEqPossession()

            elif c == 'HomeField':
                self.engineerHomeField()

            elif c == 'YardsLeft':
                self.engineerYardsLeft()

            elif c == 'PlayerAge':
                self.engineerPlayerAge()

            elif c == 'HandSnapDelta':
                self.engineerHandoffSnapDelta()

            elif c == 'BMI':
                self.engineerBMI()

            elif c == 'DefendersInTheBox_vs_Distance':
                self.engineerDefendersInTheBox_vs_Distance()
                
        self.data = self.combineFeatures(rel_back, def_feats, rush_feats, self.data)
        self.data = self.data.drop_duplicates()

        return self.data


# In[7]:


engine = FeatureEngine(train.copy())
train = engine.engineer()


# In[8]:


train.columns


# In[9]:


def DataReshaper2(data, predict=False, playersCol = []):
    """
    Takes the parsed and feature engineered data and outputs X_train and y_train
    vectors that are compatible for neural networks and machine learning algorithms

    Parameters:
    -----------
    data: parsed and feature engineered data (pandas dataframe format)
    predict: must be true in the prediction stage
    playersCol: pass the players_col created in the training stage

    Returns:
    --------
    X_train: a 2 dimentional vector housing all predictor variables
    y_train: a 1 dimentional vector housing all response variable
    players_col: The names of variables that are unique to each player (ex: height and weight)

    Note:
    -----
    Requires Standard Scalar from the scikit learn library
    
    References
    ----------
    https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
    """
    
    ### Dropping Unnecessary Columns and Filling NAs ###
    data = data.sort_values(by=['PlayId', 'Team', 'isRusher', 'JerseyNumber']).reset_index()
    data.drop(['GameId', 'PlayId', 'index', 'isRusher', 'Team'], axis=1, inplace=True)

    drop_col = []
    for c in data.columns:
            if data[c].dtype == 'object':
                drop_col.append(c)
    data.drop(drop_col, axis=1, inplace=True)
    
    data.fillna(-999, inplace=True)
    
    ### Creating One Large Row ###
    players_col = playersCol
    if(not predict):
        for col in data.columns:
            if data[col][:22].std() != 0:
                players_col.append(col)  # this measure is taken to avoid repeating data
    
    X_train = np.array(data[players_col]).reshape(-1, len(players_col)*22)

    if(not predict):
        play_col = data.drop(players_col + ['Yards'], axis=1).columns
    else:
        play_col = data.drop(players_col, axis=1).columns

    X_play_col = np.zeros(shape=(X_train.shape[0], len(play_col)))
    for i, col in enumerate(play_col):
            X_play_col[:, i] = data[col][::22]

    X_train = np.concatenate([X_train, X_play_col], axis=1)
    
    ### Reshaping y_train(only for training stage) ###
    if(not predict):
        y_train = np.zeros(shape=(X_train.shape[0], 199))
        for i, yard in enumerate(train['Yards'][::22]):
            y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
                            
    if(not predict):
        return X_train, y_train, players_col
    return X_train


# In[10]:


X_train, y_train, players_col = DataReshaper2(train)


# In[ ]:





# In[11]:


from keras.layers import Dense,Input,Dropout,BatchNormalization,LeakyReLU,PReLU,GaussianNoise
from keras.models import Model
import keras.backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback


# In[12]:


def build_model():
    input_tensor = Input(shape=(X_train.shape[1],))
    x = Dense(1024, activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = GaussianNoise(0.15)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = GaussianNoise(0.15)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = GaussianNoise(0.15)(x)
    output_tensor = Dense(199, activation='softmax')(x)
    
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer='adam', loss=crps_loss, metrics=[])
    return model


# In[13]:


ES = EarlyStopping(monitor='CRPS_score_val',
                   mode='min',
                   restore_best_weights=True,
                   patience=3
                  )


# In[14]:


class CRPSCallback(Callback):
    def __init__(self, validation, predict_batch_size=1024, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        
    def on_batch_begin(self, batch, logs={}):
        pass
    
    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')
            
    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')
            
    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')
        if(self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s


# In[15]:


# Calculate CRPS score
def crps_score(y_prediction, y_valid, shape=X_train.shape[0]):
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_prediction, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * shape)
    crps = np.round(val_s, 6)
    
    return crps


# In[16]:


#from https://www.kaggle.com/davidcairuz/nfl-neural-network-w-softmax
def crps_loss(y_true, y_pred):
    return K.mean(K.square(y_true - K.cumsum(y_pred, axis=1)), axis=1)


# In[17]:


EPOCHS = 200
BATCH_SIZE = 1024
def train_model(X_train, y_train, X_val, y_val):
    model = build_model()
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[CRPSCallback(validation = (X_val, y_val)), ES],
              verbose=1)
    
    y_pred = model.predict(X_val)
    y_valid = y_val
    crps = crps_score(y_pred, y_valid, shape=X_val.shape[0])
    
    return model, crps


# In[18]:


from sklearn.model_selection import KFold
N_SPLITS = 3
N_REPEATS = 2

nn_crps = []
models = []

for n in range(N_REPEATS):
    kf = KFold(N_SPLITS, random_state= 21 + n, shuffle = True)
    for k_fold, (tr_idx, vl_idx) in enumerate(kf.split(y_train)):
        print("-----------")
        print(f'Loop {n+1}/{N_REPEATS}' + f' Fold {k_fold+1}/{N_SPLITS}')
        print("-----------")
        x_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        x_vl, y_vl = X_train[vl_idx], y_train[vl_idx]

        model, crps = train_model(x_tr, y_tr, x_vl, y_vl)
        models.append(model)
        print("the %d fold crps (NN) is %f"%((k_fold+1), crps))
        nn_crps.append(crps)


# In[19]:


def make_pred(df, sample, env, models):
    parser = DataParser(df, predict=True, encoders=encoders)
    df = parser.parse()
    
    engine = FeatureEngine(df)
    df = engine.engineer()
    
    X = DataReshaper2(df, predict=True, playersCol=players_col)
    y_pred = np.mean([np.cumsum(model.predict(X), axis=1) for model in models], axis=0)
    yardsleft = np.array(df['YardsLeft'][::22])
    
    for i in range(len(yardsleft)):
        y_pred[i, :int(yardsleft[i]-1)] = 0
        y_pred[i, int(yardsleft[i]+100):] = 1
    env.predict(pd.DataFrame(data=y_pred.clip(0,1),columns=sample.columns))
    return y_pred


# In[20]:


for test, sample in tqdm.tqdm(env.iter_test()):
     make_pred(test, sample, env, models)


# In[21]:


env.write_submission_file()

