#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import math
import time
import codecs
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from functools import wraps
from datetime import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d
from hyperopt import Trials, STATUS_OK, tpe

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KernelDensity

import keras.backend as K
from keras import regularizers
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda, BatchNormalization
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split, KFold

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


TRAIN_DATA_PATH = Path('../input/nfl-big-data-bowl-2020/train.csv')
#TRAIN_DATA_PATH = Path('/Users/sam.taylor/Desktop/train.csv')
SEED = 0


# In[3]:


train = pd.read_csv(TRAIN_DATA_PATH, dtype={'WindSpeed': 'object'})
train.head()


# In[4]:


train.loc[509755:509769, ['WindSpeed', 'WindDirection']]


# In[5]:


replacements = [
    ('LSU', 'Louisiana State'),
    ('Southern California', 'USC'),
    ('Miami (Fla.)', 'Miami'),
    ('Miami, O.', 'Miami OH'),
    ('Miami (Ohio)', 'Miami OH'),
    ('Texas-El Paso', 'Texas')
]


# In[6]:


position_groups = {
    'CB' : 'DB',
    'FS' : 'DB',
    'SAF' : 'DB',
    'S' : 'DB',
    'SS' : 'DB',
    'DB' : 'DB',
    'OLB' : 'LB',
    'ILB' : 'LB',
    'MLB' : 'LB',
    'LB' : 'LB',
    'DE' : 'DL',
    'DT' : 'DL',
    'NT' : 'DL',
    'DL' : 'DL',
    'G' : 'OL',
    'OG' : 'OL',
    'T' : 'OL',
    'OT' : 'OL',
    'C' : 'OL',
    'RB' : 'RB',
    'FB' : 'RB',
    'HB' : 'RB',
    'WR' : 'SK',
    'QB' : 'SK',
    'TE' : 'SK'
}
posession_positions = {
    'DB': 0,
    'LB': 0,
    'DL': 0,
    'OL': 1,
    'RB': 1,
    'SK': 1
}


# In[7]:


COLS_TO_DROP = [
    'Position', 'IsRusher', 'WindDirection', 'WindSpeed'
]
CATEGORICAL_VARS = [
    'DisplayName', 'PlayerCollegeName', 'Location',
    'OffensePersonnel', 'Stadium', 'DefensePersonnel',
    'HomeTeamAbbr', 'VisitorTeamAbbr', 'FieldPosition', 'PossessionTeam',
    'StadiumType', 'Position', 'Turf', 'PlayerHeight', 'OffenseFormation',
    'JerseyNumber', 'NflId'
]
TRANSFORMED_CATEGORICALS = [
    'OffensePersonnel', 'DefensePersonnel', 'PlayerHeight',
    'NflId', 'Turf', 'FieldPosition', 'PlayerCollegeName', 'JerseyNumber',
    'PossessionTeam', 'GameWeather'
]
VARS_TO_ONE_HOT = [var for var in CATEGORICAL_VARS
                   if var not in TRANSFORMED_CATEGORICALS
                   and var not in COLS_TO_DROP]
MAX_CARDINALITY = 24


# In[8]:


cardinalities = train[VARS_TO_ONE_HOT].nunique().sort_values(ascending=False)
cardinalities


# In[9]:


def shoelace(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y,1))-np.dot(y,np.roll(x,1)))


# In[10]:


def func_timer(fn):
    @wraps(fn)
    def timer(*args, **kwargs):
        print('\n' + fn.__name__.upper())
        start_time = datetime.now()
        res = fn(*args, **kwargs)
        end_time = datetime.now()
        time_taken = end_time - start_time
        time_taken = time_taken.total_seconds() / 60
        print('{} time taken: {:.2f} mins'.format(fn.__name__, time_taken))
        return res
    return timer

def transformation_check(fn):
    @wraps(fn)
    def checker(*args, **kwargs):
        print(' -- {} -- '.format(fn.__name__))
        res = fn(*args, **kwargs)
        end_shape = res.shape
        null_prc = 100 * (res.isnull().sum() / len(res)).mean()
        print('\tshape = {}'.format(end_shape))
        print('\tnull % = {:.2f}'.format(null_prc))
        return res
    return checker


# In[11]:


class Preprocessor(TransformerMixin):
    def __init__(self, one_hot_vars, max_cardinality, cols_to_drop=[]):
        super().__init__()
        self.target = 'Yards'
        self.one_hot_vars = one_hot_vars
        self.max_cardinality = max_cardinality
        self.cols_to_drop = cols_to_drop
        self.player_cols = []
        self.one_hot_encoder = OneHotEncoder(
            sparse=False, 
            dtype=np.int, 
            handle_unknown='ignore'
        )
        self.college_encoding = {}
        self.jersey_encoding = {}
        self.map_abbr = None

    @func_timer
    def initial_cleaning(self, X):
        """ Transformative steps that don't need any 'fitted'
        objects. Also any thing that needs to be done before anything
        is fit """
        X = self._correct_team_abbreviations(X)
        X = self._encode_player_height(X)
        X = self._process_time_variables(X)
        X = self._fix_wind_variables(X)
        X = self._fix_stadium_type_and_turf(X)
        X = self._map_college_and_pos(X)
        X = self._encode_personnel(X)
        X = self._normalise_positional_data(X)
        X = self._calc_voronoi(X)
        X = self._misc_engineering(X)
        X.drop(columns=self.cols_to_drop, inplace=True)
        return X

    @func_timer
    def fit(self, X, y=None):
        # Get player related columns
        self._get_player_cols(X)

        # Fit one hot encoder
        cardinalities = X[self.one_hot_vars].nunique().to_dict()
        one_hot_vars = [var for var in self.one_hot_vars
                        if cardinalities[var] <= self.max_cardinality]
        self.one_hot_encoder.fit(X[one_hot_vars].fillna('unknown'))
        self.oh_cols_to_drop = [var for var in self.one_hot_vars
                                if var not in one_hot_vars]
        self.one_hot_vars = one_hot_vars
        
        # Fit college name and jersey number 'encoder's
        self.college_encoding =             X.groupby('PlayerCollegeName')['PlayId'].count().to_dict()
        self.jersey_encoding =             X.groupby('JerseyNumber')['PlayId'].count().to_dict()
        
        return self

    @func_timer
    def transform(self, X):
        X['PlayerCollegeName'] = X['PlayerCollegeName'].map(self.college_encoding)
        X['PlayerCollegeNameRusher'] =             X['PlayerCollegeNameRusher'].map(self.college_encoding)
        X['JerseyNumber'] = X['JerseyNumber'].map(self.jersey_encoding)
        X = self._apply_one_hot_encoder(X)
        X = self._flatten_player_vars(X)
#         X = self._previous_play_data(X)
        return X

    @transformation_check
    def _correct_team_abbreviations(self, X):
        if self.map_abbr is None:
            self.map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
            for abb in X['PossessionTeam'].unique():
                self.map_abbr[abb] = abb

        X['PossessionTeam'] = X['PossessionTeam'].map(self.map_abbr)
        X['HomeTeamAbbr'] = X['HomeTeamAbbr'].map(self.map_abbr)
        X['VisitorTeamAbbr'] = X['VisitorTeamAbbr'].map(self.map_abbr)
        return X

    @transformation_check
    def _encode_player_height(self, X):
        def string_to_inches(x):
            feet, inch = x.split('-')
            return int(inch) + 12 * int(feet)

        X['PlayerHeight'] = X['PlayerHeight'].apply(string_to_inches)
        return X

    @transformation_check
    def _process_time_variables(self, X):
        for col in ['TimeHandoff', 'TimeSnap', 'PlayerBirthDate']:
            X[col] = pd.to_datetime(X[col], utc=True, infer_datetime_format=True)
        X['TimeUntilHandoff'] = X['TimeSnap'] - X['TimeHandoff']
        X['TimeUntilHandoff'] = X['TimeUntilHandoff'].dt.total_seconds()

        X['PlayerAge'] = X['TimeSnap'] - X['PlayerBirthDate']
        X['PlayerAge'] = X['PlayerAge'].dt.total_seconds() / 31556952

        X['GameClock'] = 360 * X['GameClock'].str[:2].astype(int)                          + 60 * X['GameClock'].str[3:5].astype(int)                          + X['GameClock'].str[6:8].astype(int)
        
        X['SecondsRemaining'] = 0
        X.loc[X['Quarter'] == 1, 'SecondsRemaining'] = 45 * 60 + X['GameClock']
        X.loc[X['Quarter'] == 2, 'SecondsRemaining'] = 30 * 60 + X['GameClock']
        X.loc[X['Quarter'] == 3, 'SecondsRemaining'] = 15 * 60 + X['GameClock']
        X.loc[X['Quarter'] == 4, 'SecondsRemaining'] = 0 * 60 + X['GameClock']
        X.loc[X['Quarter'] == 5, 'SecondsRemaining'] = 0 * 60 + X['GameClock']

        X.drop(columns=['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], inplace=True)
        return X

    @transformation_check
    def _fix_wind_variables(self, X):
        def average_ranges(x):
            x = str(x)
            if '-' in x:
                low, high = x.split('-')
                return str((int(high) + int(low)) / 2)
            elif ' gusts up to ' in x:
                low, high = x.split(' gusts up to ')
                return str((int(high) + int(low)) / 2)
            else:
                return x

        def coerce_to_int(x):
            try:
                x = int(x)
            except:
                x = np.nan
            return x
        
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

        X['WindSpeed'] = X['WindSpeed'].str.lower().str.replace('mph', '')
        X['WindSpeed'] = X['WindSpeed'].str.strip()
        X['WindSpeed'] = X['WindSpeed'].apply(average_ranges)
        X['WindSpeed'] = X['WindSpeed'].apply(coerce_to_int)
        
        X['GameWeather'] = X['GameWeather'].apply(map_weather)

        acceptable_directions = [
            'NE', 'SW', 'S', 'NW', 'WSW', 'SE', 'W', 'N', 'NNE', 'WNW', 'SSW',
            'NNW', 'SSE', 'E', 'ENE', 'ESE'
        ]
        X['WindDirection'] = X['WindDirection'].str.upper()
        X['WindDirection'] = X['WindDirection'].str.replace('FROM ', '').str.replace('-', '')
        X.loc[~X['WindDirection'].isin(acceptable_directions), 'WindDirection'] = np.nan
        return X
    
    @transformation_check
    def _fix_stadium_type_and_turf(self, X):
        stadium_type_map = {
            'Outdoor': 'Outdoor',
            'Outdoors': 'Outdoor',
            'Indoors': 'Indoor',
            'Dome': 'Indoor',
            'Indoor': 'Indoor',
            'Retractable Roof': 'Retr Open',
            'Open': 'Retr Open',
            'Retr. Roof-Closed': 'Retr Closed',
            'Retr. Roof - Closed': 'Retr Closed',
            'Domed, closed': 'Retr Closed',
            'Domed, open': 'Retr Open',
            'Closed Dome': 'Retr Closed',
            'Dome, closed': 'Retr Closed',
            'Domed': 'Indoor',
            'Oudoor': 'Outdoor',
            'Indoor, Roof Closed': 'Retr Closed',
            'Retr. Roof Closed': 'Retr Closed',
            'Retr. Roof-Open': 'Retr Open',
            'Bowl': 'Outdoor',
            'Outddors': 'Outdoor',
            'Heinz Field': 'Outdoor',
            'Outdoor Retr Roof-Open': 'Retr Open',
            'Retr. Roof - Open': 'Retr Open',
            'Indoor, Open Roof': 'Retr Open',
            'Ourdoor': 'Outdoor',
            'Outdor': 'Outdoor',
            'Outside': 'Outdoor',
            'Cloudy': 'Outdoor',
            'Domed, Open': 'Retr Open'
        }
        X['StadiumType'] = X['StadiumType'].map(stadium_type_map)
        
        grass_labels = ['grass', 'natural grass', 'natural', 'naturall grass']
        X['Turf'] = np.where(X['Turf'].str.lower().isin(grass_labels), 1, 0)
        
        def get_city(x):
            x = x.replace('e.', 'east').replace('.', ',')
            return x.split(',')[0].strip().lower()
        X['Location'] = X['Location'].apply(get_city)
        
        return X
    
    @transformation_check
    def _map_college_and_pos(self, X):
        for replacement in replacements:
            X['PlayerCollegeName'] = X['PlayerCollegeName']                .replace(replacement[0], replacement[1])
#         X['CollegeConference'] = X['PlayerCollegeName'].map(college_to_conf)
#         X['CollegeConference'].fillna('BinJuice', inplace=True)
#         if 'CollegeConference' not in self.one_hot_vars:
#             self.one_hot_vars += ['CollegeConference']
        
        X['PositionGroup'] = X['Position'].map(position_groups)
        X['InPossesion'] = X['PositionGroup'].map(posession_positions)
        return X

    @transformation_check
    def _encode_personnel(self, X):

        def count_positions(x, offensive):
            offensive_counts = {'DB': 0, 'DL': 0, 'LB': 0, 'OL': 0, 'QB': 0, 'RB': 0, 'TE': 0, 'WR': 0}
            defensive_counts = {'DB': 0, 'DL': 0, 'LB': 0, 'OL': 0}
            if offensive:
                val_counts=offensive_counts
            else:
                val_counts=defensive_counts

            if isinstance(x, str):
                for position_val in x.split(','):
                    val, pos = position_val.strip().split(' ')
                    if pos in val_counts:
                        val_counts[pos] += int(val)
            return val_counts
        
        X['OffensePersonnel'] = X['OffensePersonnel']             .apply(count_positions, offensive=True)
        off_personnel_df = pd.DataFrame().from_records(X['OffensePersonnel'].values)
        off_personnel_df.index = X.index
        off_personnel_df.columns = ['NOffensive' + col for col in off_personnel_df.columns]

        X['DefensePersonnel'] = X['DefensePersonnel']             .apply(count_positions, offensive=False)
        def_personnel_df = pd.DataFrame().from_records(X['DefensePersonnel'].values)
        def_personnel_df.index = X.index
        def_personnel_df.columns = ['NDefensive' + col for col in def_personnel_df.columns]

        X.drop(columns=['OffensePersonnel', 'DefensePersonnel'], inplace=True)
        X = pd.concat([X, off_personnel_df, def_personnel_df], axis=1)
        return X
    
    @transformation_check
    def _normalise_positional_data(self, X):
        X['IsLeftDirection'] = X['PlayDirection'] == 'left'
        
        X['PossesionInOwnHalf'] = X['PossessionTeam'] == X['FieldPosition']
        possession_in_own_half = X.groupby(['GameId', 'PlayId'])['PossesionInOwnHalf'].max().reset_index()
        X = X.drop(columns='PossesionInOwnHalf')            .merge(possession_in_own_half, on=['GameId', 'PlayId'])
        
        X['DistToEndZone'] = X['YardLine']
        X.loc[X['PossesionInOwnHalf'], 'DistToEndZone'] = 50 + (50 - X['YardLine'])
        X.loc[X['YardLine'] == 50, 'DistToEndZone'] = 50
        X['YardLineStd'] = 100 - X['DistToEndZone']
        
        X['XStd'] = X['X']
        X.loc[X['IsLeftDirection'], 'XStd'] = 120 - X.loc[X['IsLeftDirection'], 'X']
        
        X['YStd'] = X['Y']
        X.loc[X['IsLeftDirection'], 'YStd'] = 160 / 3 - X.loc[X['IsLeftDirection'], 'Y']
        
        X['PlayerDistToEndZone'] = 100 - (X['XStd'] - 10)
        
        X['DirRad'] = np.mod(90 - X['Dir'], 360) * math.pi / 180.0
        X['DirStd'] = X['DirRad']
        X.loc[X['IsLeftDirection'], 'DirStd'] =             np.mod(np.pi + X.loc[X['IsLeftDirection'], 'DirRad'], 2*np.pi)
        
        # Fix the problem with orientation over the years
        X['OrientationRad'] = np.mod(X['Orientation'], 360) * math.pi / 180.0
        X.loc[X['Season'] >= 2018, 'OrientationRad']             = np.mod(X.loc[X['Season'] >= 2018, 'Orientation'] - 90, 360) * math.pi / 180.0
        
        X['OrientationStd'] = X['OrientationRad']
        X.loc[X['IsLeftDirection'], 'OrientationStd'] =             np.mod(np.pi + X.loc[X['IsLeftDirection'], 'OrientationRad'], 2 * np.pi)
        X.drop(columns=['OrientationRad'], inplace=True)
        
        X['IsLeftDirection'] = (X['IsLeftDirection']).astype(int)
        X['IsRusher'] = (X['NflId'] == X['NflIdRusher']).astype(int)
        return X
    
    @transformation_check
    def _calc_voronoi(self, X):
        max_voronoi = 120 * 53.3
        X['VoronoiArea'] = 0
        X['VoronoiAreaNoOffence'] = 0
        for play_id in X['PlayId'].unique():
            play = X.loc[X['PlayId'] == play_id].copy()

            # Only consider space 5 yards behind the player furthest back
            x_cut_off = play['XStd'].min() - 5

            # Also calculate the rusher's voronoi excluding team mates
            no_offence_play = play[play['IsRusher'].astype(bool) |
                                   ~play['InPossesion'].astype(bool)]

            def mirror_boundary(xy):
                xy = xy.values
                n_points = xy.shape[0]
                xy1 = xy.copy()
                xy1[:,1] = -xy[:,1]
                xy2 = xy.copy()
                xy2[:,1] = 320/3 - xy[:,1]
                xy3 = xy.copy()
                xy3[:,0] = 2 * x_cut_off - xy[:,0]
                xy4 = xy.copy()
                xy4[:,0] = 220 - xy[:,0]
                return np.concatenate((xy, xy1, xy2, xy3, xy4), axis=0), n_points

            # Get voronoi
            xy, n = mirror_boundary(play[['XStd', 'YStd']])
            vor = Voronoi(xy)

            no_off_xy, _ = mirror_boundary(no_offence_play[['XStd', 'YStd']])
            no_off_vor = Voronoi(no_off_xy)

            # Calculate space area
            areas = np.zeros([play.shape[0], ])
            for i in range(n):
                player_point = vor.point_region[i]
                vertices = vor.vertices[vor.regions[player_point]]
                areas[i] = shoelace(vertices[:, 0], vertices[:, 1])

            rusher_index = np.argmax(no_offence_play['IsRusher'].values)
            rusher_index_in_df = no_offence_play.index.values[rusher_index]
            rusher_region = no_off_vor.point_region[rusher_index]
            rusher_vertex_index = no_off_vor.regions[rusher_region]
            rusher_vertices = no_off_vor.vertices[rusher_vertex_index]
            rusher_area = shoelace(rusher_vertices[:, 0], rusher_vertices[:, 1])

            # Assign to main df
            X.loc[play.index, 'VoronoiArea'] = areas
            X.loc[rusher_index_in_df, 'VoronoiAreaNoOffence'] = min(rusher_area, max_voronoi)
            
        X.loc[X['VoronoiArea'] > max_voronoi, 'VoronoiArea'] = max_voronoi
        return X

    @transformation_check
    def _misc_engineering(self, X):
        X['ScoreDiff'] = X['HomeScoreBeforePlay'] - X['VisitorScoreBeforePlay']
        # Set binary variables
        X['IsAwayTeam'] = (X['Team'] == 'away').astype(int)
        X['IsInAwayEnd'] = (X['FieldPosition'] == X['VisitorTeamAbbr']).astype(int)
        X['HomePossesion'] = (X['PossessionTeam'] == X['HomeTeamAbbr']).astype(int)
        X['AwayInPosession'] = (X['InPossesion'] == X['IsAwayTeam']).astype(int)
        # Directional features
        X['SX'] = X['S'] * np.cos(X['DirStd'])
        X['SY'] = X['S'] * np.sin(X['DirStd'])
        X['AX'] = X['A'] * np.cos(X['DirStd'])
        X['AY'] = X['A'] * np.sin(X['DirStd'])
        X['OrientationCos'] = X['OrientationStd'] * np.cos(X['DirStd'])
        X['OrientationSin'] = X['OrientationStd'] * np.sin(X['DirStd'])
        # Closeness to scrimmage line
        X['XToScrimmage'] = X['YardLineStd'] - (X['XStd'] - 10)
        
        # Get Rusher features
        rusher_pos = X.loc[
            X['IsRusher'] == 1, 
            ['XStd', 'YStd', 'DirStd', 'S', 'SX', 'SY', 'A', 'AX', 'AY', 
             'Position', 'GameId', 'PlayId', 'PlayerCollegeName', 'Dis',
             'XToScrimmage', 'PlayerDistToEndZone', 'OrientationStd', 'OrientationCos',
             'OrientationSin', 'VoronoiArea', 'VoronoiAreaNoOffence']
        ]
        X = X.merge(rusher_pos, on=['GameId', 'PlayId'], suffixes=['', 'Rusher'])
        if 'PositionRusher' not in self.one_hot_vars:
            self.one_hot_vars += ['PositionRusher']
        
        # Relationship of other players to rusher
        X['XFromRusher'] = abs(X['XStd'] - X['XStdRusher'])
        X['YFromRusher'] = abs(X['YStd'] - X['YStdRusher'])
        X['DistFromRusher'] = np.sqrt(np.square(X['XFromRusher']) 
                                      + np.square(X['YFromRusher'])) 
        
        X['TimeToRusher'] = X['DistFromRusher'] / X['S']
        X.loc[np.isinf(X['TimeToRusher']), 'TimeToRusher'] = 1000
        
        # Force and momentum
        X['Force'] = X['PlayerWeight'] * X['A']
        X['Momentum'] = X['PlayerWeight'] * X['S']
        
        cols_to_drop = ['NflId', 'NflIdRusher', 'Team', 'PlayDirection', 
                        'FieldPosition', 'PossessionTeam', 'OrientationCos',
                        'OrientationSin', 'VoronoiAreaNoOffence', 
                        'PlayerDistToEndZone']
        X.drop(columns=cols_to_drop, inplace=True)
        return X
    
    @transformation_check
    def _apply_one_hot_encoder(self, X):
        col_names = []
        print('\tdropping columns: {} for having cardinality > {}'
              .format(' '.join(self.oh_cols_to_drop), self.max_cardinality))
        X.drop(columns=self.oh_cols_to_drop, inplace=True)
        self.player_cols = [col for col in self.player_cols
                            if col not in self.oh_cols_to_drop]
        print('\tone hot encoding columns: {}'.format(' '.join(self.one_hot_vars)))
        X_1h = self.one_hot_encoder.transform(X[self.one_hot_vars].fillna('unknown'))

        for i, col in enumerate(self.one_hot_vars):
            new_var_names =                 [col + '_' + val for val in self.one_hot_encoder.categories_[i]]
            col_names += new_var_names
            if col in self.player_cols:
                self.player_cols.remove(col)
                self.player_cols += new_var_names

        X_1h = pd.DataFrame(data=X_1h, index=X.index, columns=col_names)
        X = pd.concat([X.drop(columns=self.one_hot_vars), X_1h], axis=1)
        return X

    @transformation_check
    def _flatten_player_vars(self, X):
        
        # Cols to group
        college_mask = ['CollegeConference' in col for col in X.columns]
        college_cols = X.columns[college_mask]
        college_agg = X.groupby(['GameId', 'PlayId'])[college_cols].sum()
        X.drop(columns=college_cols, inplace=True)
        
        mechanics_cols = [
            'X', 'XStd', 'Y', 'YStd', 'A', 'S', 'Dir', 'DirRad', 'DirStd',
            'PlayerHeight', 'PlayerWeight', 'PlayerAge', 'Force',
            'Momentum', 'OrientationStd', 'Dis', 'XFromRusher', 'XToScrimmage',
            'YFromRusher', 'DistFromRusher', 'AX', 'AY', 'SX', 'SY', 'TimeToRusher'
        ]
        mech_agg = X.groupby(['GameId', 'PlayId', 'InPossesion'])[mechanics_cols]            .agg(['mean', 'std'])
        mech_agg.columns = ['_'.join(col).strip() for col in mech_agg.columns.values]
        mech_agg.reset_index(inplace=True)
        mech_agg.set_index(['GameId', 'PlayId'], inplace=True)
        mech_agg['InPossesion'] = mech_agg['InPossesion'].map({1: 'off', 0: 'def'})
        mech_agg = mech_agg.pivot(columns='InPossesion')
        mech_agg.columns = ['_'.join(col).strip() for col in mech_agg.columns.values]
        mech_agg.fillna(0, inplace=True)
        
        college_like_cols = [
            'PlayerCollegeName', 'JerseyNumber'
        ]
        coll_mean = X.groupby(['GameId', 'PlayId', 'InPossesion'])[college_like_cols]            .agg(['mean'])
        coll_mean.columns = ['_'.join(col).strip() for col in coll_mean.columns.values]
        coll_mean.reset_index(inplace=True)
        coll_mean.set_index(['GameId', 'PlayId'], inplace=True)
        coll_mean['InPossesion'] = coll_mean['InPossesion'].map({1: 'off', 0: 'def'})
        coll_mean = coll_mean.pivot(columns='InPossesion')
        coll_mean.columns = ['_'.join(col).strip() for col in coll_mean.columns.values]
        coll_mean.fillna(0, inplace=True)
        
        X.drop(columns=college_like_cols, inplace=True)
        
        mechanics_cols += ['VoronoiArea']
        mech_agg_pos = X.groupby(['GameId', 'PlayId', 'PositionGroup', 'InPossesion'])[mechanics_cols]            .agg(['mean', 'std'])
        mech_agg_pos.columns = ['_'.join(col).strip() for col in mech_agg_pos.columns.values]
        mech_agg_pos.reset_index(inplace=True)
        mech_agg_pos.set_index(['GameId', 'PlayId'], inplace=True)
        mech_agg_pos['PositionPossession'] = mech_agg_pos['InPossesion'].map({1: 'off', 0: 'def'})             + '_' + mech_agg_pos['PositionGroup']
        mech_agg_pos.drop(columns=['InPossesion', 'PositionGroup'], inplace=True)
        mech_agg_pos = mech_agg_pos.pivot(columns='PositionPossession')
        mech_agg_pos.columns = ['_'.join(col).strip() for col in mech_agg_pos.columns.values]
        mech_agg_pos.fillna(0, inplace=True)
        
        X.drop(columns=mechanics_cols, inplace=True)
        
        away_ind = X.groupby(['GameId', 'PlayId'])['AwayInPosession', 'ScoreDiff'].max()
        away_ind['PosessionTeamLeading'] = 0
        mask = ((away_ind['AwayInPosession'] == 1) & (away_ind['ScoreDiff'] < 0)) |             ((away_ind['AwayInPosession'] == 0) & (away_ind['ScoreDiff'] > 0))
        away_ind.loc[mask, 'PosessionTeamLeading'] = 1
        away_ind['PosessionTeamLead'] = away_ind['ScoreDiff']
        away_ind.loc[away_ind['AwayInPosession'] == 1, 'PosessionTeamLead'] =             -away_ind.loc[away_ind['AwayInPosession'] == 1, 'PosessionTeamLead']
        away_ind['GameWithinConvTouchdown'] = away_ind['ScoreDiff'].abs() <= 8
        
        # Cols to ignore
        ignore_cols = ['IsAwayTeam', 'InPossesion', 'AwayInPosession', 
                       'PositionGroup', 'ScoreDiff', 'Orientation']
        X.drop(columns=ignore_cols, inplace=True)
        
        # Cols to spread wide
        self.player_cols = [col for col in self.player_cols 
                            if col not in college_cols
                            and col not in mechanics_cols
                            and col not in college_like_cols
                            and col not in ignore_cols]
        n_player_cols = len(self.player_cols)
        if self.player_cols:
            player_data = X[self.player_cols].values.reshape(-1, n_player_cols * 22)
            new_col_names = [col + '_' + str(player_num)
                             for player_num in range(22)
                             for col in self.player_cols]

            player_data = pd.DataFrame(
                data=player_data,
                columns=new_col_names
            )
            player_data = player_data.infer_objects()

        X.drop(columns=self.player_cols, inplace=True)
        X = X.drop_duplicates().reset_index(drop=True)
        
        if self.player_cols:
            X = pd.concat([X, player_data], axis=1)
            
        X.set_index(['GameId', 'PlayId'], inplace=True)
        X = X.merge(college_agg, how='left', left_index=True, right_index=True, suffixes=['', 'College'])
        X = X.merge(mech_agg, how='left', left_index=True, right_index=True, suffixes=['', 'TeamMech'])
        X = X.merge(coll_mean, how='left', left_index=True, right_index=True, suffixes=['', 'PlayerCollege'])
        X = X.merge(mech_agg_pos, how='left', left_index=True, right_index=True, suffixes=['', 'TeamPosMech'])
        X = X.merge(away_ind, how='left', left_index=True, right_index=True, suffixes=['', 'GameInds'])   
        return X
    
    @transformation_check
    def _previous_play_data(self, X):
        X[['YardLineLastRush', 'GameClockLastRush', 'DistanceLastRush']] =             X.groupby(['GameId', 'PlayId'])['YardLineStd', 'GameClock', 'Distance'].shift()
        X['TimeSinceLastRush'] = X['GameClock'] - X['GameClockLastRush']
        X.drop(columns=['YardLineLastRush', 'GameClockLastRush', 'DistanceLastRush'], inplace=True)
        return X
    
    def _get_player_cols(self, X):
        max_vals = X.groupby(['GameId', 'PlayId']).nunique().max()
        self.player_cols = max_vals[max_vals > 1].index.tolist()
        self.team_cols = max_vals[max_vals == 2].index.tolist()


# In[12]:


class DropColinear(TransformerMixin):
    def __init__(self, max_corr=1):
        self.max_corr = max_corr
        self.all_corellated_cols = []
        
    def fit(self, X, y=None):
        corr = X.corr()
        corr = pd.DataFrame(np.triu(corr), columns=corr.columns, index=corr.index)
        for col in corr.index:
            correlated_cols = corr.columns[corr[col].abs() >= self.max_corr].tolist()
            correlated_cols = [c_col for c_col in correlated_cols if c_col != col]
            if correlated_cols:
                self.all_corellated_cols += correlated_cols
        self.all_corellated_cols = np.unique(self.all_corellated_cols)
        return self
    
    def transform(self, X):
        print('Dropping following columns for having a correlation of over {} with '
              'another variable:\n{}'.format(self.max_corr, ', '.join(self.all_corellated_cols)))
        X.drop(columns=self.all_corellated_cols, inplace=True)
        return X


# In[13]:


processor = Preprocessor(
    one_hot_vars=VARS_TO_ONE_HOT,
    max_cardinality=MAX_CARDINALITY,
    cols_to_drop=COLS_TO_DROP
)
print(train.shape)
train = processor.initial_cleaning(train)
train = processor.fit_transform(train)


# In[14]:


dropper = DropColinear(max_corr=1)
train = dropper.fit_transform(train)
print(train.shape)


# In[15]:


from sklearn.base import BaseEstimator, TransformerMixin

class FeatureChecker(TransformerMixin):
    
    def fit(self, X):
        self.std = X.std(axis=0)
        self.mean = X.mean(axis=0)
        
    def transform(self, X):
        new_mean = X.mean(axis=0)
        abs_diff = np.absolute(new_mean - self.mean)
        if any(abs_diff > self.std):
            cols = np.arange(X.shape[1])
            print('Following columns are over 1 std dev out form training: {}'
                  .format(cols[abs_diff > self.std]))
        return X


# In[16]:


train.head()


# In[17]:


train.isnull().sum().sort_values(ascending=False)[:5]


# In[18]:


plt.scatter(train['VoronoiAreaNoOffenceRusher'], train['Yards'], alpha=.3)


# In[19]:


yards = train.pop('Yards')
season = train.pop('Season')
season_weights = season.map({2017: .8, 2018: 1.2, 2019: 1.2}).fillna(1).values


# In[20]:


fill_val = train.mean()


# In[21]:


X = train.copy()

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1


# In[22]:


col_order = sorted(train.columns)
X = X[col_order]
scaler = StandardScaler()
X = scaler.fit_transform(X.fillna(fill_val))


# In[23]:


class CRPSCallback(Callback):
    
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        
        print('validation shape',len(self.validation))

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
            
        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s


# In[24]:


N_TRAIN = X.shape[0]
BATCH_SIZE = 1024
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001,
    decay_steps=STEPS_PER_EPOCH*100,                                                         
    decay_rate=1,
    staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

optimizer = get_optimizer()

def get_model(x_tr,y_tr,x_val,y_val,w_tr,dropouts):
    
    inp = Input(shape = (x_tr.shape[1],))
    x = Dropout(0.8)(inp)
    x = Dense(64, input_dim=X.shape[1], activation='relu',
             kernel_regularizer=regularizers.l1(0.01))(inp)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu',
             kernel_regularizer=regularizers.l1(0.01))(x)
    x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=[]
    )

    es = EarlyStopping(
        monitor='CRPS_score_val', 
        mode='min',
        restore_best_weights=True, 
        verbose=1, 
        patience=10
    )

    mc = ModelCheckpoint(
        'best_model.h5',
        monitor='CRPS_score_val',
        mode='min',
        save_best_only=True, 
        verbose=1, 
        save_weights_only=True
    )
    
    bsz = 1024
    steps = x_tr.shape[0] / bsz
    
    history = model.fit(
        x_tr, 
        y_tr,
        callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], 
        epochs=100, 
        batch_size=bsz,
        sample_weight=w_tr,
        verbose=1
    )
    model.load_weights("best_model.h5")
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_val.shape[0])
    crps = np.round(val_s, 6)

    return {'loss': crps, 'status': STATUS_OK, 'model': model}, history


# In[25]:


def get_data(X, y, seed):
    x_tr, y_tr, x_val, y_val = train_test_split(X, y, test_size=0.15, random_state=seed)
    return x_tr, y_tr, x_val, y_val


# In[26]:


losses = []
models = []
crps_csv = []
s_time = time.time()

for k in range(2):
    kfold = KFold(5, random_state=SEED + k, shuffle=True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
        print("-----------")
        print("-----------")
        tr_x, tr_y = X[tr_inds],y[tr_inds]
        val_x, val_y = X[val_inds],y[val_inds]
        w_tr = season_weights[tr_inds]
        results, history = get_model(tr_x,tr_y,val_x,val_y,w_tr,dropouts=[.4, .5, .5, .5])
        models.append(results['model'])
        print("the %d fold crps is %f"%((k_fold+1), results['loss']))
        print("mean crps is %f"%np.mean(crps_csv))
        crps_csv.append(results['loss'])

print("mean crps is %f"%np.mean(crps_csv))

def predict(x_te):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_te, batch_size=1024)
        else:
            y_pred += m.predict(x_te, batch_size=1024)
            
    y_pred = y_pred / model_num
    
    return y_pred


# In[27]:


print("mean crps is\t%f"%np.mean(crps_csv))
print("std crps is\t%f"%np.std(crps_csv))


# In[28]:


# 2 layers of 64, 0.8 initial drop out


# In[29]:


# Current Best Score

#print("mean crps is\t%f"%np.mean(crps_csv))
#print("std crps is\t%f"%np.std(crps_csv))


# In[30]:


from kaggle.competitions import nflrush

names = dict(zip(range(199), ['Yards%d' % i for i in range(-99, 100)]))

env = nflrush.make_env()
for i, (df_test, sample_pred) in enumerate(env.iter_test()):
    test = processor.initial_cleaning(df_test)
    test = processor.transform(test) 
#     test = dropper.transform(test)
    
    for col in col_order:
        if col not in test.columns:
            test[col] = 0
            
    test = test[col_order]
    scaled_test = scaler.transform(test.fillna(fill_val))   
    y_pred = predict(scaled_test)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

    preds_df = pd.DataFrame(data=[y_pred], columns=sample_pred.columns)
    if i == 0:
        all_preds = preds_df
        all_test_rows = test
    else:
        all_preds = pd.concat([all_preds, preds_df], ignore_index=True, sort=False)
        all_test_rows = pd.concat([all_test_rows, test], ignore_index=True, sort=False)
    env.predict(preds_df)
all_test_rows.to_csv('X_test.csv')
env.write_submission_file()


# In[ ]:




