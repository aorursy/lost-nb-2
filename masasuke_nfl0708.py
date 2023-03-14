#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import KFold,GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import sklearn.metrics as mtr
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, merge, Add
from keras.layers.embeddings import Embedding
import tensorflow as tf
import lightgbm as lgb
from IPython.core.display import display

print(keras.__version__)
print(tf.__version__)


# In[2]:


# nfl_data_reader.py
def nfl_read_train():
    if os.path.isfile('./input/nfl-big-data-bowl-2020/train.csv.gz'):
        # for vscode(pickle)
        return pd.read_pickle('./input/nfl-big-data-bowl-2020/train.csv.gz')
    if os.path.isfile('./input/nfl-big-data-bowl-2020/train.csv'):
        # for vscode(csv)
        X = pd.read_csv('./input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
        pd.to_pickle(X, './input/nfl-big-data-bowl-2020/train.csv.gz')
        return X
    # for Kaggle
    return pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

def nfl_read_test():
    if os.path.isfile('./input/nfl-big-data-bowl-2020/test.csv.gz'):
        # for vscode(pickle)
        return pd.read_pickle('./input/nfl-big-data-bowl-2020/test.csv.gz')
    if os.path.isfile('./input/nfl-big-data-bowl-2020/test.csv'):
        # for vscode(csv)
        return pd.read_csv('./input/nfl-big-data-bowl-2020/test.csv', low_memory=False)
    return None

# nfl_lgb_extraction.py
class ModelExtractionCallback(object):
    def __init__(self):
        self._model = None

    def __call__(self, env):
        # _CVBooster �̎Q�Ƃ�ێ�����
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        return self._model

    @property
    def raw_boosters(self):
        self._assert_called_cb()
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        return self._model.best_iteration

# nfl_metric.py
def nfl_eval_crps(labels, predictions):
    # from https://www.kaggle.com/zero92/lbgm-eval-metric-crps
    scaler = preprocessing.StandardScaler()
    y_pred = np.zeros((len(labels), 199))
    y_ans = np.zeros((len(labels), 199))
    j = np.array(range(199))
    for i,(p,t) in enumerate(zip(np.round(scaler.inverse_transform(predictions)),labels)) :
        k2 = j[j>=p-10]
        y_pred[i][k2]=(k2+10-p)*0.05
        k1 = j[j>=p+10]
        y_pred[i][k1]= 1.0
        k3 = j[j>=t]
        y_ans[i][k3]= 1.0
                           
    return 'CRPS', np.sum((y_pred - y_ans)**2)/(199 * y_pred.shape[0]), False


def nfl_create_cdf(X):
    # from https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm
    print(X.shape)
    y_ = np.array([X["Yards"][i] for i in range(0, len(X), 22)])
    print(y_.shape)
    scaler = preprocessing.StandardScaler()
    scaler.fit(y_.reshape(-1, 1))
    return scaler.transform(y_.reshape(-1, 1)).flatten()


def nfl_create_yards_cdf(yard:int):
    y_train = np.zeros(shape=(1, 199))
    y_train[0, yard+99:] = np.ones(shape=(1, 100-yard))
    return y_train


def nfl_calc_crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])

# nfl_metric_keras.py
class NFL_NN_Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_train[-1].shape[0])
        tr_s = np.round(tr_s, 6)
        logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid[-1].shape[0])
        val_s = np.round(val_s, 6)
        logs['val_CRPS'] = val_s
        print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)

# nfl_plot_field.py
def nfl_show_play(X, play_id):
    def create_football_field(linenumbers=True,
                            endzones=True,
                            highlight_line=True,
                            highlight_line_number=50,
                            highlighted_name='Line of Scrimmage',
                            fifty_is_los=False,
                            figsize=(12, 6.33)):
        # https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position
        """
        Function that plots the football field for viewing plays.
        Allows for showing or hiding endzones.
        """
        rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                                edgecolor='r', facecolor='darkgreen', zorder=0)

        fig, ax = plt.subplots(1, figsize=figsize)
        ax.add_patch(rect)

        plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
                80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
                [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
                53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
                color='white')
        if fifty_is_los:
            plt.plot([60, 60], [0, 53.3], color='gold')
            plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
        # Endzones
        if endzones:
            ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            ax.add_patch(ez1)
            ax.add_patch(ez2)
        plt.xlim(0, 120)
        plt.ylim(-5, 58.3)
        plt.axis('off')
        if linenumbers:
            for x in range(20, 110, 10):
                numb = x
                if x > 50:
                    numb = 120 - x
                plt.text(x, 5, str(numb - 10),
                        horizontalalignment='center',
                        fontsize=20,  # fontname='Arial',
                        color='white')
                plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                        horizontalalignment='center',
                        fontsize=20,  # fontname='Arial',
                        color='white', rotation=180)
        if endzones:
            hash_range = range(11, 110)
        else:
            hash_range = range(1, 120)

        for x in hash_range:
            ax.plot([x, x], [0.4, 0.7], color='white')
            ax.plot([x, x], [53.0, 52.5], color='white')
            ax.plot([x, x], [22.91, 23.57], color='white')
            ax.plot([x, x], [29.73, 30.39], color='white')

        if highlight_line:
            hl = highlight_line_number + 10
            plt.plot([hl, hl], [0, 53.3], color='yellow')
            plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                    color='yellow')
        return fig, ax


    def get_dx_dy(angle, dist):
        radian_angle = angle * math.pi/180.0
        dx = dist * math.cos(radian_angle)
        dy = dist * math.sin(radian_angle)
        return dx, dy


    df = X[X.PlayId == play_id]

    df['ToLeft'] = df.PlayDirection == "left"
    df['IsBallCarrier'] = df.NflId == df.NflIdRusher
    df['Dir_std'] = np.mod(90 - df.Dir, 360)

    fig, ax = create_football_field()
    ax.scatter(df.X, df.Y, cmap='rainbow', c=~(df.Team == 'home'), s=100)
    rusher_row = df[df.NflIdRusher == df.NflId]
    ax.scatter(rusher_row.X, rusher_row.Y, color='black')
    yards_covered = rusher_row["Yards"].values[0]
    x = rusher_row["X"].values[0]
    y = rusher_row["Y"].values[0]
    rusher_dir = rusher_row["Dir_std"].values[0]
    rusher_speed = rusher_row["S"].values[0]
    dx, dy = get_dx_dy(rusher_dir, rusher_speed)

    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3, color='black')
    left = 'left' if df.ToLeft.sum() > 0 else 'right'
    plt.title(f'Play # {play_id} moving to {left}, yard distance is {yards_covered}', fontsize=20)
    plt.legend()
    plt.show()

# nfl_trn_bins_yard.py
class NFL_trn_BinsYard(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        bins = [0,2,4,6,8,10,15,20,30,40,50,60,70,80,90,100]    
        # Create YardsToGo columuns
        X['YardsToGo'] = X[['FieldPosition','PossessionTeam','YardLine']].apply(         lambda x: (50-x['YardLine'])+50 if x['PossessionTeam']==x['FieldPosition'] else x['YardLine'],1)    

        # Binning        
        cut_YardsToGo = pd.cut(X['YardsToGo'],bins = bins)
        yard_bins = pd.get_dummies(cut_YardsToGo)
        yard_bins.columns = [str(i) for i in yard_bins.columns.tolist()]    

        # merge to X
        X = pd.merge(X,yard_bins,left_index = True, right_index = True)
        X.columns = [str(x).replace(']','').replace('(','') for x in X.columns]        

        return X

# nfl_trn_cat_one_hot.py
class NFL_trn_CatOneHot(BaseEstimator, TransformerMixin):
    def __init__(self, cat_list, cat_column, cat_onehot_prefix=None):
        cat_list = pd.Series(cat_list).unique()
        cat_list.sort()
        if cat_onehot_prefix==None:
            cat_onehot_prefix = cat_column
        
        rename_dic = {cat_list[i]: cat_onehot_prefix + '_' + str(i) for i in range(len(cat_list))}
        dummies = pd.get_dummies(cat_list)
        dummies = dummies.rename(columns=rename_dic)

        self.cat_column = cat_column
        self.onehot_table = pd.DataFrame(cat_list, columns=[cat_column])
        self.onehot_table = pd.concat([self.onehot_table, dummies], axis=1)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.merge(X, self.onehot_table, how='left', left_on=[self.cat_column], right_on=[self.cat_column])


class NFL_trn_CatOneHot2(BaseEstimator, TransformerMixin):
    def __init__(self, cat_series, cat_onehot_prefix=None, drop_org=True):
        cat_se = cat_series.copy().drop_duplicates().dropna().sort_values()
        cat_list = list(cat_se.values)
        cat_onehot_prefix = cat_se.name
        
        rename_dic = {cat_list[i]: cat_onehot_prefix + '_' + str(i) for i in range(len(cat_list))}
        dummies = pd.get_dummies(cat_list)
        dummies = dummies.rename(columns=rename_dic)
        
        self.columns = dummies.columns
        self.drop_org = drop_org
        self.cat_column = cat_se.name
        self.onehot_table = pd.DataFrame(cat_list, columns=[cat_se.name])
        self.onehot_table = pd.concat([self.onehot_table, dummies], axis=1)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.merge(X, self.onehot_table, how='left', left_on=[self.cat_column], right_on=[self.cat_column])
        if self.drop_org:
            X.drop(self.cat_column, axis=1, inplace=True)
        for col in self.columns:
            X[col].fillna(0, inplace=True)
            X[col] = X[col].astype('uint8')
        return X


class NFL_trn_FillNaForCat(BaseEstimator, TransformerMixin):
    def __init__(self, cat_column=[]):
        self.cat_column = cat_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.cat_column:
            X[col] = X[col].astype('object')
            X[col].fillna('Unknown', inplace=True)
            X[col] = X[col].astype('category')
        return X


class NFL_trn_FillNaMean(BaseEstimator, TransformerMixin):
    def __init__(self, cat_column=[]):
        self.cat_column = cat_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.cat_column:
            X[col] = X[col].astype('object')
            X[col].fillna('Unknown', inplace=True)
            X[col] = X[col].astype('category')
        return X

# nfl_trn_ch_same_dir.py
class NFL_trn_ChangeSameDirection(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['ToLeft'] = X['PlayDirection'] == "left"
        # X
        X.loc[df['ToLeft'], 'X'] = 120 - df.loc[df['ToLeft'], 'X']
        # Y
        X.loc[df['ToLeft'], 'Y'] = 160/3 - df.loc[df['ToLeft'], 'Y']
        # Dir
        X.loc[df['ToLeft'], 'Dir'] = (df.loc[df['ToLeft'], 'Dir']+180) % 360
        # Orientation
        X.loc[df['ToLeft'], 'Orientation'] = (df.loc[df['ToLeft'], 'Orientation']+180) % 360
        # PlayDirection
        X['PlayDirection'] = X['PlayDirection'].astype('object')
        X.loc[df['ToLeft'], 'PlayDirection'] = 'right'
        X['PlayDirection'] = X['PlayDirection'].astype('category')
        # YardLine
        X.loc[df['ToLeft'], 'YardLine'] =  100 - df.loc[df['ToLeft'], 'YardLine']
        return X

# nfl_trn_data_selector.py
def get_categorical_column(X, res_no_cat=""):
    res = []
    for col in X.columns:
        if type(X[col].dtype)==pd.core.dtypes.dtypes.CategoricalDtype:
            res.append(col)
    
    if len(res)<=0:
        res = res_no_cat
        
    return res


def get_numeric_column(X, res_no_num=""):
    res = []
    numerics = [ 'int8',  'int16',  'int32',  'int64', 'float16', 'float32', 'float64',
                'uint8', 'uint16', 'uint32', 'uint64']
    for col in X.columns:
        col_type = X[col].dtypes
        if col_type in numerics:
            res.append(col)
    return res


class NFL_trn_DataSelector(BaseEstimator, TransformerMixin):
    def __init__(self, select_columns, number_only=False):
        self.select_columns = select_columns
        self.number_only    = number_only

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in df.columns:
            if not col in self.select_columns:
                df.drop(col, axis=1, inplace=True)
        if self.number_only:
            return df.select_dtypes(include='number')
        return df


class NFL_trn_DataDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_columns=[], remain_columns=[]):
        self.drop_columns = drop_columns
        self.remain_columns = remain_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for attribute in self.drop_columns:
            if attribute in df.columns:
                df.drop(attribute, axis=1, inplace=True)
        for attribute in df.columns:
            if df[attribute].dtype=='object' and not attribute in self.remain_columns:
                df.drop(attribute, axis=1, inplace=True)
        return df


class NFL_trn_NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, remain_categories=True):
        self.remain_categories = remain_categories

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for attribute in df.columns:
            if not np.issubdtype(X[attribute].dtype, np.number):
                if True==self.remain_categories and type(X[attribute].dtype)==pd.core.dtypes.dtypes.CategoricalDtype:
                    continue
                df.drop(attribute, axis=1, inplace=True)
        return df


class NFL_trn_DataSelector_OneRowEachPlayId(BaseEstimator, TransformerMixin):
    def __init__(self, remain_categories=True):
        self.remain_categories = remain_categories

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        return df[df['NflId'] == df['NflIdRusher']]

# nfl_trn_data_types_cleaner.py
class NFL_trn_DataTypesCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        categories = [
            'Team',          'JerseyNumber',      'Season',  'PossessionTeam',    
            'FieldPosition', 'OffenseFormation',
            'PlayDirection', 'PlayerCollegeName', 
            'Position',      'HomeTeamAbbr',      'VisitorTeamAbbr',   
            'Stadium',       'Location',          'StadiumType',       
            'Turf',          'GameWeather',       'WindDirection',
            'NflId',         'NflIdRusher',
            'rusherPosition'
        ]
        
        if self.verbose:
            start_mem = X.memory_usage().sum() / 1024**2

        for col in X.columns:
            col_type = X[col].dtypes
            if col_type in numerics:
                c_min = X[col].min()
                c_max = X[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        X[col] = X[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        X[col] = X[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        X[col] = X[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        X[col] = X[col].astype(np.int64)
                else:
                    c_prec = X[col].apply(lambda x: np.finfo(x).precision).max()
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:
                        X[col] = X[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                        X[col] = X[col].astype(np.float32)
                    else:
                        X[col] = X[col].astype(np.float64)
            elif col in categories:
                X[col] = X[col].astype('category')

        if self.verbose:
            end_mem = X.memory_usage().sum() / 1024**2
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

        return X

# nfl_trn_defense_x_spread.py
class NFL_trn_DefenseXspead(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def get_defense_x_spread(X):
            df = X.copy()
            rusher = df[df['NflId'] == df['NflIdRusher']][['PlayId','Team']]
            rusher.columns = ['PlayId','RusherTeam']

            defense = pd.merge(df, rusher,on=['PlayId'],how='inner')
            defense = defense[defense['Team']!=defense['RusherTeam']][['PlayId','X']]
            defense = defense.groupby(['PlayId']).agg({'X':['min','max']}).reset_index()
            defense.columns = ['PlayId','def_min_X','def_max_X']
            defense['defense_x_spread'] = defense['def_max_X'] - defense['def_min_X']
            defense.drop(['def_min_X','def_max_X'], axis=1, inplace=True)
            return defense
        
        res = get_defense_x_spread(X)
        return X.merge(res, on='PlayId')

# nfl_trn_distance_to_centroid.py
class NFL_trn_DistanceToCentroid(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def euclidean_distance(x1, y1, x2, y2):
                x_diff = (x1-x2)**2
                y_diff = (y1-y2)**2
                return np.sqrt(x_diff + y_diff)
        
        def get_centroid(df):
            df = X.copy()
            rusher = df[df['NflId'] == df['NflIdRusher']][['PlayId','Team','X','Y']]
            rusher.columns = ['PlayId','RusherTeam','RusherX','RusherY']

            base = pd.merge(df,rusher,on=['PlayId'],how='inner')

            defense = base[base['Team']!=base['RusherTeam']][['PlayId','X','Y','RusherX','RusherY']]
            defense = defense.groupby(['PlayId']).agg({'X':['mean'], 'Y':['mean']}).reset_index()
            defense.columns = ['PlayId','cent_defence_X','cent_defence_Y']

            offence = base[base['Team']==base['RusherTeam']][['PlayId','X','Y','RusherX','RusherY']]
            offence = offence.groupby(['PlayId']).agg({'X':['mean'], 'Y':['mean']}).reset_index()
            offence.columns = ['PlayId','cent_offence_X','cent_offence_Y']

            base = base[['PlayId', 'RusherX','RusherY']]
            base = base.merge(defense, on=['PlayId'], how='inner')
            base = base.merge(offence, on=['PlayId'], how='inner')
            base.drop_duplicates('PlayId', inplace=True)
            base['distance_to_centroid']         =                     base[['RusherX','RusherY','cent_defence_X','cent_defence_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
            base['distance_to_offence_centroid'] =                     base[['RusherX','RusherY','cent_offence_X','cent_offence_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
            
            return base[['PlayId', 'distance_to_centroid', 'distance_to_offence_centroid']]
        
        res = get_centroid(X)
        return X.merge(res, on='PlayId')

# nfl_trn_distance_to_qb.py
class NFL_trn_DistanceToQB(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def euclidean_distance(x1, y1, x2, y2):
            x_diff = (x1-x2)**2
            y_diff = (y1-y2)**2
            return np.sqrt(x_diff + y_diff)

        def get_distance_to_qb(X):
            df = X.copy()
            rusher = df[df['NflId'] == df['NflIdRusher']][['PlayId','Team','X','Y']]
            rusher.columns = ['PlayId','RusherTeam','RusherX','RusherY']

            base = pd.merge(df,rusher,on=['PlayId'],how='inner')
            offence = base[base['Team']==base['RusherTeam']][['PlayId','X','Y','Position']]
            offence = offence.query('Position == ["QB"]')
            offence = offence.groupby('PlayId').agg({'X':['mean'], 'Y':['mean']}).reset_index()
            offence.columns = ['PlayId', 'qbX', 'qbY']
            
            res = pd.merge(rusher, offence, on='PlayId')
            res['distance_to_qb'] = res[['RusherX','RusherY','qbX','qbY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
            return res[['PlayId', 'distance_to_qb']]
        
        res = get_distance_to_qb(X)
        return X.merge(res, on='PlayId')

# nfl_trn_fillna.py
class NFL_trn_FillNa(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if X[col].isnull().any():
                if hasattr(X[col], 'cat'):
                    X[col] = X[col].astype('object')
                    X[col].fillna('Unknown', inplace=True)
                    X[col] = X[col].astype('category')
                elif np.issubdtype(X[col].dtypes, np.number):
                    X[col].fillna(X[col].mean(), inplace=True)
        return X


class NFL_trn_FillNa2(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tgt_cols = ['X', 'Y', 'S', 'A', 'Dis', 'Orientation',
                'Dir', 'NflId', 'DisplayName', 'JerseyNumber', 'Season', 
                #'YardLine',
                'Quarter', 'GameClock', 'PossessionTeam', 'Down', 'Distance',
                'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay',
                'OffenseFormation', 'OffensePersonnel',
                'DefendersInTheBox', 'DefensePersonnel', 'PlayDirection', 'TimeHandoff',
                'TimeSnap', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate',
                'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr',
                'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather',
                'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']
        self.fill_val = np.zeros(len(self.tgt_cols))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = X.copy()
        values = np.vstack((self.fill_val, X[self.tgt_cols].values))
        df = pd.DataFrame(values, columns=self.tgt_cols)
        df.fillna(method='ffill', axis=0, inplace=True)
        self.fill_val = df.values[df.shape[0]-1, ]
        res[self.tgt_cols] = df.iloc[1:,].values
        return res

# nfl_trn_game_weather.py
class NFL_trn_GameWeather(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rain = ['Rainy', 'Rain Chance 40%', 'Showers',
                'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',
                'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain']
        overcast = ['Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 
                    'Cloudy, chance of rain', 'Coudy', 
                    'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',
                    'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',
                    'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',
                    'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 
                    'Mostly Cloudy', 'Partly Cloudy', 'Cloudy']
        clear = ['Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',
                'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',
                'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',
                'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',
                'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',
                'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny', 'Sunny, Windy']
        snow  = ['Heavy lake effect snow', 'Snow']
        none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']
        X['GameWeather'] = X['GameWeather'].replace(rain,'rain')
        X['GameWeather'] = X['GameWeather'].replace(overcast,'overcast')
        X['GameWeather'] = X['GameWeather'].replace(clear,'clear')
        X['GameWeather'] = X['GameWeather'].replace(snow,'snow')
        X['GameWeather'] = X['GameWeather'].replace(none,'none')
        X.fillna({'GameWeather': 'none'}, inplace=True)
        X['GameWeather'] = X['GameWeather'].astype('category')
        return X

# nfl_trn_normalized_moving_X.py
class NFL_trn_NormalizedMovingX(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform_org(self, X):
        def create_normalized_moving_X(df_rus):
            try:        
                rusFieldPosition = df_rus['FieldPosition']
                rusYardLine = df_rus['YardLine']
                rusTeam = df_rus['Team']
                rusHomeTeamAbbr = df_rus['HomeTeamAbbr']

                rusPlayDirection = df_rus['PlayDirection']
                if rusPlayDirection == 'left':
                    rusX = df_rus['X'] - 10                                    
                else:
                    rusX = 100 - (df_rus['X'] - 10) 

                # Calc remaining_yardline_X
                remaining_yardline_X = 0
                if rusTeam == 'home':
                    if rusFieldPosition == rusHomeTeamAbbr:
                        remaining_yardline_X = 100 - rusYardLine
                    else:
                        remaining_yardline_X = rusYardLine    
                else:
                    if rusFieldPosition != rusHomeTeamAbbr:
                        remaining_yardline_X = 100 - rusYardLine
                    else:
                        remaining_yardline_X = rusYardLine                

                # Calc moving X
                moving_X = rusX - remaining_yardline_X        

                # Calc normalized_moving_X                            
                normalized_moving_X = (moving_X / remaining_yardline_X)*100 
                if(normalized_moving_X > 100):
                    normalized_moving_X = 100
                if(normalized_moving_X < 0):
                    normalized_moving_X =0
                
                # Check for each data                
                # print('FPandHTA[{},{}],D[{}] {};r{}'.format(rusFieldPosition,rusHomeTeamAbbr,rusPlayDirection,rusTeam,remaining_yardline_X))                                                    
                # print('[rusX:remX]->[{}:{}]'.format(rusX,remaining_yardline_X))
                # print('mov={}[{}%]'.format(moving_X,normalized_moving_X))

                # Create returned dataframe
                returned_df = df_rus[["PlayId"]]
                returned_df["normalized_moving_X"] = normalized_moving_X                

            except Exception as e:
                print(e)
                print("error")
            
            return returned_df
            
        result_list = X[X['NflId'] == X['NflIdRusher']].apply(lambda x: create_normalized_moving_X(x),1)
        X = pd.merge(X,result_list,on=['PlayId'],how='inner')
        return X

    def transform(self, X):
        def get_rem_yard_line(rusTeam, rusFieldPosition, rusHomeTeamAbbr, rusYardLine):
            if rusTeam == 'home':
                if rusFieldPosition == rusHomeTeamAbbr:
                    remaining_yardline_X = 100 - rusYardLine
                else:
                    remaining_yardline_X = rusYardLine    
            else:
                if rusFieldPosition != rusHomeTeamAbbr:
                    remaining_yardline_X = 100 - rusYardLine
                else:
                    remaining_yardline_X = rusYardLine 
            return remaining_yardline_X
        
        def get_rus_x(x, PlayDirection):
            if PlayDirection == 'left':
                rusX = x - 10                                    
            else:
                rusX = 100 - (x - 10) 
            return rusX
        
        def get_moving_x(x, yardline, dir):
            if dir=='right':
                return 110 - x
            else:
                return x - 10
        
        def get_normalized_moving_X(X):
            df = X.copy()
            cols = ['PlayId','X','Team','YardLine','PlayDirection','FieldPosition','HomeTeamAbbr']
            rusher = df[df['NflId'] == df['NflIdRusher']][cols]
            rusher['remYardLine'] = rusher[['Team','FieldPosition','HomeTeamAbbr','YardLine']].apply(lambda x: get_rem_yard_line(x[0],x[1],x[2],x[3]), axis=1)
            rusher['rusX'] = rusher[['X','PlayDirection']].apply(lambda x: get_rus_x(x[0],x[1]), axis=1)
            rusher['moving_X'] = rusher[['rusX','remYardLine']].apply(lambda x: np.abs(x[0]-x[1]), axis=1)
            rusher['normalized_moving_X'] = rusher[['moving_X','remYardLine']].apply(lambda x: (x[0]/x[1])*100, axis=1)
            return rusher[['PlayId', 'normalized_moving_X']]
        
        res = get_normalized_moving_X(X)
        return X.merge(res, on='PlayId')

# nfl_trn_offence_lead.py
class NFL_trn_OffenseLead(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform_org(self, X):
        X['OffenseLead'] = X[['PossessionTeam','HomeTeamAbbr','HomeScoreBeforePlay','VisitorScoreBeforePlay']].apply(lambda x: x[2]-x[3] if x[0] == x[1] else x[3]-x[2], axis = 1)
        return X

    def transform(self, X):
        df = X.copy()
        df = df[df['NflId'] == df['NflIdRusher']]
        df = df[['PlayId', 'PossessionTeam','HomeTeamAbbr','HomeScoreBeforePlay','VisitorScoreBeforePlay']]
        df['OffenseLead'] = df.apply(lambda x: x[3]-x[4] if x[1]==x[2] else x[4]-x[3], axis=1)
        return X.merge(df[['PlayId', 'OffenseLead']], on='PlayId')

# nfl_trn_orientation.py
class NFL_trn_Orientation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[X['Season']==2017, 'Orientation'] = np.mod(90 + X.loc[X['Season']==2017, 'Orientation'], 360)
        return X

# nfl_trn_orientation_std.py
class NFL_trn_Orientation_std(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def std_orientation(x):
            if x == None:
                x = 0
            x %= 360
            if x>180:
                x = 360 - x
            return x

        X['Orientation_std'] = X['Orientation'].apply(std_orientation)
        return X

# nfl_trn_personnel_splitter.py
class NFL_trn_PersonnelSplitter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

        def personnel_features(df):
            personnel = df[['GameId','PlayId','OffensePersonnel','DefensePersonnel']].drop_duplicates()
            personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: split_personnel(x))
            personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: defense_formation(x))
            personnel['DefenceDL'] = personnel['DefensePersonnel'].apply(lambda x: x[0])
            personnel['DefenceLB'] = personnel['DefensePersonnel'].apply(lambda x: x[1])
            personnel['DefenceDB'] = personnel['DefensePersonnel'].apply(lambda x: x[2])
            personnel['DefenceOL'] = 0

            personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: split_personnel(x))
            personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: offense_formation(x))
            personnel['OffenceQB'] = personnel['OffensePersonnel'].apply(lambda x: x[0])
            personnel['OffenceRB'] = personnel['OffensePersonnel'].apply(lambda x: x[1])
            personnel['OffenceWR'] = personnel['OffensePersonnel'].apply(lambda x: x[2])
            personnel['OffenceTE'] = personnel['OffensePersonnel'].apply(lambda x: x[3])
            personnel['OffenceOL'] = personnel['OffensePersonnel'].apply(lambda x: x[4])
            personnel['OffenceDB'] = 0
            personnel['OffenceDL'] = 0
            personnel['OffenceLB'] = 0

            # Let's create some features to specify if the OL is covered
            personnel['OL_diff'] = personnel['OffenceOL'] - personnel['DefenceDL']
            personnel['OL_TE_diff'] = (personnel['OffenceOL'] + personnel['OffenceTE']) - personnel['DefenceDL']
            # Let's create a feature to specify if the defense is preventing the run
            # Let's just assume 7 or more DL and LB is run prevention
            personnel['run_def'] = (personnel['DefenceDL'] + personnel['DefenceLB'] > 6).astype(int)

            personnel.drop(['OffensePersonnel','DefensePersonnel'], axis=1, inplace=True)
            
            return personnel
        
        res = personnel_features(X)
        res.drop('GameId', axis=1, inplace=True)
        return X.merge(res, on='PlayId')

# nfl_trn_player_age.py
class NFL_trn_PlayerAge(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def calc_player_age(birth_day, season):
            birth_year = birth_day.split('/')[2]
            return season - int(birth_year)
        
        mapped_list = map(calc_player_age, X['PlayerBirthDate'], X['Season'])
        X['PlayerAge'] = list(mapped_list)
        return X

# nfl_trn_player_dist_defenders.py
class NFL_trn_DistDefenders(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def euclidean_distance(x1,y1,x2,y2):
            x_diff = (x1-x2)**2
            y_diff = (y1-y2)**2
            return np.sqrt(x_diff + y_diff)

        def defense_features(df):
            rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
            rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

            defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
            defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
            defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

            defense = defense.groupby(['GameId','PlayId'])                            .agg({'def_dist_to_back':['min','max','mean','std']})                            .reset_index()
            defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

            return defense
        
        res = defense_features(X)
        res.drop('GameId', axis=1, inplace=True)
        return X.merge(res, on='PlayId')

# nfl_trn_player_dist_rb2defenders.py
class NFL_trn_DistRB2Defenders(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        def euclidean_distance(x1, y1, x2, y2):
            x_diff = (x1-x2)**2
            y_diff = (y1-y2)**2
            return np.sqrt(x_diff + y_diff)
        
        def get_dist_rb2defense(X):
            df = X.copy()
            rusher = df[df['NflId'] == df['NflIdRusher']][['PlayId','Team']]
            rusher.columns = ['PlayId','RusherTeam']

            base = pd.merge(df,rusher,on=['PlayId'],how='inner')

            offence = base[base['Team']==base['RusherTeam']][['PlayId','X','Y','Position']]
            offence = offence.query('Position == ["RB", "FB", "HB", "TB"]')
            if len(offence)<=0:
                rusher['RB2Defence_avg'] = 10.218225670975324
                rusher['RB2Defence_var'] = 36.048544281959956
                return rusher[['PlayId', 'RB2Defence_avg', 'RB2Defence_var']]
                
            offence = offence.groupby('PlayId').agg({'X':['mean'], 'Y':['mean']}).reset_index()
            offence.columns = ['PlayId', 'rbX', 'rbY']
            defense = base[base['Team']!=base['RusherTeam']][['PlayId','X','Y']]
            
            res = defense.merge(offence, on='PlayId')
            res['dist'] = res[['X','Y','rbX','rbY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
            res = res.groupby(['PlayId']).agg({'dist':['mean', np.var]}).reset_index()
            res.columns = ['PlayId', 'RB2Defence_avg', 'RB2Defence_var']
            return res
        
        res = get_dist_rb2defense(X)
        return X.merge(res, on='PlayId')

# nfl_trn_player_height.py
class NFL_trn_PlayerHeight(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def calc_height_mm(heightYard):
            h = heightYard.split('-')
            h_feet = float(h[0])
            h_inch = float(h[1])
            return (h_feet * 304.8) + (h_inch * 25.4)
        
        mapped_list = map(calc_height_mm, X['PlayerHeight'])
        X['PlayerHeight'] = list(mapped_list)
        return X

# nfl_trn_player_relative_to_back.py
class NFL_trn_PlayerRelativeToBack(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

        def update_yardline(df):
            new_yardline = df[df['NflId'] == df['NflIdRusher']].copy()
            #display(new_yardline[['PossessionTeam','FieldPosition','YardLine']])
            #raise ValueError("error!")
            new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
            #new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x.loc['PossessionTeam'],x.loc[0:, 'FieldPosition'],x.loc[0:, 'YardLine']), axis=1)
            new_yardline = new_yardline[['GameId','PlayId','YardLine']]

            return new_yardline

        def update_orientation(df, yardline):
            df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
            df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
            df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

            df = df.drop('YardLine', axis=1)
            df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

            return df

        def back_direction(orientation):
            if orientation > 180.0:
                return 1
            else:
                return 0

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

            player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])                                            .agg({'dist_to_back':['min','max','mean','std']})                                            .reset_index()
            player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                    'min_dist','max_dist','mean_dist','std_dist']
            
            return player_distance
        
        df = X.copy()
        yardline = update_yardline(df)
        df = update_orientation(df, yardline)
        back_feats = back_features(df)
        res = features_relative_to_back(df, back_feats)
        res.drop('GameId', axis=1, inplace=True)
        X = X.merge(res, on='PlayId')
        X['back_oriented_down_field'] = X['back_oriented_down_field'].astype('category')
        X['back_moving_down_field'] = X['back_moving_down_field'].astype('category')
        return X

# nfl_trn_player_rusher.py
class NFL_trn_PlayerRusher(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rusherCol = ['S', 'A', 'Dis', 'Orientation', 'Dir', 
                    'PlayerHeight', 'PlayerWeight', 'PlayerAge', 'Position']
        df = X[X['NflId'] == X['NflIdRusher']]
        res = pd.DataFrame()
        res['PlayId'] = df['PlayId']
        for col in rusherCol:
            if col in df.columns:
                res[f'rusher{col}'] = df[col]
        
        res['rusherS_horizontal'] = df['S'] * np.sin(np.deg2rad(res['rusherDir']))
        res['rusherS_vertical']   = df['S'] * np.cos(np.deg2rad(res['rusherDir']))
        res['rusherY_std'] = df['Y'] - (53.3 / 2)
        
        X = X.merge(res, on='PlayId')
        if 'rusherPosition' in X.columns:
            X['rusherPosition'] = X['rusherPosition'].astype('category')
        
        return X

# nfl_trn_process_two.py
class NFL_trn_ProcessTwo(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def process_two(t_):
            t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) - np.square(t_.Y.values))))
            t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
            t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
            t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
            radian_angle = (90 - t_['Dir']) * np.pi / 180.0
            t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
            t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
            return t_
        
        return process_two(X)

# nfl_trn_runner_distance_to_los.py
class NFL_trn_RunnerDistance2los(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def new_line(rush_team, field_position, yardline):
            if rush_team == field_position:
                # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
                return 10.0 + yardline
            else:
                # half the field plus the yards between midfield and the line of scrimmage
                return 60.0 + (50 - yardline)

        def update_yardline(X):
            df = X.copy()
            new_yardline = df[df['NflId'] == df['NflIdRusher']].copy()
            new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
            return new_yardline
        
        def get_runner_dist_to_los(df):
            rusher = df[['PlayId','X','YardLine']].copy()
            rusher.columns = ['PlayId','RusherX','YardLine']
            rusher['runner_distance_to_los'] = rusher[['RusherX','YardLine']].apply(lambda x: np.abs(x[0] - x[1]), axis=1)
            return rusher.drop(['RusherX', 'YardLine'], axis=1)
        
        df = update_yardline(X)
        res = get_runner_dist_to_los(df)
        return X.merge(res, on='PlayId')

# nfl_trn_runner_vs_1stdefensor_speed.py
class NFL_trn_RunnerVs1stDefensorSpeed(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def euclidean_distance(x1,y1,x2,y2):
            x_diff = (x1-x2)**2
            y_diff = (y1-y2)**2
            return np.sqrt(x_diff + y_diff)
        
        def get_runner_vs_1stdefensor_speed(X):
            df = X.copy()
            rusher = df[df['NflId'] == df['NflIdRusher']][['PlayId','Team','X','Y','S']]
            rusher.columns = ['PlayId','RusherTeam','RusherX','RusherY','RusherS']
            
            defense = pd.merge(df,rusher,on='PlayId',how='inner')
            defense = defense[defense['Team'] != defense['RusherTeam']][['PlayId','RusherX','RusherY','RusherS','X','Y','S']]
            defense['dist'] = defense.apply(lambda x: euclidean_distance(x[1],x[2],x[4],x[5]), axis=1)
            dist_min = defense.groupby('PlayId').agg({'dist':['min']}).reset_index()
            dist_min.columns = ['PlayId', 'dist_min']
            defense = defense.merge(dist_min, on='PlayId')
            defense = defense[defense['dist']==defense['dist_min']]
            defense['runner_vs_1stdefensor_speed'] = defense['RusherS'] / defense['dist_min']
            return defense[['PlayId', 'runner_vs_1stdefensor_speed']]
        
        res = get_runner_vs_1stdefensor_speed(X)
        return X.merge(res, on='PlayId')

# nfl_trn_seconds_since_gamestart.py
class NFL_trn_SecondsSinceGameStart(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tmp = X.copy()    
        tmp['tmp'] = X['GameClock'].apply(lambda x: 15 * 60 - (int(x[0:2])*60 + int(x[3:5]))) 
        X['game_seconds_left'] = X['Quarter'].map({1:2700, 2:1800, 3:900, 4:0}) + tmp['tmp']            
        return X

# nfl_trn_seconds_since_start.py
class NFL_trn_SecondsSinceStart(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['seconds_since_start'] = X['GameClock'].apply(lambda x: 15 * 60 - (int(x[0:2])*60 + int(x[3:5]))) 
        return X

# nfl_trn_stadium_type.py
class NFL_trn_StadiumType(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        outdoor       = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 
                         'Outside', 'Outddors','Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']
        indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof',
                         'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']
        indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']
        dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']
        dome_open     = ['Domed, Open', 'Domed, open']
        X['StadiumType'] = X['StadiumType'].replace(outdoor,'outdoor')
        X['StadiumType'] = X['StadiumType'].replace(indoor_closed,'indoor_closed')
        X['StadiumType'] = X['StadiumType'].replace(indoor_open,'indoor_open')
        X['StadiumType'] = X['StadiumType'].replace(dome_closed,'dome_closed')
        X['StadiumType'] = X['StadiumType'].replace(dome_open,'dome_open')
        X.fillna({'StadiumType': 'Unknown'}, inplace=True)
        X['StadiumType'] = X['StadiumType'].astype('category')
        return X

# nfl_trn_team_abbr.py
class NFL_trn_TeamAbbr(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['VisitorTeamAbbr'] = X['VisitorTeamAbbr'].astype('object')
        X['HomeTeamAbbr'] = X['HomeTeamAbbr'].astype('object')
        X.loc[X.VisitorTeamAbbr == "ARI",'VisitorTeamAbbr'] = "ARZ"
        X.loc[X.HomeTeamAbbr == "ARI",'HomeTeamAbbr'] = "ARZ"
        X.loc[X.VisitorTeamAbbr == "BAL",'VisitorTeamAbbr'] = "BLT"
        X.loc[X.HomeTeamAbbr == "BAL",'HomeTeamAbbr'] = "BLT"
        X.loc[X.VisitorTeamAbbr == "CLE",'VisitorTeamAbbr'] = "CLV"
        X.loc[X.HomeTeamAbbr == "CLE",'HomeTeamAbbr'] = "CLV"
        X.loc[X.VisitorTeamAbbr == "HOU",'VisitorTeamAbbr'] = "HST"
        X.loc[X.HomeTeamAbbr == "HOU",'HomeTeamAbbr'] = "HST"
        X['VisitorTeamAbbr'] = X['VisitorTeamAbbr'].astype('category')
        X['HomeTeamAbbr'] = X['HomeTeamAbbr'].astype('category')
        return X

# nfl_trn_team_stat.py
class NFL_trn_TeamStat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def get_team_stat(df):
            df = X.copy()
            rusher = df[df['NflId'] == df['NflIdRusher']][['PlayId','Team']]
            rusher.columns = ['PlayId','RusherTeam']

            base = pd.merge(df,rusher,on=['PlayId'],how='inner')

            defense = base[base['Team']!=base['RusherTeam']][['PlayId','S','A']]
            defense = defense.groupby(['PlayId']).agg({'S':['mean'], 'A':['mean']}).reset_index()
            defense.columns = ['PlayId','S_defense_mean','A_defense_mean']

            offence = base[base['Team']==base['RusherTeam']][['PlayId','S','A']]
            offence = offence.groupby(['PlayId']).agg({'S':['mean'], 'A':['mean']}).reset_index()
            offence.columns = ['PlayId','S_offence_mean','A_offence_mean']

            res = pd.merge(offence, defense, on='PlayId')
            res['SpeedDifference'] = res['S_offence_mean'] - res['S_defense_mean']
            return res
        
        res = get_team_stat(X)
        return X.merge(res, on='PlayId')

# nfl_trn_time_to_tackle.py
class NFL_trn_TimeToTackle(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform_org(self, X):
        def get_time_to_tackle(df):
            rusher  = df[df['NflIdRusher']==df['NflId']].iloc[0]
            defense = df[df['Team']!=rusher['Team']]
            tm = []
            for i in range(len(defense)):
                dist  = np.linalg.norm(rusher[['X', 'Y']].values - defense.iloc[i][['X', 'Y']].values, ord=2)
                speed = defense.iloc[i]['S']
                if speed>0:
                    tm.append(dist/speed)
            return np.amin(tm), np.average(tm)
    
        res = X.groupby('PlayId').apply(get_time_to_tackle)
        res = pd.DataFrame(res).reset_index()
        res[['min_time_to_take_S', 'avg_time_to_take_S']] = res[0].apply(pd.Series)
        return X.merge(res.drop(0, axis=1), on='PlayId')
    
    def transform(self, X):
        def get_tackle_time(x1, y1, x2, y2, speed):
            x_diff = (x1-x2)**2
            y_diff = (y1-y2)**2
            return np.sqrt(x_diff + y_diff) / speed
        
        def get_tackle_time_df(X):
            df = X.copy()
            rusher = df[df['NflId'] == df['NflIdRusher']][['PlayId','Team','X','Y','S']]
            rusher.columns = ['PlayId','RusherTeam','RusherX','RusherY','RusherS']

            base = pd.merge(df, rusher,on=['PlayId'],how='inner')
            
            defense = base[base['Team']!=base['RusherTeam']][['PlayId','X','Y', 'S','RusherX','RusherY']]
            defense['tackle_time'] = defense.apply(lambda x: get_tackle_time(x[1],x[2],x[3],x[4],x[5]), axis=1)
            defense = defense.groupby('PlayId').agg({'tackle_time':['min','mean']}).reset_index()
            defense.columns = ['PlayId', 'min_time_to_take_S', 'avg_time_to_take_S']
            return defense

        res = get_tackle_time_df(X)
        return X.merge(res,  on='PlayId')

# nfl_trn_turf.py
class NFL_trn_Turf(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        natural_grass = ['natural grass','Naturall Grass','Natural Grass']
        grass         = ['Grass']
        fieldturf     = ['FieldTurf','Field turf','FieldTurf360','Field Turf']
        artificial    = ['Artificial','Artifical']
        X['Turf'] = X['Turf'].replace(natural_grass,'natural_grass')
        X['Turf'] = X['Turf'].replace(grass,'grass')
        X['Turf'] = X['Turf'].replace(fieldturf,'fieldturf')
        X['Turf'] = X['Turf'].replace(artificial,'artificial')
        X.fillna({'Turf': 'Unknown'}, inplace=True)
        X['Turf'] = X['Turf'].astype('category')
        return X

# nfl_trn_wind_direction.py
class NFL_trn_WindDirection(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def clean_wind_direction(windDirection):
            if not isinstance(windDirection, str):
                windDirection = '0'
            if windDirection.isnumeric():
                windDirection = '0'
            windDirection = windDirection.lower()
            windDirection = windDirection.replace('calm' , '0')
            windDirection = windDirection.replace('north', 'n')
            windDirection = windDirection.replace('south', 's')
            windDirection = windDirection.replace('east' , 'e')
            windDirection = windDirection.replace('west' , 'w')
            windDirection = windDirection.replace(' '    , '')
            windDirection = windDirection.replace('-'    , '')
            windDirection = windDirection.replace('/'    , '')
            if 'from' in windDirection:
                windDirection = windDirection.replace('from' , '')
                windDirection = windDirection.replace('n' , 's')
                windDirection = windDirection.replace('e' , 'w')
                windDirection = windDirection.replace('s' , 'n')
                windDirection = windDirection.replace('w' , 'e')
            return windDirection
        
        X['WindDirection'] = X['WindDirection'].apply(clean_wind_direction)
        X['WindDirection'] = X['WindDirection'].astype('category')
        return X

# nfl_trn_wind_speed.py
class NFL_trn_WindSpeed(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def clean_wind_speed(speed):
            if not isinstance(speed, str):
                return 0.
            speed = speed.lower()
            speed = speed.replace(' ', '')
            speed = speed.replace('mph', '')
            speed = speed.replace('calm', '0')
            speed = speed.replace('gustsupto', '-')
            if '-' in speed:
                tmp = speed.split('-')
                return (int(tmp[0]) + int(tmp[1])) / 2.
            if not speed.isnumeric():
                return 0.
            return int(speed)
        
        X['WindSpeed'] = X['WindSpeed'].apply(clean_wind_speed)
        X['WindSpeed'] = X['WindSpeed'].astype(np.float16)
        return X

# nfl_trn_x_std.py
class NFL_trn_Xstd(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        rusher = df[df['NflId'] == df['NflIdRusher']][['PlayId','Team',]]
        rusher.columns = ['PlayId','RusherTeam']
        defense = pd.merge(df,rusher,on=['PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['PlayId','X']]
        defense = defense.groupby(['PlayId']).agg({'X':['std']}).reset_index()
        defense.columns = ['PlayId','defence_x_std']
        X = X.merge(defense, on='PlayId')
        offence = pd.merge(df,rusher,on=['PlayId'],how='inner')
        offence = offence[offence['Team'] == offence['RusherTeam']][['PlayId','X']]
        offence = offence.groupby(['PlayId']).agg({'X':['std']}).reset_index()
        offence.columns = ['PlayId','offence_x_std']
        X = X.merge(offence, on='PlayId')
        return X

# nfl_clf_ensemble.py
class Nfl_clf_EnsembleVoting(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=None, flatten_transform=True):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
    
    def softmax(self, x):
        u = np.sum(np.exp(x))
        return np.exp(x)/u
        
    def fit(self, X, y, sample_weight=None):
        for name, clf in self.estimators:
            clf.fit(X, y)
        return self

    def predict(self, X):
        res = [clf.predict(X) for name, clf in self.estimators]
        return np.mean(res, axis=0)

# nfl_clf_lgb.py
class Nfl_clf_LightGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose=False):
        self.use_feat = [
            'GameId', 'PlayId', 'Team', 'Season', 'YardLine', 'Quarter', 
            'PossessionTeam', 'Down', 'Distance', 'HomeScoreBeforePlay', 
            'VisitorScoreBeforePlay', 'NflIdRusher', 'OffenseFormation', 
            'DefendersInTheBox', 'PlayDirection', 
            'Yards', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'Stadium', 'Location', 
            'StadiumType', 'Turf', 'GameWeather', 'Temperature', 'Humidity', 'WindSpeed',
            'OffenceDB', 'OffenceDL', 'OffenceLB', 'OffenceOL', 
            'OffenceQB', 'OffenceRB', 'OffenceTE', 'OffenceWR', 
            'DefenceDB', 'DefenceDL', 'DefenceLB', 'DefenceOL',
            'OL_diff', 'OL_TE_diff', 'run_def',
            'RB2Defence_avg', 'RB2Defence_var',
            'rusherS', 'rusherA', 'rusherDis', 'rusherOrientation', 'rusherDir', 
            'rusherPlayerHeight', 'rusherPlayerWeight', 'rusherPlayerAge',
            'rusherPlayerWeight', 'rusherPosition', 'min_time_to_take_S', 'avg_time_to_take_S',
            'rusherS_horizontal', 'rusherS_vertical',
            'seconds_since_start', 'distance_to_qb', 'offence_x_std', 'defence_x_std',
            'A_offence_mean', 'A_defense_mean', 'S_offence_mean', 'S_defense_mean', 'SpeedDifference',
            'normalized_moving_X', 'Orientation_std', 'runner_distance_to_los', 'runner_vs_1stdefensor_speed', 
            'defense_x_spread', 'rusherY_std', 'distance_to_centroid', 'distance_to_offence_centroid', 
            '0, 2', '2, 4', '4, 6', '6, 8', '8, 10', '10, 15', '15, 20', '20, 30',
            '30, 40', '40, 50', '50, 60', '60, 70', '70, 80', '80, 90', '90, 100',
            'OffenseLead', 'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist',
            'back_from_scrimmage', 'back_oriented_down_field',
            'back_moving_down_field', 'min_dist', 'max_dist', 'mean_dist', 'std_dist',
            'game_seconds_left',
            'fe1', 'fe5', 'fe7', 'fe8', 'fe10', 'fe11'
        ]
        self.verbose = verbose

    def fit(self, X, y):
        X_train = X.copy()
        self.pipeline_pre_predict_ = make_pipeline(
            NFL_trn_DataSelector_OneRowEachPlayId(),
            NFL_trn_DataSelector(select_columns=self.use_feat),
        )
        X_train = self.pipeline_pre_predict_.fit_transform(X_train)
        if self.verbose:
            X_train.info(max_cols=X_train.shape[1])
            display(datetime.datetime.now().isoformat())
        
        X = NFL_trn_DataDropper(drop_columns=['GameId', 'PlayId', 'Yards']).fit_transform(X_train)
        y = np.zeros((len(X_train['Yards']), 199))
        for idx, target in enumerate(list(X_train['Yards'].values)):
            y[idx][99 + target] = 1
        y = np.argmax(y, axis=1)

        model_extraction = ModelExtractionCallback()

        param = {
            #'num_iterations': 10, 
            'num_iterations': 1000, 
            'num_leaves': 50, #Original 50
            'min_data_in_leaf': 30, #Original 30
            'objective':'multiclass',
            'num_class': 199, # 199 possible places
            'max_depth': -1,
            'learning_rate': 0.01,
            "min_child_samples": 20,
            "boosting": "gbdt",
            "feature_fraction": 0.7, #0.9
            "bagging_freq": 1,
            "bagging_fraction": 0.9,
            "bagging_seed": 11,
            "metric": "multi_logloss",
            "lambda_l1": 0.1,
            "verbosity": -1,
            "seed":1234
        }
        args_common = {
            # for train and cv ##############
            "fobj": None,
            #"feval": EvalFunction,
            "init_model": None,
            "feature_name": 'auto',
            "categorical_feature": 'auto',
            "early_stopping_rounds": 10,
            "callbacks": [model_extraction, ],
        }
        args_cv = {
            # for cv ########################
            "folds": None,
            "nfold": 5,
            "stratified": True,
            "shuffle": True,
            #"metrics": 'multi_logloss',
            "fpreproc": None,
            "verbose_eval": 10,
            "show_stdv": True,
            "seed": 0,
            #"eval_train_metric": False,
        }

        res = lgb.cv(param, lgb.Dataset(X, y), **{**args_common, **args_cv})
        self.model_ = model_extraction.boosters_proxy
        if self.verbose:
            feature_importance = pd.DataFrame()
            feature_importance["feature"] = X.columns
            feature_importance["importance"] = np.array(self.model_.feature_importance()).mean(axis=0)
            feature_importance.sort_values("importance", inplace=True, ascending=False)
            plt.figure(figsize=(8, 12))
            sns.barplot(x="importance", y="feature", data=feature_importance[:50])
            plt.show()
            display(datetime.datetime.now().isoformat())
        
        return self

    def predict(self, X):
        X_test = X.copy()
        X_test = self.pipeline_pre_predict_.fit_transform(X_test)
        X = NFL_trn_DataDropper(drop_columns=['GameId', 'PlayId', 'Yards']).fit_transform(X_test)
        y_pred = self.model_.predict(X, num_iteration = self.model_.best_iteration)
        y_pred = np.array(y_pred)
        y_pred = np.mean(y_pred, axis=0)
        return y_pred

# nfl_clf_nn.py
class Nfl_clf_NN(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose=False):
        self.use_feat = ['GameId', 'PlayId', 'back_from_scrimmage',
            'back_oriented_down_field', 'back_moving_down_field', 'min_dist',
            'max_dist', 'mean_dist', 'std_dist', 'def_min_dist',
            'def_max_dist', 'def_mean_dist', 'def_std_dist', 
            'X', 'Y', 
            'rusherS', #'S',
            'rusherA', #'A', 
            'rusherDis', #'Dis', 
            'rusherOrientation', #'Orientation',
            'rusherDir', #'Dir', 
            'YardLine', 'Quarter', 'Down',
            'Distance', 'DefendersInTheBox', 
            'DefenceDL', #'num_DL', 
            'DefenceLB', #'num_LB', 
            'DefenceDB', #'num_DB',
            'OffenceQB', #'num_QB', 
            'OffenceRB', #'num_RB', 
            'OffenceWR', #'num_WR', 
            'OffenceTE', #'num_TE', 
            'OffenceOL', #'num_OL', 
            'OL_diff',
            'OL_TE_diff', 'run_def', 'Yards',
            'fe1', 'fe5', 'fe7', 'fe8', 'fe10', 'fe11'
        ]
        self.verbose = verbose

    def get_numeric_column(self, X, res_no_num=""):
        res = []
        numerics = [ 'int8',  'int16',  'int32',  'int64', 'float16', 'float32', 'float64',
                    'uint8', 'uint16', 'uint32', 'uint64']
        for col in X.columns:
            col_type = X[col].dtypes
            if col_type in numerics:
                res.append(col)
        return res

    def prepare_misc(self, X, yards=None):
        drop_feat = ['GameId', 'PlayId', 'Yards', 'game_seconds_left']
        y = None
        if yards is not None:
            y = np.zeros((yards.shape[0], 199))
            for idx, target in enumerate(list(yards)):
                y[idx][99 + target] = 1
        df = X.copy()
        df = NFL_trn_DataDropper(drop_columns=drop_feat).fit_transform(df)
        df['back_oriented_down_field'] = df['back_oriented_down_field'].astype('int')
        df['back_moving_down_field'] = df['back_moving_down_field'].astype('int')
        cat = ['back_oriented_down_field', 'back_moving_down_field']
        num = self.get_numeric_column(df)
        return df, y, cat, num
    
    def model_396_1(self, X, cat, num):
        inputs = []
        embeddings = []
        for i in cat:
            input_ = Input(shape=(1,))
            embedding = Embedding(int(np.absolute(X[i]).max() + 1), 10, input_length=1)(input_)
            embedding = Reshape(target_shape=(10,))(embedding)
            inputs.append(input_)
            embeddings.append(embedding)
        input_numeric = Input(shape=(len(num),))
        embedding_numeric = Dense(512, activation='relu')(input_numeric) 
        inputs.append(input_numeric)
        embeddings.append(embedding_numeric)
        x = Concatenate()(embeddings)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(199, activation='softmax')(x)
        model = Model(inputs, output)
        return model
    
    def fit(self, X, y):
        X_train = X.copy()
        self.pipeline_pre_predict_ = make_pipeline(
            NFL_trn_DataSelector_OneRowEachPlayId(),
            NFL_trn_CatOneHot2(X['Team']),
            NFL_trn_CatOneHot2(X['PossessionTeam']),
            NFL_trn_CatOneHot2(X['FieldPosition']),
            NFL_trn_CatOneHot2(X['OffenseFormation']),
            NFL_trn_CatOneHot2(X['PlayDirection']),
            NFL_trn_CatOneHot2(X['HomeTeamAbbr']),
            NFL_trn_CatOneHot2(X['VisitorTeamAbbr']),
            NFL_trn_CatOneHot2(X['Stadium']),
            NFL_trn_CatOneHot2(X['Location']),
            NFL_trn_CatOneHot2(X['StadiumType']),
            NFL_trn_CatOneHot2(X['Turf']),
            NFL_trn_CatOneHot2(X['GameWeather']),
            NFL_trn_CatOneHot2(X['Position']),
            NFL_trn_CatOneHot2(X['rusherPosition']),
            NFL_trn_DataSelector(select_columns=self.use_feat),
        )
        X_train = self.pipeline_pre_predict_.fit_transform(X)
        if self.verbose:
            X_train.info(max_cols=X_train.shape[1])
            display(datetime.datetime.now().isoformat())
        
        self.scaler_ = preprocessing.StandardScaler()
        X, y, cat, num = self.prepare_misc(X_train, X_train['Yards'])
        X[num] = self.scaler_.fit_transform(X[num])

        n_splits = 5
        kf = GroupKFold(n_splits=n_splits)
        score = []
        models = []

        for i_369, (tdx, vdx) in enumerate(kf.split(X, y, X_train['GameId'])):
            if self.verbose:
                print(f'Fold : {i_369}')
    
            X_train, X_val, y_train, y_val = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]
            X_train = [np.absolute(X_train[i]) for i in cat] + [X_train[num]]
            X_val = [np.absolute(X_val[i]) for i in cat] + [X_val[num]]
            model = self.model_396_1(X, cat, num)
            model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[])
            es = EarlyStopping(monitor='val_CRPS', 
                        mode='min',
                        restore_best_weights=True, 
                        verbose=2, 
                        patience=5)
            es.set_model(model)
            metric = NFL_NN_Metric(model, [es], [(X_train,y_train), (X_val,y_val)])
            for i in range(1):
                model.fit(X_train, y_train, verbose=self.verbose)
            for i in range(1):
                model.fit(X_train, y_train, batch_size=64, verbose=self.verbose)
            for i in range(1):
                model.fit(X_train, y_train, batch_size=128, verbose=self.verbose)
            for i in range(1):
                model.fit(X_train, y_train, batch_size=256, verbose=self.verbose)
            model.fit(X_train, y_train, callbacks=[metric], epochs=100, batch_size=1024, verbose=self.verbose)
            score_ = nfl_calc_crps(y_val, model.predict(X_val))
            #model.save(f'keras_369_{i_369}.h5')
            models.append(model)
            score.append(score_)
            if self.verbose:
                print(score_)
        
        self.models_ = models
        self.score_ = np.mean(score)
        if self.verbose:
            print(np.mean(self.score_))
            display(datetime.datetime.now().isoformat())
        
        return self

    def predict(self, X):
        X_test = X.copy()
        X_test = self.pipeline_pre_predict_.fit_transform(X_test)
        X, _, cat, num = self.prepare_misc(X_test)
        X[num] = self.scaler_.transform(X[num])
        X = [np.absolute(X[i]) for i in cat] + [X[num]]
        y_pred = np.mean([model.predict(X) for model in self.models_], axis=0)
        return y_pred


# In[3]:


display(datetime.datetime.now().isoformat())


# In[4]:


pipeline_fe = make_pipeline(
    NFL_trn_FillNa2(),
    NFL_trn_DataTypesCleaner(),
    NFL_trn_GameWeather(),
    NFL_trn_PlayerAge(),
    NFL_trn_PlayerHeight(),
    NFL_trn_StadiumType(),
    NFL_trn_TeamAbbr(),
    NFL_trn_Turf(),
    NFL_trn_WindDirection(),
    NFL_trn_WindSpeed(),
    NFL_trn_NormalizedMovingX(),
    NFL_trn_PlayerRelativeToBack(),
    NFL_trn_ChangeSameDirection(),
    NFL_trn_Orientation_std(),
    NFL_trn_RunnerDistance2los(),
    NFL_trn_RunnerVs1stDefensorSpeed(),
    NFL_trn_DefenseXspead(),
    NFL_trn_PersonnelSplitter(),
    NFL_trn_DistRB2Defenders(),
    NFL_trn_Orientation(),
    NFL_trn_PlayerRusher(),
    NFL_trn_TimeToTackle(),
    NFL_trn_DistanceToCentroid(),
    NFL_trn_SecondsSinceStart(),
    NFL_trn_DistanceToQB(),
    NFL_trn_Xstd(),
    NFL_trn_TeamStat(),
    NFL_trn_BinsYard(),
    NFL_trn_OffenseLead(),
    NFL_trn_DistDefenders(),
    NFL_trn_SecondsSinceGameStart(),
    NFL_trn_ProcessTwo(),
)
if not os.path.isfile('./res/nfl08_train.csv.gz'):
    if os.path.isfile('/kaggle/input/masasuke-nfl/nfl08_train.pickle'):
        X_train = pd.read_pickle('/kaggle/input/masasuke-nfl/nfl08_train.pickle', compression='gzip')
    else:
        X_train = nfl_read_train()
        X_train = pipeline_fe.fit_transform(X=X_train)
        if not os.path.isdir('/kaggle/working'):
            X_train.to_pickle('./res/nfl08_train.csv.gz')
else:
    X_train = pd.read_pickle('./res/nfl08_train.csv.gz')

X_train.info(max_cols=X_train.shape[1])
display(datetime.datetime.now().isoformat())


# In[5]:


clf_lgb = Nfl_clf_LightGBM()
clf_nn  = Nfl_clf_NN()
eclf = Nfl_clf_EnsembleVoting(estimators=[('nn', clf_nn), ('lgb', clf_lgb)])


# In[6]:


eclf.fit(X_train, y=X_train['Yards'])
display(datetime.datetime.now().isoformat())


# In[7]:


from kaggle.competitions import nflrush
env = nflrush.make_env()

for (test_df, sample_prediction_df) in env.iter_test():
    X_test = pipeline_fe.fit_transform(X=test_df)
    y_pred = eclf.predict(X_test)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]
    preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
    env.predict(preds_df)

env.write_submission_file()
display(datetime.datetime.now().isoformat())

