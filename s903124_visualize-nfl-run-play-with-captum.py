#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.preprocessing import StandardScaler,MinMaxScaler

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler(feature_range=(20,400))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


get_ipython().system('pip install captum')


# In[3]:


import numpy as np
from numba import jit


def create_faetures(df):
    xysdir_o = df[(df.IsOnOffense == True) & (df.IsRusher == False)][['X','Y','X_S','Y_S']].values
    xysdir_rush = df[df.IsRusher == True][['X','Y','X_S','Y_S']].values
    xysdir_d = df[df.IsOnOffense == False][['X','Y','X_S','Y_S']].values
    
    off_x = np.array(df[(df.IsOnOffense == True) & (df.IsRusher == False)].groupby('PlayId')['X'].apply(np.array))
    def_x = np.array(df[(df.IsOnOffense == False) ].groupby('PlayId')['X'].apply(np.array))
    off_y = np.array(df[(df.IsOnOffense == True) & (df.IsRusher == False)].groupby('PlayId')['Y'].apply(np.array))
    def_y = np.array(df[(df.IsOnOffense == False) ].groupby('PlayId')['Y'].apply(np.array))
    off_sx = np.array(df[(df.IsOnOffense == True) & (df.IsRusher == False)].groupby('PlayId')['X_S'].apply(np.array))
    def_sx = np.array(df[(df.IsOnOffense == False) ].groupby('PlayId')['X_S'].apply(np.array))
    off_sy = np.array(df[(df.IsOnOffense == True) & (df.IsRusher == False)].groupby('PlayId')['Y_S'].apply(np.array))
    def_sy = np.array(df[(df.IsOnOffense == False) ].groupby('PlayId')['Y_S'].apply(np.array))
    
    player_vector = []
    for play in range(len(off_x)):
        player_feat = player_feature(off_x[play],def_x[play],off_y[play],def_y[play],off_sx[play],def_sx[play],
                                     off_sy[play],def_sy[play],xysdir_rush[play])
        player_vector.append(player_feat)
    
    return np.array(player_vector)

    
def player_feature(off_x,def_x,off_y,def_y,off_sx,def_sx,off_sy,def_sy,xysdir_rush):
    if(len(off_x<10)):
        off_x = np.pad(off_x,(10-len(off_x),0), 'mean' )
        off_y = np.pad(off_y,(10-len(off_y),0), 'mean' )
        off_sx = np.pad(off_sx,(10-len(off_sx),0), 'mean' )
        off_sy = np.pad(off_sy,(10-len(off_sy),0), 'mean' )
    if(len(def_x<11)):
        def_x = np.pad(def_x,(11-len(def_x),0), 'mean' )
        def_y = np.pad(def_y,(11-len(def_y),0), 'mean' )
        def_sx = np.pad(def_sx,(11-len(def_sx),0), 'mean' )
        def_sy = np.pad(def_sy,(11-len(def_sy),0), 'mean' )

    dist_def_off_x = def_x.reshape(-1,1)-off_x.reshape(1,-1)
    dist_def_off_sx = def_sx.reshape(-1,1)-off_sx.reshape(1,-1)
    dist_def_off_y = def_y.reshape(-1,1)-off_y.reshape(1,-1)
    dist_def_off_sy = def_sy.reshape(-1,1)-off_sy.reshape(1,-1)
    dist_def_rush_x = def_x.reshape(-1,1)-np.repeat(xysdir_rush[0],10).reshape(1,-1)
    dist_def_rush_y = def_y.reshape(-1,1)-np.repeat(xysdir_rush[1],10).reshape(1,-1)
    dist_def_rush_sx = def_sx.reshape(-1,1)-np.repeat(xysdir_rush[2],10).reshape(1,-1)
    dist_def_rush_sy = def_sy.reshape(-1,1)-np.repeat(xysdir_rush[3],10).reshape(1,-1)
    def_sx = np.repeat(def_sx,10).reshape(11,-1)
    def_sy = np.repeat(def_sy,10).reshape(11,-1)
    feats = [dist_def_off_x, dist_def_off_sx, dist_def_off_y, dist_def_off_sy, dist_def_rush_x, dist_def_rush_y,
            dist_def_rush_sx, dist_def_rush_sy, def_sx, def_sy]
    
    return np.stack(feats)


def get_def_speed(df):
    df_cp = df[~df.IsOnOffense].copy()
    speed = df_cp["S"].T.values
    speed = speed.reshape(-1, 1, 1, 11) 
    speed = np.repeat(speed, 10, axis=2)

    return speed


def get_dist(df, col1, col2, type="defence"):
    if type == "defence":
        df_cp = df[~df.IsOnOffense].copy()
    elif type == "offence":
        df_cp = df[df.IsOnOffense].copy()
    dist = np.linalg.norm(df_cp[col1].values - df_cp[col2].values, axis=1)
    dist = dist.T
    dist = dist.reshape(-1, 1, 1, 11)
    dist = np.repeat(dist, 10, axis=2)

    return dist



def dist_def_off(df, n_train, cols):
    off_x = np.array(df[(df.IsOnOffense) & (~train.IsRusher)].groupby('PlayId')['X'].apply(np.array))
    def_x = np.array(df[(~df.IsOnOffense) ].groupby('PlayId')['X'].apply(np.array))
    off_y = np.array(df[(df.IsOnOffense) & (~train.IsRusher)].groupby('PlayId')['Y'].apply(np.array))
    def_y = np.array(df[(~df.IsOnOffense) ].groupby('PlayId')['Y_S'].apply(np.array))
    off_xs = np.array(df[(df.IsOnOffense) & (~train.IsRusher)].groupby('PlayId')['X_S'].apply(np.array))
    def_xs = np.array(df[(~df.IsOnOffense) ].groupby('PlayId')['X_S'].apply(np.array))
    off_ys = np.array(df[(df.IsOnOffense) & (~train.IsRusher)].groupby('PlayId')['Y_S'].apply(np.array))
    def_ys = np.array(df[(~df.IsOnOffense) ].groupby('PlayId')['Y_S'].apply(np.array))
    feats = []
    for play in range(len(off_x)):
        dist_x = off_x[play].reshape(-1, 1) - def_x[play].reshape(1, -1)
        dist_y = off_y[play].reshape(-1, 1) - def_y[play].reshape(1, -1)
        dist = np.concatenate([dist_x[:, :, np.newaxis], dist_y[:, :, np.newaxis]], axis=2)
        dist_xy = np.linalg.norm(dist.astype(np.float64), axis=2)
        dist_xs = off_xs[play].reshape(-1, 1) - def_xs[play].reshape(1, -1)
        dist_ys = off_ys[play].reshape(-1, 1) - def_ys[play].reshape(1, -1)
        dist = np.concatenate([dist_xs[:, :, np.newaxis], dist_ys[:, :, np.newaxis]], axis=2)
        dist_xys = np.linalg.norm(dist.astype(np.float64), axis=2)
        feats.append(np.concatenate([dist_xy[np.newaxis, :], dist_xys[np.newaxis, :]], axis=0))
    return np.array(feats)


# In[4]:


import numpy as np


def reorient(df, flip_left, aug=False):
    df['ToLeft'] = df.PlayDirection == "left"
    
    df.loc[df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    df.loc[df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    df.loc[df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    df.loc[df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    df.loc[df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    df.loc[df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    df.loc[df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    df.loc[df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df['IsOnOffense'] = df.Team == df.TeamOnOffense  # Is player on offense?
    df['YardLine_std'] = 100 - df.YardLine
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam, 'YardLine_std'] =         df.loc[df.FieldPosition.fillna('') == df.PossessionTeam, 'YardLine']
    df.loc[df.ToLeft, 'X'] = 120 - df.loc[df.ToLeft, 'X']
    df.loc[df.ToLeft, 'Y'] = 160 / 3 - df.loc[df.ToLeft, 'Y']
    df.loc[df.ToLeft, 'Orientation'] = np.mod(180 + df.loc[df.ToLeft, 'Orientation'], 360)
    df['Dir'] = 90 - df.Dir
    df.loc[df.ToLeft, 'Dir'] = np.mod(180 + df.loc[df.ToLeft, 'Dir'], 360)
    df.loc[df.IsOnOffense, 'Dir'] = df.loc[df.IsOnOffense, 'Dir'].fillna(0).values
    df.loc[~df.IsOnOffense, 'Dir'] = df.loc[~df.IsOnOffense, 'Dir'].fillna(180).values

    df['IsRusher'] = df['NflId'] == df['NflIdRusher']
    if flip_left:
        tmp = df[df['IsRusher']].copy()
        # df['left'] = df.Y < 160/6
        tmp['left'] = tmp.Dir < 0
        df = df.merge(tmp[['PlayId', 'left']], how='left', on='PlayId')
        df['Y'] = df.Y
        df.loc[df["left"], 'Y'] = 160 / 3 - df.loc[df["left"], 'Y']
        df['Dir'] = df.Dir
        df.loc[df["left"], 'Dir'] = np.mod(- df.loc[df["left"], 'Dir'], 360)
        df.drop('left', axis=1, inplace=True)

    df["S"] = df["Dis"] * 10
    df['X_dir'] = np.cos((np.pi / 180) * df.Dir)
    df['Y_dir'] = np.sin((np.pi / 180) * df.Dir)
    df['X_S'] = df.X_dir * df.S
    df['Y_S'] = df.Y_dir * df.S
    df['X_A'] = df.X_dir * df.A
    df['Y_A'] = df.Y_dir * df.A
    #df.loc[df['Season'] == 2017, 'S'] = (df['S'][df['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570
    df['time_step'] = 0.0
    df = df.sort_values(by=['PlayId', 'IsOnOffense', 'IsRusher', 'Y']).reset_index(drop=True)
    
    if aug:
        df_aug = df.copy()
        df_aug["Y"] = 53.3 - df_aug["Y"]
        df = df.append(df_aug).reset_index()
    
    return df


def merge_rusherfeats(df):
    rusher_feats = df[df['NflId'] == df['NflIdRusher']].drop_duplicates()
    rusher_feats = rusher_feats[["PlayId", "X", "Y", "X_S", "Y_S"]]
    rusher_feats = rusher_feats.rename(
        columns={"X": "Rusher_X", "Y": "Rusher_Y", "X_S": "Rusher_X_S", "Y_S": "Rusher_Y_S"})
    df = df.merge(rusher_feats, how="left", on="PlayId")

    return df

def scaling(feats, sctype="standard"):
    v1 = []
    v2 = []
    for i in range(feats.shape[1]):
        feats_ = feats[:, i, :]
        if sctype == "standard":
            mean_ = np.mean(feats_)
            std_ = np.std(feats_)
            feats[:, i, :] -= mean_
            feats[:, i, :] /= std_
            v1.append(mean_)
            v2.append(std_)
        elif sctype == "minmax":
            max_ = np.max(feats_)
            min_ = np.min(feats_)
            feats[:, i, :] = (feats_ - min_) / (max_ - min_)
            v1.append(max_)
            v2.append(min_)

    return feats, v1, v2


# In[5]:


import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(10, 128, kernel_size=1, stride=1, bias=False),
            nn.CELU(),
            nn.Conv2d(128, 160, kernel_size=1, stride=1, bias=False),
            nn.CELU(),
            nn.Conv2d(160, 128, kernel_size=1, stride=1, bias=False),
            nn.CELU()
        )
        self.pool1 = nn.AdaptiveAvgPool2d((1, 11))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 160, kernel_size=(1, 1), stride=1, bias=False),
            nn.CELU(),
            nn.BatchNorm2d(160),
            nn.Conv2d(160, 96, kernel_size=(1, 1), stride=1, bias=False),
            nn.CELU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=(1, 1), stride=1, bias=False),
            nn.CELU(),
            nn.BatchNorm2d(96),
        )
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.last_linear = nn.Sequential(
            Flatten(),
            nn.Linear(96, 256),
            nn.LayerNorm(256),
            nn.CELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.last_linear(x)

        return x


# In[6]:


model = CnnModel(num_classes=199)

model.load_state_dict(torch.load('/kaggle/input/1st-place-reproduction-10feats-dev/exp1_reproduce_fold0.pth'))


# In[7]:


DATA_DIR = "../input/nfl-big-data-bowl-2020"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")

train = pd.read_csv(TRAIN_PATH, dtype={'WindSpeed': 'object'})

train = reorient(train, flip_left=True)
train = merge_rusherfeats(train)


x = create_faetures(train)


# In[8]:



from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    FeatureAblation
)


# In[9]:


handoff_frame = train[train.PlayId == train.loc[6000].PlayId] #select a random play
sample_x = create_faetures(handoff_frame)


# In[10]:


dl = DeepLiftShap(model)
attributions_0, delta_0 = dl.attribute(torch.Tensor(sample_x), torch.Tensor(x[:2000]), target=100, return_convergence_delta=True)

# print('DeepLiftSHAP Attributions:', attributions)
# print('Convergence Delta:', delta)


# In[11]:


# ig = IntegratedGradients(model)
# baseline = np.zeros_like(sample_x)
# attributions, delta = ig.attribute(torch.tensor(sample_x, dtype=torch.float32), torch.tensor(baseline, dtype=torch.float32), target=0, return_convergence_delta=True,n_steps=1000)


# In[12]:


# gs = GradientShap(model)
# baseline = np.zeros_like(sample_x)
# # We define a distribution of baselines and draw `n_samples` from that
# # distribution in order to estimate the expectations of gradients across all baselines

# attributions, delta = gs.attribute(torch.Tensor(sample_x), stdevs=0.09, n_samples=8, baselines=torch.Tensor(baseline),
#                                    target=0, return_convergence_delta=True)


# In[13]:


x_scale = standard_scaler.fit_transform(np.sum(attributions_0[0][0,:,:].detach().numpy(),axis=1).reshape(-1,1) ).flatten()
y_scale = standard_scaler.fit_transform( np.sum(attributions_0[0][2,:,:].detach().numpy(),axis=1).reshape(-1,1)).flatten()
vx_scale = standard_scaler.fit_transform( np.sum(attributions_0[0][1,:,:].detach().numpy(),axis=1).reshape(-1,1) ).flatten()
vy_scale = standard_scaler.fit_transform( np.sum(attributions_0[0][3,:,:].detach().numpy(),axis=1).reshape(-1,1) ).flatten()

size_out = minmax_scaler.fit_transform(np.clip(1.5**(x_scale+y_scale+vx_scale+vy_scale),-3.5,3.5).reshape(-1,1)).flatten()


# In[14]:


plt.scatter(handoff_frame[handoff_frame.IsOnOffense == False]['Y'],handoff_frame[handoff_frame.IsOnOffense == False]['X'],
           s=size_out)

plt.scatter(handoff_frame[(handoff_frame.IsOnOffense == True) & (handoff_frame.IsRusher == False)]['Y'],handoff_frame[(handoff_frame.IsOnOffense == True) & (handoff_frame.IsRusher == False)]['X'],s=100)
rusher_df = handoff_frame[(handoff_frame.IsOnOffense == True) & (handoff_frame.IsRusher == True)]
plt.scatter(rusher_df['Y'],rusher_df['X'],s=100)
plt.arrow(rusher_df['Y'].values[0],rusher_df['X'].values[0],rusher_df['Y_S'].values[0],rusher_df['X_S'].values[0],head_width=1)
plt.axhline(y=rusher_df.YardLine_std.values[0]+10, color='r', linestyle='--')
plt.xlim([0,53.3])

