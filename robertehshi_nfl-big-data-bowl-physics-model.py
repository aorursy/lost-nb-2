#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import numpy.matlib
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import gc
import sys

# Any results you write to the current directory are saved as output.

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Add, concatenate
from keras.optimizers import SGD
from  keras.losses import categorical_crossentropy
from keras.regularizers import l2
keras.backend.set_floatx('float32')

from xgboost import XGBClassifier

from kaggle.competitions import nflrush


# In[2]:


def error_correcting_codes(df):
    df = df.replace('BLT', 'BAL')
    df = df.replace('HST', 'HOU')
    df = df.replace('ARZ', 'ARI')
    df = df.replace('CLV', 'CLE')
    return df
  

def organize_positions(df):
    return (df.loc[(df['PossessionTeam']==df['HomeTeamAbbr'])&(df['Team']=='away') | (df['PossessionTeam']==df['VisitorTeamAbbr'])&(df['Team']=='home')].copy().reset_index(),
      df.loc[((df['PossessionTeam']==df['HomeTeamAbbr'])&(df['Team']=='home') | (df['PossessionTeam']==df['VisitorTeamAbbr'])&(df['Team']=='away'))&(df['NflId']!=df['NflIdRusher'])].copy().reset_index(),
      df.loc[df['NflId']==df['NflIdRusher']].copy().reset_index())
    

def doubledown(X, doublings=1):
    np.random.seed(3)
    for w in range(doublings):
        X_dupe2 = np.concatenate((X.copy(), X.copy()), axis=0)
        for i in range(X.shape[0]):
            X_dupe2[2*i, :] = X[i, :]
            X_dupe2[2*i+1, :] = X[i, :]
        X = X_dupe2

    return X


def physics_init(df):
    way = -2*(df['PlayDirection']=='left') + 1
    theta = way*df['Dir']*np.pi/180
    df['X'] = (df['PlayDirection']=='right')*df['X'] + (df['PlayDirection']=='left')*(120 - df['X'])
    df['Sx'] = np.sin(theta)*df['S']
    df['Sy'] = np.cos(theta)*df['S']
    df['Ax'] = np.sin(theta)*df['A']
    df['Ay'] = np.cos(theta)*df['A']
    df['EquivYardLine'] = (df['PossessionTeam']==df['FieldPosition'])*(df['YardLine']+10) + (df['PossessionTeam']!=df['FieldPosition'])*(110-df['YardLine'])

    defn, off, RBs = organize_positions(df)

    defn['X'] -= RBs.loc[[i//11 for i in defn.index], 'X'].values
    defn['Y'] -= RBs.loc[[i//11 for i in defn.index], 'Y'].values
    defn = defn.loc[:, ('PlayId', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay', 'X', 'Y')]
    defn.fillna(0, inplace=True)
    defn['Infl'] = defn['PlayerWeight']/(np.square(defn['X']) + np.square(defn['Y']))**0.5
    defn['AngularMomentum'] = -defn['PlayerWeight']*(defn['X']*defn['Sx'] + defn['Y']*defn['Sy'])/(np.square(defn['X']) + np.square(defn['Y']))

    off['X'] -= RBs.loc[[i//10 for i in off.index], 'X'].values
    off['Y'] -= RBs.loc[[i//10 for i in off.index], 'Y'].values
    off = off.loc[:, ('PlayId', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay', 'X', 'Y')]
    off.fillna(0, inplace=True)
    off['Infl'] = off['PlayerWeight']/(np.square(off['X']) + np.square(off['Y']))**0.5
    off['AngularMomentum'] = -off['PlayerWeight']*(off['X']*off['Sx'] + off['Y']*off['Sy'])/(np.square(off['X']) + np.square(off['Y']))

    RBs['YardsBehindScrimmage'] = RBs['EquivYardLine'] - RBs['X']
    RBs['X'] = 0
    RBs['Y'] = 0
    RBs = RBs.loc[:, ('PlayId', 'YardsBehindScrimmage', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay', 'X', 'Y')]
    RBs.fillna(0, inplace=True)
    
    return defn, off, RBs


def action(defn, off, RBs, timestep=0.1):
    t = 0.0
    while t<timestep:
        for X in (defn, off, RBs):
            X['X'] += X['Sx']*0.01 +X['Ax']*0.01**2/2
            X['Y'] += X['Sy']*0.01 +X['Ay']*0.01**2/2
            X['Sx'] += X['Ax']*0.01
            X['Sy'] += X['Ay']*0.01
            X['Ax'] *= 0.99
            X['Ay'] *= 0.99
        t += 0.01

        defn['X'] -= RBs.loc[[i//11 for i in defn.index], 'X'].values
        defn['Y'] -= RBs.loc[[i//11 for i in defn.index], 'Y'].values
        defn['Infl'] = defn['PlayerWeight']/(np.square(defn['X']) + np.square(defn['Y']))**0.5
        defn['AngularMomentum'] = -defn['PlayerWeight']*(defn['X']*defn['Sx'] + defn['Y']*defn['Sy'])/(np.square(defn['X']) + np.square(defn['Y']))

        off['X'] -= RBs.loc[[i//10 for i in off.index], 'X'].values
        off['Y'] -= RBs.loc[[i//10 for i in off.index], 'Y'].values
        off['Infl'] = off['PlayerWeight']/(np.square(off['X']) + np.square(off['Y']))**0.5
        off['AngularMomentum'] = -off['PlayerWeight']*(off['X']*off['Sx'] + off['Y']*off['Sy'])/(np.square(off['X']) + np.square(off['Y']))

        RBs['X'] = 0
        RBs['Y'] = 0

    return defn, off, RBs


def physics_doubledown(X, doublings, width):
    np.random.seed(3)
    for w in range(doublings):
        X_dupe = X.copy()
        X_dupe2 = np.concatenate((X.copy(), X.copy()), axis=0)
        numpy_sucks = np.arange(11)
        np.random.shuffle(numpy_sucks)
        for (i,j) in enumerate(numpy_sucks):
            X_dupe[:, width*i:width*i+width] = X[:, width*j:width*j+width]
        numpy_sucks = np.arange(10)
        np.random.shuffle(numpy_sucks)
        for (i,j) in enumerate(numpy_sucks):
            X_dupe[:, width*(i+11):width*(i+12)] = X[:, width*(j+11):width*(j+12)]
        for i in range(X.shape[0]):
            X_dupe2[2*i, :] = X[i, :]
            X_dupe2[2*i+1, :] = X_dupe[i, :]
        X = X_dupe2

    return X


def generate_physics(df, forward_action=0, timestep=0.1, doublings=0):
    d, o, r = physics_init(df)
    df = None

    defn = [d.copy()]
    off = [o.copy()]
    RBs = [r.copy()]

    for a in range(forward_action):
        d, o, r = action(d, o, r, timestep)
        defn.append(d.copy())
        off.append(o.copy())
        RBs.append(r.copy())
    d, o, r = None, None, None

    for X in (defn, off, RBs):
        for i in range(len(defn)):
            X[i]['Px'] = X[i]['Sx']*X[i]['PlayerWeight']
            X[i]['Py'] = X[i]['Sy']*X[i]['PlayerWeight']
            X[i]['Fx'] = X[i]['Ax']*X[i]['PlayerWeight']
            X[i]['Fy'] = X[i]['Ay']*X[i]['PlayerWeight']

    for i in range(len(defn)):
        if i==0:
            bigD = defn[i].loc[:, ('Px', 'Py', 'X', 'Y', 'Infl', 'AngularMomentum')].astype(np.float32)
            bigO = off[i].loc[:, ('Px', 'Py', 'X', 'Y', 'Infl', 'AngularMomentum')].astype(np.float32)
            backs = RBs[i].loc[:, ('YardsBehindScrimmage', 'Px', 'Py', 'Fx', 'Fy')].astype(np.float32)
            if bigD.shape[0]==0 | bigO.shape[0]==0:
                bigD = pd.DataFrame(data=np.random.randn(11,6), columns=('Px', 'Py', 'X', 'Y', 'Infl', 'AngularMomentum'))
                bigO = pd.DataFrame(data=np.random.randn(10,6), columns=('Px', 'Py', 'X', 'Y', 'Infl', 'AngularMomentum'))
            inst = np.concatenate((np.reshape(bigD.copy().values, (bigD.shape[0]//11, 11*6)), 
                                            np.reshape(bigO.copy().values, (bigO.shape[0]//10, 10*6)), 
                                            backs.copy().values), axis=1)
            summary = physics_doubledown(inst, doublings, 6)
        else:
            bigD = defn[i].loc[:, ('Infl', 'AngularMomentum')].astype(np.float32)
            bigO = off[i].loc[:, ('Infl', 'AngularMomentum')].astype(np.float32)
            if bigD.shape[0]==0 | bigO.shape[0]==0:
                bigD = pd.DataFrame(data=np.random.randn(11,2), columns=('Infl', 'AngularMomentum'))
                bigO = pd.DataFrame(data=np.random.randn(10,2), columns=('Infl', 'AngularMomentum'))
            inst = np.concatenate((np.reshape(bigD.copy().values, (bigD.shape[0]//11, 11*2)), 
                                            np.reshape(bigO.copy().values, (bigO.shape[0]//10, 10*2))), axis=1)
            summary = np.concatenate((summary, physics_doubledown(inst, doublings, 2)), axis=1)
    defn, off, RBs = None, None, None

    return summary


def generate_physics_II(df):
    defn, off, RBs = physics_init(df)

    bigD = defn[['PlayerWeight', 'X', 'Y', 'Sx', 'Sy', 'Ax', 'Ay']].astype(np.float32)
    bigO = off[['PlayerWeight', 'X', 'Y', 'Sx', 'Sy', 'Ax', 'Ay']].astype(np.float32)
    backs = RBs[['YardsBehindScrimmage', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay']].astype(np.float32)
    if bigD.shape[0]==0 | bigO.shape[0]==0:
        bigD = pd.DataFrame(data=np.random.randn(11,7), columns=('PlayerWeight', 'X', 'Y', 'Sx', 'Sy', 'Ax', 'Ay'))
        bigO = pd.DataFrame(data=np.random.randn(10,7), columns=('PlayerWeight', 'X', 'Y', 'Sx', 'Sy', 'Ax', 'Ay'))
    summary = np.concatenate((backs.copy().values,
                             np.reshape(bigD.copy().values, (bigD.shape[0]//11, 11*7)), 
                              np.reshape(bigO.copy().values, (bigO.shape[0]//10, 10*7))), axis=1)
    defn, off, RBs = None, None, None
    bigD, bigO, backs = None, None, None

    return summary


def down_situation(df):
    X = df.loc[::22, ('Down', 'Distance')].copy().astype(np.float32)
    X.fillna(-1, inplace=True)
    framer = pd.DataFrame(columns=(1.0, 2.0, 3.0, 4.0, 'Distance'), dtype=np.float32)
    concatenation = pd.concat((pd.get_dummies(X['Down']), X['Distance']), axis=1, join='outer')
    concatenation = pd.concat((framer, concatenation), axis=0, join='outer')
    concatenation.fillna(0, inplace=True)
    return concatenation.values


def RB_wins_it(df):
    X = df.loc[::22, ('NflIdRusher')]
    framer = pd.DataFrame(columns=baxxx, dtype=np.float32)
    OHE = pd.get_dummies(X)
    concatenation = pd.concat((framer, OHE.loc[:, :framer.shape[1]]), axis=0, join='outer')
    concatenation.fillna(0, inplace=True)
    return concatenation.values.astype(np.float32)


def stats(array):
    return array.mean(axis=0), array.std(axis=0)


def stats_II(X):
    means = X[:, :11].mean(axis=0)
    deviants = X[:, :11].std(axis=0)
    for start, step, amt in [(11,7,11), (88,7,10)]: #D and O
        repmeans = []
        repdevs = []
        for i in range(step):
            repmeans.append(X[:, start+i:start+step*amt:step].mean())
            repdevs.append(X[:, start+i:start+step*amt:step].std())
        for j in range(amt):
            means = np.concatenate((means, repmeans))
            deviants = np.concatenate((deviants, repdevs))
    means = np.concatenate((means, [0 for i in range(158, 529)]), axis=0)
    deviants = np.concatenate((deviants, [1 for i in range(158, 529)]), axis=0)
    return means, deviants


def normalize(X, mn, stand):
    return (X - mn) / stand


def generate_yardudge(df):
    Y = np.zeros((df.shape[0]//22, 199))
    for (i, yerds) in enumerate(df['Yards'][::22]):
        Y[i, yerds+99] = 1
    return Y.astype(np.float32)


# In[3]:


# global meen_I, sigma_I, PCs, doublings, forward_actions, timestep

# train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

# doublings = 5
# forward_actions = 8
# timestep = 0.1

# Y = doubledown(generate_yardudge(train_df), doublings)

# train_df = error_correcting_codes(train_df)
# X = np.concatenate(
#                 (doubledown(down_situation(train_df), doublings), 
#                  generate_physics(train_df, forward_actions, timestep, doublings)), 
#             axis=1)
# train_df = None

# meen_I, sigma_I = stats(X)
# X = normalize(X, meen_I, sigma_I)

# PCs = np.linalg.eig(np.dot(np.transpose(X), X))[1]
# X = np.dot(X, PCs)

# gc.collect()


# In[4]:


# stuff = [0.01541, 0.00000173, 0.8858, 0.3867, 0.6044, 14, 180, 1024, 512, 256]

# learning_rate = stuff[0]
# decay = stuff[1]
# beta = stuff[2]
# clipse = stuff[3]
# dropout_rate = stuff[4]
# epochs = stuff[5]
# batch_size = stuff[6]
# hidden_layer1_size = int(round(stuff[7]))
# hidden_layer2_size = int(round(stuff[8]))
# hidden_layer3_size = int(round(stuff[9]))

# np.random.seed(1729)
# model_NN1 = Sequential()
# model_NN1.add(Dense(units=hidden_layer1_size, activation='relu'))
# model_NN1.add(Dropout(dropout_rate))
# model_NN1.add(Dense(units=hidden_layer2_size, activation='relu'))
# model_NN1.add(Dropout(dropout_rate))
# model_NN1.add(Dense(units=hidden_layer3_size, activation='relu'))
# model_NN1.add(Dropout(dropout_rate))
# model_NN1.add(Dense(units=199, activation='softmax'))
# model_NN1.compile(loss=categorical_crossentropy, optimizer=SGD(lr=learning_rate, decay=decay, momentum=beta, nesterov=True, clipnorm=clipse))
# model_NN1.fit(X, Y, epochs=epochs, batch_size=batch_size) #,  validation_split=0.1)


# In[5]:


# test_X = X[20853*2**doublings:]
# P_NN1_partsy = model_NN1.predict(test_X)
# P_NN1 = None
# for i in range(P_NN1_partsy.shape[0]//2**doublings):
#     if P_NN1 is None:
#         P_NN1 = np.reshape(np.mean(P_NN1_partsy[2**doublings*i:2**doublings*(i+1), :], axis=0), (1, 199))
#     else:
#         P_NN1 = np.concatenate((P_NN1, np.reshape(np.mean(P_NN1_partsy[2**doublings*i:2**doublings*(i+1), :], axis=0), (1, 199))), axis=0)


# In[6]:


# X, Y = None, None
# gc.collect()


# In[7]:


global meen_II, sigma,_II, baxxx

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

baxxx = train_df['NflIdRusher'].unique()

Y = generate_yardudge(train_df)

train_df = error_correcting_codes(train_df)

X = np.concatenate((down_situation(train_df), 
                    generate_physics_II(train_df),
                    RB_wins_it(train_df)), axis=1)
train_df = None

meen_II, sigma_II = stats_II(X)
X = normalize(X, meen_II, sigma_II)

gc.collect()


# In[8]:


stuff = [4.81725572e-02, 1.48089908e-05, 9.57455511e-01, 1.58078791e+00,
       9.93707682e-02, 5.04421303e-01, 6.51849665e+01, 9.68255755e+02,
       5.14218542e+02, 5.09462973e+02, 5.17812944e+02, 8.04146527e+02,
       1.64480385e+02, 9.80945456e+02, 8.50852251e+01, 1.03668046e+03,
       6.57080119e+01]

learning_rate = stuff[0]
lr_decay =stuff[1]
beta = stuff[2]
clipse = stuff[3]
drop_rate1 = stuff[4]
drop_rate2 = stuff[5]
epochs = int(round(stuff[6]))
batch_size = int(round(stuff[7]))
hidden_layer1_nodes = int(round(stuff[8]))
hidden_layer2_nodes = int(round(stuff[9]))
hidden_layer3_nodes = int(round(stuff[10]))
function_layer1_nodes = int(round(stuff[11]))
function_layer2_nodes = int(round(stuff[12]))
function_layer3_nodes = int(round(stuff[13]))
function_layer4_nodes = int(round(stuff[14]))
function_layer5_nodes = int(round(stuff[15]))
function_layer6_nodes = int(round(stuff[16]))

ipt = [Input(shape=(11,))] + [Input(shape=(7,)) for i in range(11)] + [Input(shape=(7,)) for i in range(10)] + [Input(shape=(baxxx.shape[0],))]

np.random.seed(314159)

expanse = Dense(function_layer1_nodes, activation='relu') 
collapsor = Dense(function_layer2_nodes, activation='linear')
expanse2 = Dense(function_layer3_nodes, activation='relu')
collapsor2 = Dense(function_layer4_nodes, activation='linear')
faux_embedding_expanse = Dense(function_layer5_nodes, activation='relu') 
faux_embedding_collapsor = Dense(function_layer6_nodes, activation='linear') 
D = Add()([collapsor(expanse(ipt[i])) for i in range(1,12)])
O = Add()([collapsor(expanse(ipt[i])) for i in range(12,22)])
R = faux_embedding_collapsor(faux_embedding_expanse(ipt[22]))
p = concatenate([ipt[0], D, O, R], axis=-1)
x = Dropout(drop_rate1)(p)
x = Dense(hidden_layer1_nodes, activation='relu')(x)
x = Dropout(drop_rate2)(x)
x = Dense(hidden_layer2_nodes, activation='relu')(x)
x = Dropout(drop_rate2)(x)
x = Dense(hidden_layer3_nodes, activation='relu')(x)
x = Dropout(drop_rate2)(x)
output = Dense(199, activation='softmax')(x)

model_NN2 = Model(inputs=ipt, outputs=output)

model_NN2.compile(optimizer=SGD(lr=learning_rate, decay=lr_decay, momentum=beta, nesterov=True, clipnorm=clipse), loss=categorical_crossentropy)
model_NN2.fit([X[:,:11]]+[X[:,7*i+11:7*i+18] for i in range(0,11)]+
                   [X[:,7*i+11:7*i+18] for i in range(11,21)]+[X[:,158:]],
                   Y, epochs=epochs, batch_size=batch_size) #, validation_split=0.1)

gc.collect()


# In[9]:


# P_NN2 = model_NN2.predict([X[20853:,:11]]+[X[20853:,7*i+11:7*i+18] for i in range(0,11)]+
#                        [X[20853:,7*i+11:7*i+18] for i in range(11,21)]+[X[20853:,158:]])


# In[10]:


# global meen_III, sigma_III, PCs_III, forward_actions, timestep

# forward_actions = 8
# timestep = 0.1

# train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

# Y = generate_yardudge(train_df)

# train_df = error_correcting_codes(train_df)
# X = np.concatenate(
#                 (down_situation(train_df), 
#                  generate_physics(train_df, forward_actions, timestep)), 
#             axis=1)
# train_df = None

# meen_III, sigma_III = stats(X)
# X = normalize(X, meen_III, sigma_III)

# PCs_III = np.linalg.eig(np.dot(np.transpose(X), X))[1]
# X = np.dot(X, PCs_III)

# gc.collect()


# In[11]:


Yxgb = np.argmax(Y, axis=1) - 99
model_xgb1 = XGBClassifier(objective='multi:softmax', seed=27182818, max_depth=1)
model_xgb1.fit(X, Yxgb)


# In[12]:


# offset = int((modelo.predict(X[20853:20854,:]) - np.argmax(modelo.predict_proba(X[20853:20854, :]), axis=1))[0])
# P_xgb1 = modelo.predict_proba(X[20853:, :])
# P_xgb1 = np.concatenate((np.zeros(Y.shape[0], 99+offset), P_xgb1, np.ones(Y.shape[0], 100-P_xgb1.shape[1]-offset)), axis=1)
# P_xgb1.shape


# In[13]:


# def CRPS(P, Y):
#     cucumber = np.zeros(P.shape[0])
#     C = np.zeros(P.shape)
#     for i in range(1, P.shape[1]):
#         cucumber = cucumber + np.maximum(P[:, i], 0)
#         C[:, i] = np.minimum(cucumber, 1)
#     N = np.matlib.repmat(np.reshape([i for i in range(-99,100)], (1, 199)), P.shape[0], 1)
#     Ynot = np.matlib.repmat(Y, 1, 199)
#     return np.sum(np.square(C - (N>=Y)))/199/C.shape[0]


# In[14]:


# best = 1000
# optimo = None
# for i in np.arange(0,1.01,0.01):
#     junk = i*P_NN2 + (1-i)*P_xgb1
#     noo = CRPS(junk, Y[20853:, :])
#     if noo < best:
#         best = noo
#         optimo = (i, j, 1-i-j)
# print(optimo)


# In[15]:


# create an nfl environment
env = nflrush.make_env()


# In[16]:


def make_my_predictions(model1, model2, model3, test_df, sample_predictions):
    test_df = error_correcting_codes(test_df)
#     # Neural Network I

#     X_downer = doubledown(down_situation(test_df.copy()), doublings)
#     X_phys = generate_physics(test_df.copy(), forward_actions, timestep, doublings)
#     X = np.concatenate((X_downer, X_phys), axis=1)
#     X = normalize(X, meen_I, sigma_I)
#     X = np.dot(X, PCs)
#     my_predictions_I = np.mean(model1.predict(X), axis=0)

    # Neural Network II
    X = np.concatenate((down_situation(test_df.copy()), 
                    generate_physics_II(test_df.copy()),
                    RB_wins_it(test_df.copy())), axis=1)
    X = normalize(X, meen_II, sigma_II)
    my_predictions_II = model2.predict([X[:,:11]]+[X[:,7*i+11:7*i+18] for i in range(0,11)]+
                   [X[:,7*i+11:7*i+18] for i in range(11,21)]+[X[:,158:]])
    
    # XGB I
#     X = np.concatenate(
#                 (down_situation(test_df.copy()), 
#                  generate_physics(test_df.copy(), forward_actions, timestep)), 
#             axis=1)
#     X = normalize(X, meen_III, sigma_III)
#     X = np.dot(X, PCs_III)
    offset = int((model3.predict(X) - np.argmax(model3.predict_proba(X), axis=1))[0])
    my_predictions_III = model3.predict_proba(X)
    my_predictions_III = np.concatenate((np.zeros((1, 99+offset)), my_predictions_III, np.ones((1, 100-my_predictions_III.shape[1]-offset))), axis=1)

    # Ensemble
#     my_predictions = 0.8*(0.62*my_predictions_I + 0.38*my_predictions_II) + 0.2*my_predictions_III
    my_predictions = np.reshape(0.62*my_predictions_II + 0.38*my_predictions_III, (199,0))
#     my_predictions = np.reshape(my_predictions_III, (199,0))
    
    
    cucumber = 0
    for i in range(1, my_predictions.shape[0]-1):
        cucumber += np.maximum(my_predictions[i], 0)
        sample_predictions.iloc[0, i] = np.minimum(cucumber, 1)

    return sample_predictions


# In[17]:


for (test_df, sample_prediction_df) in env.iter_test():
    predictions_df = make_my_predictions(model_xgb1, model_NN2, model_xgb1, test_df, sample_prediction_df)
    env.predict(predictions_df)
        
env.write_submission_file()

