#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np
import pandas as pd
import kagglegym

# Credit to the kernel https://www.kaggle.com/scirpus/two-sigma-financial-modeling/last-public-gp
if True:
    print('Started')
    low_y_cut = -0.086093
    high_y_cut = 0.093497
    env = kagglegym.make()
    observation = env.reset()
    train = observation.train
    print(train.y.mean())
    y_is_above_cut = (train.y > high_y_cut)
    y_is_below_cut = (train.y < low_y_cut)
    y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
    median_values = train[y_is_within_cut].median(axis=0)
    defaulty = train[y_is_within_cut].y.mean()
    defaultsids = dict(train[y_is_within_cut].groupby(["id"])["y"].median())
    previousTechnical11 = {}
    previousTechnical13 = {}
    previousTechnical20 = {}
    previousTechnical25 = {}
    previousTechnical30 = {}
    previousTechnical44 = {}
    previousFundamental0 = {}


# In[3]:


# Train
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()

model_cols = ["technical_20_Cur", "technical_30_Cur"]
target_col = 'y'

if True:    
    train.fillna(median_values, inplace=True)
    
    train['technical_20_Cur'] = train["technical_20"]
    train['technical_30_Cur'] = train["technical_30"]
    
    x_train = train[y_is_within_cut][model_cols]
    y = train[y_is_within_cut][target_col]

    print("Train Data: ", x_train.shape, y.shape)
    lr_model.fit(x_train.values,y.values)


# In[4]:


if True:
    new_data = None
    count = 0
    nb_refresh = 0
    nb_threshold = 100
    
    while True:
        firstsids = []
        yarray = np.zeros(observation.target.y.shape[0])
        observation.features.fillna(median_values, inplace=True)
        timestamp = observation.features["timestamp"][0]
        allData = None
        for i in range(observation.target.y.shape[0]):
            sid = observation.features["id"].values[i]
            if(sid in previousTechnical11.keys()):
                data = np.zeros(shape=(1, 15))
                data[0, 0] = previousTechnical11[sid]
                data[0, 1] = observation.features["technical_11"][i]
                data[0, 2] = previousTechnical13[sid]
                data[0, 3] = observation.features["technical_13"][i]
                data[0, 4] = previousTechnical20[sid]
                data[0, 5] = observation.features["technical_20"][i]
                data[0, 6] = previousTechnical25[sid]
                data[0, 7] = observation.features["technical_25"][i]
                data[0, 8] = previousTechnical30[sid]
                data[0, 9] = observation.features["technical_30"][i]
                data[0, 10] = previousTechnical44[sid]
                data[0, 11] = observation.features["technical_44"][i]
                data[0, 12] = previousFundamental0[sid]
                data[0, 13] = observation.features["fundamental_0"][i]
                data[0, 14] = sid
                if(allData is None):
                    allData = data.copy()
                else:
                    allData = np.concatenate([allData, data])
            else:
                yarray[i] = -999999
                firstsids.append(sid)

            previousTechnical11[sid] =                 observation.features["technical_11"][i]
            previousTechnical13[sid] =                 observation.features["technical_13"][i]
            previousTechnical20[sid] =                 observation.features["technical_20"][i]
            previousTechnical25[sid] =                 observation.features["technical_25"][i]
            previousTechnical30[sid] =                 observation.features["technical_30"][i]
            previousTechnical44[sid] =                 observation.features["technical_44"][i]
            previousFundamental0[sid] =                 observation.features["fundamental_0"][i]
        if(allData is not None):
            gpdata = pd.DataFrame({'technical_11_Prev': allData[:, 0],
                                   'technical_11_Cur': allData[:, 1],
                                   'technical_13_Prev': allData[:, 2],
                                   'technical_13_Cur': allData[:, 3],
                                   'technical_20_Prev': allData[:, 4],
                                   'technical_20_Cur': allData[:, 5],
                                   'technical_25_Prev': allData[:, 6],
                                   'technical_25_Cur': allData[:, 7],
                                   'technical_30_Prev': allData[:, 8],
                                   'technical_30_Cur': allData[:, 9],
                                   'technical_44_Prev': allData[:, 10],
                                   'technical_44_Cur': allData[:, 11],
                                   'fundamental_0_Prev': allData[:, 12],
                                   'fundamental_0_Cur': allData[:, 13],
                                   'id': allData[:, 14]
                                })
            x_train = gpdata[model_cols]
            yarray[yarray == 0] = lr_model.predict(x_train.values).clip(low_y_cut, high_y_cut)

        yarray[yarray == -999999] = defaulty
        observation.target.y = yarray
        target = observation.target
        
        observation, reward, done, info = env.step(target)
        
        if((timestamp % 100 == 0)) or (count == 0):
            print(timestamp, reward)

        if done:
            break
           
        # Re-Train
        if(allData is not None):
            if count == 0:
                new_data = gpdata
            else:
                new_data = pd.concat([new_data, gpdata])
            count += 1

        if True and count >= nb_threshold: 
            count = 0
            nb_refresh += 1

            new_data['ft_Cur'] = new_data["technical_13_Cur"] + new_data["technical_20_Cur"] - new_data["technical_30_Cur"]
            new_data['ft_Next'] = new_data[['id', 
'ft_Cur']].groupby('id')['ft_Cur'].shift(-1)
            
            # Re-calculate y
            new_data['y'] = (new_data['ft_Next'] - 0.92*new_data['ft_Cur'])/0.07

            # RE-TRAIN
            new_data1 = new_data[new_data['y'].notnull()]

            y_is_above_cut = (new_data1.y > high_y_cut)
            y_is_below_cut = (new_data1.y < low_y_cut)
            y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

            new_data2 = new_data1[y_is_within_cut]
            
            x_train = new_data2[model_cols]
            y = new_data2[target_col]

            print("UPDATE:", nb_refresh, x_train.shape, y.shape)

            lr_model.fit(x_train.values,y.values)
            #
            
    print(info)
    print('Finished')

