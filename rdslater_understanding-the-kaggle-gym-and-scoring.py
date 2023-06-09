#!/usr/bin/env python
# coding: utf-8

# In[1]:


#not my code.  Please direct your thanks to Yunfeng Zhu
import kagglegym
import numpy as np
import pandas as pd
# from sklearn import linear_model as lm
from sklearn.linear_model import Ridge

target = 'y'

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

# cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']

# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

ridge1 = Ridge() ## f11 only
ridge2 = Ridge() ## t30 and f11
ridge3 = Ridge() ## t20 and f11
ridge4 = Ridge() ## t30, t20, f11

train = train.loc[y_is_within_cut]

index1 = train.query("technical_30 == 0.0 & technical_20 == 0.0").index
index2 = train.query("technical_30 != 0.0 & technical_20 == 0.0").index
index3 = train.query("technical_30 == 0.0 & technical_20 != 0.0").index
index4 = train.query("technical_30 != 0.0 & technical_20 != 0.0").index

ridge1.fit(train.loc[index1, ["fundamental_11"]].values, train.loc[index1].y)
ridge2.fit(train.loc[index2, ['technical_30', 'fundamental_11']].values, train.loc[index2].y)
ridge3.fit(train.loc[index3, ['technical_20', 'fundamental_11']].values, train.loc[index3].y)
ridge4.fit(train.loc[index4, ['technical_30', 'technical_20', 'fundamental_11']].values, train.loc[index4].y)


# model = Ridge()
# model.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, target])

# ymean_dict = dict(train.groupby(["id"])["y"].mean())
ymedian_dict = dict(train.groupby(["id"])["y"].median())


def get_weighted_y(series):
    id, y = series["id"], series["y"]
    # return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y
    return 0.95 * y + 0.05 * ymedian_dict[id] if id in ymedian_dict else y


# In[2]:


observation.features.fillna(mean_values, inplace=True)
# test_x = np.array(observation.features[cols_to_use].values)
# observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)

index1 = observation.features.query("technical_30 == 0.0 & technical_20 == 0.0").index
index2 = observation.features.query("technical_30 != 0.0 & technical_20 == 0.0").index
index3 = observation.features.query("technical_30 == 0.0 & technical_20 != 0.0").index
index4 = observation.features.query("technical_30 != 0.0 & technical_20 != 0.0").index

if len(index1) > 0:
    observation.target.loc[index1, 'y'] = ridge1.predict(observation.features.loc[index1, ["fundamental_11"]].values).clip(low_y_cut, high_y_cut)
if len(index2) > 0:
    observation.target.loc[index2, 'y'] = ridge2.predict(observation.features.loc[index2, ['technical_30', 'fundamental_11']].values).clip(low_y_cut, high_y_cut)
if len(index3) > 0:
    observation.target.loc[index3, 'y'] = ridge3.predict(observation.features.loc[index3, ['technical_20', 'fundamental_11']].values).clip(low_y_cut, high_y_cut)
if len(index4) > 0:
    observation.target.loc[index4, 'y'] = ridge4.predict(observation.features.loc[index4, ['technical_30', 'technical_20', 'fundamental_11']].values).clip(low_y_cut, high_y_cut)

## weighted y using average value
observation.target.y = observation.target.apply(get_weighted_y, axis = 1)

#observation.target.fillna(0, inplace=True)
target = observation.target
timestamp = observation.features["timestamp"][0]


# In[3]:


real_y=pd.read_hdf("../input/train.h5")
real_y=real_y.iloc[:,[0,1,-1]]
real_y=real_y[real_y.timestamp>905]


# In[4]:


u=np.mean(real_y[real_y.timestamp==906].y)
num=np.sum((real_y[real_y.timestamp==906].y.values-observation.target.y)**2)
dem=np.sum((real_y[real_y.timestamp==906].y-u)**2)
r2=1-num/dem
if r2<0:
    print(-np.sqrt(-r2))
else:
    print(np.sqrt(r2))


# In[5]:


observation, reward, done, info = env.step(target)
print(reward)


# In[6]:


observation.features.fillna(mean_values, inplace=True)
# test_x = np.array(observation.features[cols_to_use].values)
# observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)

index1 = observation.features.query("technical_30 == 0.0 & technical_20 == 0.0").index
index2 = observation.features.query("technical_30 != 0.0 & technical_20 == 0.0").index
index3 = observation.features.query("technical_30 == 0.0 & technical_20 != 0.0").index
index4 = observation.features.query("technical_30 != 0.0 & technical_20 != 0.0").index

if len(index1) > 0:
    observation.target.loc[index1, 'y'] = ridge1.predict(observation.features.loc[index1, ["fundamental_11"]].values).clip(low_y_cut, high_y_cut)
if len(index2) > 0:
    observation.target.loc[index2, 'y'] = ridge2.predict(observation.features.loc[index2, ['technical_30', 'fundamental_11']].values).clip(low_y_cut, high_y_cut)
if len(index3) > 0:
    observation.target.loc[index3, 'y'] = ridge3.predict(observation.features.loc[index3, ['technical_20', 'fundamental_11']].values).clip(low_y_cut, high_y_cut)
if len(index4) > 0:
    observation.target.loc[index4, 'y'] = ridge4.predict(observation.features.loc[index4, ['technical_30', 'technical_20', 'fundamental_11']].values).clip(low_y_cut, high_y_cut)

## weighted y using average value
observation.target.y = observation.target.apply(get_weighted_y, axis = 1)

#observation.target.fillna(0, inplace=True)
target = observation.target
timestamp = observation.features["timestamp"][0]


# In[7]:


u=np.mean(real_y[real_y.timestamp==907].y)
num=np.sum((real_y[real_y.timestamp==907].y.values-observation.target.y)**2)
dem=np.sum((real_y[real_y.timestamp==907].y-u)**2)
r2=1-num/dem
if r2<0:
    print(-np.sqrt(-r2))
else:
    print(np.sqrt(r2))


# In[8]:


observation, reward, done, info = env.step(target)
print(reward)


# In[9]:


import kagglegym
import numpy as np
import pandas as pd
# from sklearn import linear_model as lm
from sklearn.linear_model import Ridge
real_y=pd.read_hdf("../input/train.h5")
real_y=real_y.iloc[:,[0,1,-1]]
real_y=real_y[real_y.timestamp>905]

target = 'y'

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

# cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']

# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

ridge1 = Ridge() ## f11 only
ridge2 = Ridge() ## t30 and f11
ridge3 = Ridge() ## t20 and f11
ridge4 = Ridge() ## t30, t20, f11

train = train.loc[y_is_within_cut]

index1 = train.query("technical_30 == 0.0 & technical_20 == 0.0").index
index2 = train.query("technical_30 != 0.0 & technical_20 == 0.0").index
index3 = train.query("technical_30 == 0.0 & technical_20 != 0.0").index
index4 = train.query("technical_30 != 0.0 & technical_20 != 0.0").index

ridge1.fit(train.loc[index1, ["fundamental_11"]].values, train.loc[index1].y)
ridge2.fit(train.loc[index2, ['technical_30', 'fundamental_11']].values, train.loc[index2].y)
ridge3.fit(train.loc[index3, ['technical_20', 'fundamental_11']].values, train.loc[index3].y)
ridge4.fit(train.loc[index4, ['technical_30', 'technical_20', 'fundamental_11']].values, train.loc[index4].y)


# model = Ridge()
# model.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, target])

# ymean_dict = dict(train.groupby(["id"])["y"].mean())
ymedian_dict = dict(train.groupby(["id"])["y"].median())


def get_weighted_y(series):
    id, y = series["id"], series["y"]
    # return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y
    return 0.95 * y + 0.05 * ymedian_dict[id] if id in ymedian_dict else y

tot=0
while True:
    observation.features.fillna(mean_values, inplace=True)
    # test_x = np.array(observation.features[cols_to_use].values)
    # observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)
    
    index1 = observation.features.query("technical_30 == 0.0 & technical_20 == 0.0").index
    index2 = observation.features.query("technical_30 != 0.0 & technical_20 == 0.0").index
    index3 = observation.features.query("technical_30 == 0.0 & technical_20 != 0.0").index
    index4 = observation.features.query("technical_30 != 0.0 & technical_20 != 0.0").index
    
    if len(index1) > 0:
        observation.target.loc[index1, 'y'] = ridge1.predict(observation.features.loc[index1, ["fundamental_11"]].values).clip(low_y_cut, high_y_cut)
    if len(index2) > 0:
        observation.target.loc[index2, 'y'] = ridge2.predict(observation.features.loc[index2, ['technical_30', 'fundamental_11']].values).clip(low_y_cut, high_y_cut)
    if len(index3) > 0:
        observation.target.loc[index3, 'y'] = ridge3.predict(observation.features.loc[index3, ['technical_20', 'fundamental_11']].values).clip(low_y_cut, high_y_cut)
    if len(index4) > 0:
        observation.target.loc[index4, 'y'] = ridge4.predict(observation.features.loc[index4, ['technical_30', 'technical_20', 'fundamental_11']].values).clip(low_y_cut, high_y_cut)

    ## weighted y using average value
    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)
    
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    ###here is the only thing I changed###
    tot=np.sum((real_y[real_y['timestamp']==timestamp].y.values-observation.target.y)**2)+tot
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)


# In[10]:


u=np.mean(real_y.y)
dem=np.sum((real_y.y.values-u)**2)
r2=1-tot/dem
if r2<0:
    print(-np.sqrt(-r2))
else:
    print(np.sqrt(r2))

