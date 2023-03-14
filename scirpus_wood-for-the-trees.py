#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


low_y_cut = -0.086093
high_y_cut = 0.093497


# In[3]:


directory = '../input/'
with pd.HDFStore(directory+'train.h5') as train:
    fullset = train.get('train')
fullset = fullset[['id', 'technical_20', 'technical_30', 'y']]
print(fullset.shape)
fullset.fillna(fullset.median(), inplace=True)
fullset = fullset[fullset['id'] == 2047]
fullset = fullset[fullset.y < high_y_cut]
fullset = fullset[fullset.y > low_y_cut]
print(fullset.shape)


# In[4]:


y = fullset.iloc[1:-1]['y'].values
fullset.drop('y', inplace=True, axis=1)
wdw = pd.DataFrame()
for i in range(3):
    if(i == 0):
        wdw['technical_20_Row_Offset_0'] =             fullset['technical_20'].iloc[0:-2].values
        wdw['technical_30_Row_Offset_0'] =             fullset['technical_30'].iloc[0:-2].values
    elif(i == 1):
        wdw['technical_20_Row_Offset_1'] =             fullset['technical_20'].iloc[1:-1].values
        wdw['technical_30_Row_Offset_1'] =             fullset['technical_30'].iloc[1:-1].values
    else:
        wdw['technical_20_Row_Offset_2'] =             fullset['technical_20'].iloc[2:].values
        wdw['technical_30_Row_Offset_2'] =             fullset['technical_30'].iloc[2:].values
wdw['y'] = y


# In[5]:


print(fullset.head())


# In[6]:


print(wdw.head())


# In[7]:


def r_score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    r = np.sign(r2) * np.sqrt(np.abs(r2))
    return max(-1, r)


# In[8]:


def GPTechnicalPrediction(data):
    p = (((((8.0) * (data["technical_20_Row_Offset_2"] + (data["technical_30_Row_Offset_1"] - (data["technical_20_Row_Offset_1"] + data["technical_30_Row_Offset_2"])))) - (data["technical_30_Row_Offset_1"] - ((((data["technical_30_Row_Offset_1"] + ((((data["technical_30_Row_Offset_1"] + data["technical_20_Row_Offset_1"]) * data["technical_30_Row_Offset_1"]) + data["technical_20_Row_Offset_1"])/2.0))/2.0) + data["technical_20_Row_Offset_2"])/2.0)))) )
    return p.values.clip(low_y_cut,high_y_cut)


# In[9]:


yhat = GPTechnicalPrediction(wdw)


# In[10]:


print('R Score: ',r_score(wdw.y.values, yhat))


# In[11]:


plt.figure(figsize=(8,8))
plt.plot(wdw.y.values)
plt.plot(yhat)

