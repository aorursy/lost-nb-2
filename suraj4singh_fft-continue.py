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

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[2]:


import matplotlib.pyplot as plt
from tqdm import tqdm #Makes iterations look better
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


# In[3]:


train_data = pd.read_csv('../input/train.csv',dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


# In[4]:


# Display the head of the dataframe
pd.options.display.precision = 10

train_data.head()


# In[5]:


# Dimensions of the given training data
print("Rows: {}, Columns: {}".format(train_data.shape[0],train_data.shape[1]))


# In[6]:


def features(df):
    


# In[7]:


# Create segments of 150000 data points as mentioned by the competition organisers.
from scipy.fftpack import fft
segment_size = 150000
num_segments = int(np.floor(train_data.shape[0]/segment_size))
features = []
bucket_size = 1500
num_features = int(np.floor(segment_size/bucket_size));
for i in range(num_features):
    features.append("feature" + str(i+1))
X_train = pd.DataFrame(index=range(num_segments),columns=features,dtype=np.float64)
y_train = pd.DataFrame(index=range(num_segments),columns=['time_to_failure'],dtype=np.float64)

for i in tqdm(range(num_segments)):
    segment_i = train_data.iloc[i*segment_size:i*segment_size+segment_size]
    x = segment_i['acoustic_data'].values
    y = segment_i['time_to_failure'].values[-1]
    yf = fft(x)
    for j in range(num_features):
        bucket = yf[j*bucket_size:min(segment_size, ((j+1)*bucket_size))]
        bucket = np.abs(bucket)
        X_train.loc[i, features[j]] = bucket.mean()
    
    
#     X_train.loc[i,'std'] = x.std()
#     X_train.loc[i,'max'] = x.max()
#     X_train.loc[i,'min'] = x.min()
    
    y_train.loc[i,'time_to_failure'] = y


# In[8]:


X_train.head()


# In[9]:


y_train.head()


# In[10]:


print("X_train Shape: {}, y_train Shape: {}".format(X_train.shape,y_train.shape))


# In[11]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled


# In[12]:


model = GradientBoostingRegressor(learning_rate=0.1,n_estimators=200,loss='ls').fit(X_train_scaled,y_train.values.flatten())
y_predictions = model.predict(X_train_scaled)
y_predictions


# In[13]:


mean_absolute_error(y_train.values.flatten(),y_predictions)


# In[14]:


# Create testing data / handle the testing part

submission_files = pd.read_csv('../input/sample_submission.csv',index_col='seg_id')
segment_size = 150000
bucket_size = 1500
num_features = int(np.floor(segment_size/bucket_size))
X_test = pd.DataFrame(columns=X_train.columns,index=submission_files.index,dtype=np.float64)

for seg_id in tqdm(X_test.index):
    segment = pd.read_csv('../input/test/'+seg_id+'.csv')
    x = segment['acoustic_data'].values
    yf = fft(x)
    for j in range(num_features):
        bucket = yf[j*bucket_size:min(segment_size, ((j+1)*bucket_size))]
        bucket = np.abs(bucket)
        X_test.loc[seg_id, features[j]] = bucket.mean()
X_test


# In[15]:


X_test_scaled = scaler.transform(X_test)
X_test_scaled


# In[16]:


y_test_predictions = model.predict(X_test_scaled)
submission_files['time_to_failure'] = y_test_predictions
submission_files.to_csv('submission3.csv')


# In[17]:




