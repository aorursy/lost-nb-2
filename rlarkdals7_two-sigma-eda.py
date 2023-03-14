#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
train_my_model(market_train_df, news_train_df)

for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
 predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
 env.predict(predictions_df)

env.write_submission_file()


# In[ ]:


# get environment!

from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


# market data
market_train_df.head()


# In[ ]:


# news data
news_train_df.head()


# In[ ]:


# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()


# In[ ]:


(market_obs_df, news_obs_df, predictions_template_df) = next(days)


# In[ ]:


#market test data
market_obs_df.head()


# In[ ]:


#news test data
news_obs_df.head()


# In[ ]:


#prediction template
predictions_template_df.head()


# In[ ]:


import pandas


# In[ ]:


news_train_df.describe()


# In[ ]:


market_train_df.describe()


# In[ ]:


market_train_df.head(15)


# In[ ]:


market_train_df.groupby('assetName').describe()

