#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import load_model
import pickle
import tensorflow_hub as hub
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

counter = 0


# In[2]:


train = pd.read_csv('../input/google-quest-challenge/train.csv')
test = pd.read_csv('../input/google-quest-challenge/test.csv')
sub = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
counter += 1


# In[3]:


model = load_model('../input/roberta-finetuned/distilbert_fold5.tf')
with open('../input/roberta-xte/Roberta_Xte.pkl', 'rb') as f:
  Xte = pickle.load(f)
test_preds = model.predict(Xte[:2])

counter += 1


# In[4]:


if counter == 2:
    print("executed first")
    sub.iloc[:,1:] = test_preds
    sub.to_csv('submission.csv',index=False)
elif counter == 1:
    print("executed second")
    sub = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
    #sub.iloc[:,1:] = test_preds[4].round(decimals = 5)
    sub.to_csv('submission.csv',index=False)

