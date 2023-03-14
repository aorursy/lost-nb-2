#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc
import glob
import os
import json

import pprint

from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D,     MaxPooling1D, Dense, BatchNormalization, Dropout, Embedding, Reshape, Concatenate
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import scipy as sp

from collections import Counter
from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
import lightgbm as lgb

from gensim.models import KeyedVectors

import nltk
import string
import re
import pickle
#import lda

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
get_ipython().run_line_magic('matplotlib', 'inline')

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
#from bokeh.transform import factor_cmap

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import math
from subprocess import check_output
from tensorflow.keras.preprocessing.text import Tokenizer

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("lda").setLevel(logging.WARNING)

import zipfile
from subprocess import check_output
from keras.preprocessing.sequence import pad_sequences
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('/kaggle/input/mercari/train.tsv', sep='\t')
test = pd.read_csv('/kaggle/input/mercari/test.tsv', sep='\t')


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


print(train.shape)
print(test.shape)


# In[6]:


train.info()


# In[7]:


test.info()


# In[8]:


print('Missing values of data train\n',train.isnull().sum())
print('----------------------')
print('Missing values of data test\n',test.isnull().sum())


# In[9]:


def process_mis_values(dataset):
    dataset.category_name.fillna(value = 'missing', inplace = True)
    dataset.brand_name .fillna(value = 'missing', inplace = True)
    dataset.item_description .fillna(value = 'missing', inplace = True)
    return dataset


# In[10]:


train = process_mis_values(train)
test = process_mis_values(test)
print(train.isnull().sum())
print(test.isnull().sum())


# In[11]:


# Check types of variable
# Numeric variable
number = [f for f in train.columns if train.dtypes[f] != 'object']
# Object variable
objects = [f for f in train.columns if train.dtypes[f] == 'object']
print(number)
print(objects)


# In[12]:


le = LabelEncoder()
le.fit(np.hstack([train.category_name,test.category_name]))
train.category_name = le.transform(train.category_name)
test.category_name = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name,test.brand_name]))
train.brand_name = le.transform(train.brand_name)
test.brand_name = le.transform(test.brand_name)

del le


# In[13]:


train.head()


# In[14]:


test.head()


# In[15]:


# Text to sequence processing
token = Tokenizer()
raw_text = np.hstack([train.item_description.str.lower(), test.item_description.str.lower()])
token.fit_on_texts(raw_text)

train['seq_item_descri'] = token.texts_to_sequences(train.item_description.str.lower())
test['seq_item_descri'] = token.texts_to_sequences(test.item_description.str.lower())
train['seq_name'] = token.texts_to_sequences(train.name.str.lower())
test['seq_name'] = token.texts_to_sequences(test.name.str.lower())
train.head()


# In[ ]:





# In[16]:


test.head()


# In[17]:


#print(len(train.seq_item_descri.max()))
#print(type(train.seq_name[2]))


# In[18]:


#print(token.word_index)


# In[19]:


max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))),np.max(test.seq_name.apply(lambda x:len(x)))])
max_seq_item_descri = np.max([np.max(train.seq_item_descri.apply(lambda x: len(x))),np.max(test.seq_item_descri.apply(lambda x:len(x)))])


# In[20]:


print('max name seq', max_name_seq)
print('max seq item descri',max_seq_item_descri )


# In[21]:


train.seq_name.apply(lambda x: len(x)).hist()


# In[22]:


train.seq_item_descri.apply(lambda x: len(x)).hist()


# In[23]:


#Base on the histograms, we select the next lengths
max_name_seq = 10
max_descri = 100
max_text = np.max([np.max(train.seq_name.max()),
                np.max(test.seq_name.max()),
                np.max(train.seq_item_descri.max()),
                np.max(test.seq_item_descri.max())])+2
max_categoty = np.max([np.max(train.category_name),np.max(test.category_name)])+1
max_brand = np.max([np.max(train.brand_name), np.max(train.brand_name)])+1
max_condition = np.max([train.item_condition_id.max(), test.item_condition_id.max()])+1


# In[24]:


print(max_text)
print(max_categoty)
print(max_brand)
print(max_condition)


# In[25]:


#SCALE target variable
train["target"] = np.log(train.price+1)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
train["target"] = target_scaler.fit_transform(train.target.values.reshape(-1,1))
pd.DataFrame(train.target).hist()


# In[26]:


#EXTRACT DEVELOPTMENT TEST
#dtrain, dvalid = train_test_split(train[['train_id','brand_name','category_name','item_condition_id','price','shipping','seq_item_descri','seq_name']], random_state=123, train_size=0.99)
#print(dtrain.shape)
#print(dvalid.shape)


# In[27]:


def get_keras_data(dataset):
    df_name = pd.DataFrame(data=pad_sequences(dataset.seq_name, maxlen=max_name_seq),index = train['train_id'], columns=['name_factor' + '_' + str(k) for k in range(max_name_seq)])
    df_name = df_name.reset_index()
    df_item = pd.DataFrame(data=pad_sequences(dataset.seq_item_descri, maxlen=max_descri),index = train['train_id'], columns=['item_factor' + '_' + str(k) for k in range(max_descri)])
    df_item = df_item.reset_index()
    X = dataset[['train_id','item_condition_id','brand_name','category_name','shipping','target']]
    X_1 = pd.merge(df_name,df_item, on = 'train_id')
    X_final = pd.merge(X,X_1, on = 'train_id')
    X_final = X_final.drop('train_id',axis = 1)
    return X_final


# In[28]:


X = get_keras_data(train)


# In[29]:


X.head()


# In[30]:


def get_keras_data_stg2(dataset):
    df_name = pd.DataFrame(data=pad_sequences(dataset.seq_name, maxlen= max_name_seq),index = test['test_id'], columns=['name_factor' + '_' + str(k) for k in range(max_name_seq)])
    df_name = df_name.reset_index()
    df_item = pd.DataFrame(data=pad_sequences(dataset.seq_item_descri, maxlen= max_descri),index = test['test_id'], columns=['item_factor' + '_' + str(k) for k in range(max_descri)])
    df_item = df_item.reset_index()
    X = dataset[['test_id','item_condition_id','brand_name','category_name','shipping']]
    X_1 = pd.merge(df_name,df_item, on = 'test_id')
    X_final = pd.merge(X,X_1, on = 'test_id')
    X_final = X_final.drop('test_id',axis = 1)
    return X_final


# In[31]:


X_stg2 = get_keras_data_stg2(test)


# In[32]:


X_stg2.head()


# In[33]:


# Model base
# LightGBM
import lightgbm as lgb

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 70,
          'max_depth': 9,
          'learning_rate': 0.5,
          'bagging_fraction': 0.85,
          'feature_fraction': 0.8,
          'min_split_gain': 0.02,
          'min_child_samples': 150,
          'min_child_weight': 0.02,
          'lambda_l2': 0.0475,
          'verbosity': -1,
          'data_random_seed': 17,
          'tree learner':'feature'
          }

# Additional parameters:
early_stop = 1000
verbose_eval = 50
num_rounds = 5000
n_splits = 5


# In[34]:


from sklearn.model_selection import KFold


# In[35]:


def LGBM_train(X_train, X_test):
    
    kfold = KFold(n_splits, random_state = 1337 )
    oof_train = np.zeros([X_train.shape[0]])
    oof_test = np.zeros([X_test.shape[0], n_splits])
    i = 0
    for train_index, valid_index in kfold.split(X_train, X_train['target'].values):
    
        X_tr = X_train.iloc[train_index, :]
        X_val = X_train.iloc[valid_index, :]

        y_tr = X_tr['target'].values
        X_tr = X_tr.drop(['target'], axis=1)

        y_val = X_val['target'].values
        X_val = X_val.drop(['target'], axis=1)
    
#         print('\ny_tr distribution: {}'.format(Counter(y_tr)))

        d_train = lgb.Dataset(X_tr, label=y_tr)
        d_valid = lgb.Dataset(X_val, label=y_val)
        watchlist = [d_train, d_valid]

        print('training LGB:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        val_pred = model.predict(X_val, num_iteration = model.best_iteration)
        test_pred = model.predict(X_test, num_iteration = model.best_iteration)

        oof_train[valid_index] = val_pred
        oof_test[:,i] = test_pred
        i +=1
        
    return oof_train, oof_test, model


# In[36]:


oof_train, oof_test, model = LGBM_train(X,X_stg2)


# In[37]:


val_preds = target_scaler.inverse_transform(oof_test)
val_preds = np.expm1(val_preds)


# In[38]:


A = pd.DataFrame(data = val_preds)


# In[39]:


A


# In[40]:


A['final_price'] = A.mean(axis = 1)


# In[41]:


A


# In[42]:


submission = pd.DataFrame({'test_id': test['test_id'], 'price': A['final_price']})


# In[43]:


submission


# In[44]:


submission.to_csv("submission.csv", index=False)

