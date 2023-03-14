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
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/train/Train.csv',low_memory=False, parse_dates=["saledate"])


# In[ ]:


data.saledate


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


display_all(data.sample(10).T)


# In[ ]:


display_all(data.isnull().sum().sort_values(ascending=False)/len(data))


# In[ ]:


data_needed = data[['SalesID',                    
'state'  ,                 
'fiProductClassDesc',          
'fiBaseModel',         
'fiModelDesc' ,        
'ProductGroup' ,       
'saledate',      
'datasource',     
'ModelID' ,    
'MachineID',   
'SalePrice' ,
'YearMade',   
'ProductGroupDesc',  
'Enclosure', 
'auctioneerID' ,
'Hydraulics',
'fiSecondaryDesc'  ,
'Coupler' ,
'Forks',
'ProductSize'  ,
'Transmission']]


# In[ ]:


data_needed.SalePrice = np.log(data_needed.SalePrice)


# In[ ]:


data_needed.head()


# In[ ]:


add_datepart(data_needed, 'saledate',drop=False)


# In[ ]:


data_needed.sort_values('saledate',inplace=True)


# In[ ]:


data_needed.head(20)


# In[ ]:


data_needed.drop('saledate',axis=1,inplace=True)


# In[ ]:


display_all(data_needed.head())


# In[ ]:


train_cats(data_needed)


# In[ ]:


data_needed.dtypes


# In[ ]:


#let's see our data with the changes we have made so far
display_all(data_needed.head(100))


# In[ ]:


data_needed.state.cat.categories


# In[ ]:


data_needed.state.cat.codes.sort_index()


# In[ ]:


data_needed.isnull().sum().sort_values(ascending=False)/len(data_needed)*100


# In[ ]:


#let see in transmission column and try to make a better intution about why is data missing in this column
data_needed.Transmission


# In[ ]:


data_needed.drop('Transmission',axis=1,inplace=True)


# In[ ]:


#let's see the next column
data_needed.ProductSize       


# In[ ]:


#it's the sizes of the product so i think we can play around so will fill in values with the perivous instant as x[1] which
#is NaN will be filled with x[0]
data_needed.ProductSize.fillna(method = 'bfill', axis=0)


# In[ ]:


#coupler
data_needed.Coupler                 


# In[ ]:


#as before fill in the next value to the NaN
data_needed.Coupler.fillna(method = 'bfill', axis=0)


# In[ ]:


data_needed.fiSecondaryDesc         


# In[ ]:


data_needed.fiSecondaryDesc.fillna(method = 'bfill', axis=0)


# In[ ]:


data_needed.Hydraulics.fillna(method = 'bfill', axis=0)            


# In[ ]:


data_needed.auctioneerID             


# In[ ]:


#Since we finally have a numerical type data we will fill with the median of the column
data_needed.auctioneerID = data_needed.auctioneerID.fillna(data_needed.auctioneerID.median())


# In[ ]:


#small number of NaN so will just drop it
data_needed.Enclosure.dropna()              


# In[ ]:


df, y, nas = proc_df(data_needed, 'SalePrice')


# In[ ]:


#see the source code of the proc fn
get_ipython().run_line_magic('pinfo2', 'proc_df')


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df,y)


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=40,n_jobs=-1,oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])


# In[ ]:


preds[:,0], np.mean(preds[:,0]), y_valid[0]


# In[ ]:


set_rf_samples(80000)


# In[ ]:


m = RandomForestRegressor(n_estimators=80, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


#reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators=80, n_jobs=-1, min_samples_leaf=3, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=80, min_samples_leaf=4, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:




