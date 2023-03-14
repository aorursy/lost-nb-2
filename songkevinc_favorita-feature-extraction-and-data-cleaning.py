#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

sns.set_style('whitegrid')


# In[ ]:


train_db = pd.read_csv('../input/train.csv', parse_dates=['date'])
test_db = pd.read_csv('../input/test.csv', parse_dates=['date'])
transactions_db = pd.read_csv('../input/transactions.csv', parse_dates=['date'])
oilprice_db = pd.read_csv('../input/oil.csv', parse_dates=['date'])
holiday_db = pd.read_csv('../input/holidays_events.csv', parse_dates=['date'])
items_db = pd.read_csv('../input/items.csv')
stores_db = pd.read_csv('../input/stores.csv')


# In[ ]:


print("Training Data")
for col in train_db.columns:
    print (col, train_db[col].isnull().any())
print ("="*70)
print("Test Data")
for col in test_db.columns:
    print (col, test_db[col].isnull().any())


# In[ ]:


supData = [transactions_db, oilprice_db, holiday_db, items_db, stores_db]
for sup in supData:
    print ("="*70)
    for col in sup.columns:
        print (col, sup[col].isnull().any())
        
del supData, sup # deleting them to make sure I have enough memory


# In[ ]:


# Data Cleaning
train_db.loc[(train_db.unit_sales<0),'unit_sales'] = 0 # Cleaning all negative values to be 0

# Add 'dow' and 'doy'
train_db['dow'] = train_db['date'].dt.dayofweek # adding day of week as a feature
train_db['doy'] = train_db['date'].dt.dayofyear # adding day of year as a feature

test_db['dow'] = test_db['date'].dt.dayofweek # adding day of week as a feature
test_db['doy'] = test_db['date'].dt.dayofyear # adding day of year as a feature


# In[ ]:


train_db.loc[:,'onpromotion'].fillna(2, inplace=True) # Replace NaNs with 2
train_db.loc[:,'onpromotion'].replace(True, 1, inplace=True) # Replace Trues with 1
train_db.loc[:,'onpromotion'].replace(False, 0, inplace=True) # Replace Falses with 0

# Do the same for test set
test_db.loc[:,'onpromotion'].fillna(2, inplace=True) # Replace NaNs with 2
test_db.loc[:,'onpromotion'].replace(True, 1, inplace=True) # Replace Trues with 1
test_db.loc[:,'onpromotion'].replace(False, 0, inplace=True) # Replace Falses with 0


# In[ ]:


# Grouping columns unit sales by
# item_nbr, store_nbr, dow (day of week)
# And storing means as dataframe
ma_dw = train_db[['item_nbr','store_nbr','dow','unit_sales']]             .groupby(['item_nbr','store_nbr','dow'])['unit_sales']             .mean().to_frame('madw')
ma_dw.reset_index(inplace=True)

# Storing weekly averages
ma_wk = ma_dw[['item_nbr', 'store_nbr','madw']]        .groupby(['item_nbr', 'store_nbr'])['madw']        .mean().to_frame('mawk')
ma_wk.reset_index(inplace=True)


# In[ ]:


# Oilprice dataset has many missing date values
# I'm just going to create a new database with every day registered
# Fill in the values by interpolating

index = pd.date_range(start='2013-01-01', end='2017-08-31')
new_oilprice_db = pd.DataFrame(index=index, columns=['date'] )
new_oilprice_db['date'] = index
new_oilprice_db.reset_index(inplace=True)
del new_oilprice_db['index']

# Linearly interpolating and manually filling in 3 points for linear interpolation
td = oilprice_db.date.diff() # time differece vector
interp = oilprice_db.dcoilwtico.shift(1) + ((oilprice_db.dcoilwtico.shift(-1) - oilprice_db.dcoilwtico.shift(1)))         * td / (td.shift(-1) + td)

oilprice_db['dcoilwtico'] = oilprice_db['dcoilwtico'].fillna(interp)

# Manually added the very first point 
oilprice_db['dcoilwtico'][0] = 93.14
oilprice_db['dcoilwtico'][1174] = 46.57
oilprice_db['dcoilwtico'][1175] = 46.75

# Merge the new oil price dataframe with the old dataframe
new_oilprice_db = pd.merge(new_oilprice_db, oilprice_db, on='date', how='left')

interp = new_oilprice_db.dcoilwtico.shift(2) +
         ((new_oilprice_db.dcoilwtico.shift(-2) - new_oilprice_db.dcoilwtico.shift(2)))           / 2
new_oilprice_db['dcoilwtico'] = new_oilprice_db['dcoilwtico'].fillna(interp)

# Repeating interpolating twice because the first time only filled 1 of the back-to-back
# NaN values. If I repeate the interpolation twice, it should fill in all values

interp = new_oilprice_db.dcoilwtico.shift(1) +
         ((new_oilprice_db.dcoilwtico.shift(-1) - new_oilprice_db.dcoilwtico.shift(1)))           / 2
new_oilprice_db['dcoilwtico'] = new_oilprice_db['dcoilwtico'].fillna(interp)


# In[ ]:


new_oilprice_db[new_oilprice_db['dcoilwtico'].isnull()]


# In[ ]:


train = pd.merge(train_db, stores_db, on='store_nbr', how='left')
train = pd.merge(train, ma_dw, on=['item_nbr', 'store_nbr', 'dow'], how='left')
train = pd.merge(train, ma_wk, on=['item_nbr', 'store_nbr'], how='left')
train = pd.merge(train, items_db, on='item_nbr', how='left')
train = pd.merge(train, oilprice_db, on='date', how='left')
train = pd.merge(train, holiday_db, on='date', how='left')

train.head()


# In[ ]:


for i in train.columns:
    print i, train[i].isnull().any()


# In[ ]:


test = pd.merge(test_db, stores_db, on='store_nbr', how='left')
test = pd.merge(test, ma_dw, on=['item_nbr', 'store_nbr', 'dow'], how='left')
test = pd.merge(test, ma_wk, on=['item_nbr', 'store_nbr'], how='left')
test = pd.merge(test, items_db, on='item_nbr', how='left')
test = pd.merge(test, holiday_db, on='date', how='left') # we have 2017 holiday info
test = pd.merge(test, oilprice_db, on='date', how='left')

test.head()


# In[ ]:


del ma_dw, ma_wk, holiday_db, oilprice_db, stores_db, test_db, items_db


# In[ ]:


train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train['day'] = train['date'].dt.day


# In[ ]:


test['month'] = test['date'].dt.month
test['year'] = test['date'].dt.year
test['day'] = test['date'].dt.day


# In[ ]:


#train.dropna(inplace=True)
x_train = train[['store_nbr', 'item_nbr', 'cluster', 'dow', 'doy', 'madw',
                 'perishable', 'dcoilwtico','onpromotion', 'day', 'month', 'year']]
y_train = train['unit_sales']

del train


# In[ ]:


joblib.dump(x_train, 'x_train/x_train.pkl')
joblib.dump(y_train, 'y_train/y_train.pkl')


# In[ ]:


# Splitting x_train into 
from sklearn.cross_validation import train_test_split

xRealTrain, xRealTest, yRealTrain, yRealTest = train_test_split(x_train, y_train, train_size=0.1)
# train_size = between 0 and 1.

