#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Load data.
train = pd.read_csv('../input/train.csv', index_col='id', parse_dates=[1], 
                    true_values=['yes'], false_values=['no'])
test = pd.read_csv('../input/test.csv', index_col='id', parse_dates=[1], 
                   true_values=['yes'], false_values=['no'])
print(train.shape)
print(test.shape)


# In[3]:


test['price_doc'] = -1
data = pd.concat([train, test])

# Fix ecology column
ecology_map = {'poor': 1, 'satisfactory': 2, 'good': 3, 'excellent': 4, 'no data': np.NaN}
data['ecology'] = data['ecology'].apply(lambda x: ecology_map[x])

# There are 33 NaNs in the product_type column. 
# Set them to is_investment=True as that is the most common value.
data['is_investment'] = data['product_type'].apply(
    lambda x: False if x == 'OwnerOccupier' else True)
del data['product_type']

# Create a categorical value for each sub area
sub_areas = list(data['sub_area'].unique())
for area in sub_areas:
    data[area] = data['sub_area'].apply(lambda x: True if x == area else False)
del data['sub_area']

# Find columns with NaN values...
column_names = data.columns.values.tolist()
NaN_columns = []
for i, col_name in enumerate(column_names):
    s = sum(pd.isnull(data.iloc[:,i]))
    if s > 0:
        NaN_columns.append(i)
# ...and set most of these to the median value
for i in NaN_columns:
    if i in [2, 7, 8]: # life_sq, num_rooms, kitchen_sq
        continue
    else:
        data[column_names[i]]=data[column_names[i]].fillna(data[column_names[i]].median())

# Update NaN values for life_sq, num_room and kitch_sq
life_sq_to_full_sq = float(data['life_sq'].sum()) /     float(data.loc[data['life_sq'] > 0, 'full_sq'].sum())
average_room_size = float(data.loc[data['num_room'] > 0, 'full_sq'].sum()) /     float(data['num_room'].sum())
life_sq_to_kitch_sq = float(data['kitch_sq'].sum()) /     float(data.loc[data['kitch_sq'] > 0, 'full_sq'].sum())
data.loc[data['life_sq'].isnull(), 'life_sq'] =     data.loc[data['life_sq'].isnull(), 'full_sq'] * life_sq_to_full_sq
data.loc[data['num_room'].isnull(), 'num_room'] =     np.round(data.loc[data['num_room'].isnull(), 'full_sq'] / average_room_size)
data.loc[data['kitch_sq'].isnull(), 'kitch_sq'] =     data.loc[data['kitch_sq'].isnull(), 'full_sq'] * life_sq_to_kitch_sq

# Remove outliers from buildyear
median_build_year = data['build_year'].median()
data['build_year'] = data['build_year'].apply(     lambda x: median_build_year if x < 1800 else median_build_year if x > 2017 else x)

# Should output a 0 meaning that there are no NaNs left.
data.isnull().sum().sum()


# In[4]:


print("column name\tquantile 1\tquantile 99")
column_quantiles = {}
for c in data.columns.values[2:10]:
    column_quantiles[c] = (data[c].quantile(.0001), data[c].quantile(.9999))
    print(c, "\t", data[c].quantile(.0001), "\t", data[c].quantile(.9999))


# In[5]:


time_group = data.set_index('timestamp').groupby(pd.TimeGrouper(freq='M'))
plt.figure(figsize=(15,8))
(time_group['price_doc'].sum() / time_group['full_sq'].sum()).plot()
plt.title('Price per square meter for Moscow apartements', fontsize=16)
plt.xlabel('Time')
plt.ylabel('Price in RUB (starts at 60,000)')
plt.ylim([60000,180000])
plt.xlim(['2011-11-01', '2015-06-30'])
plt.grid()


# In[6]:


# Read macro data
# #! pattern found in some columns, treat as NaN
macro = pd.read_csv('../input/macro.csv', na_values='#!', parse_dates=[0])

# Fill in NaN values
macro.fillna(method='ffill', inplace=True)
macro.fillna(method='bfill', inplace=True)

# Remove thousand separator and convert to double
macro_column_names = macro.columns.values.tolist()
for i, col_name in enumerate(macro_column_names):
    if macro.ix[:,i].dtype == object:
        macro.ix[:,i] = macro.ix[:,i].str.replace(',','')
        macro.ix[:,i] = pd.to_numeric(macro.ix[:,i])


# In[7]:


square_meter_time_series = time_group['price_doc'].sum() / time_group['full_sq'].sum()

fig, ax1 = plt.subplots(figsize=(15,8))
ax1.plot(square_meter_time_series, 'b-')
ax1.set_xlabel('Time')
ax1.set_ylabel('Price in RUB', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim([60000, 180000])
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(macro.timestamp, macro.cpi, '-', color='r')
ax2.set_ylabel('CPI', color='r')
ax2.tick_params('y', colors='r')
ax2.set_xlim([pd.to_datetime('2011-11-01'), pd.to_datetime('2015-06-30')])
# Set the scale to make the lines approximately match at the beginning of the period
ax2.set_ylim([110, 600])

fig.tight_layout()
plt.title('Price per square meter compared to Consumer Price Index (CPI)', fontsize=16)
plt.show()


# In[8]:


plt.figure(figsize=(15,8))
plt.plot(time_group['price_doc'].quantile(.9), color='b')
plt.plot(time_group['price_doc'].quantile(.5), color='r')
plt.plot(time_group['price_doc'].quantile(.1), color='g')
plt.xlim(['2011-11-01', '2015-06-30'])
plt.title('Price for the apartment at the 10, 50 and 90 percentile', fontsize=16)
plt.ylabel('Price in RUB')
plt.xlabel('Time')
plt.grid()


# In[9]:


plt.figure(figsize=(15,8))
time_group.size().plot(kind='bar')
plt.title('Number of apartements for each month in dataset', fontsize=16)
plt.ylabel('Number of apartements')
plt.xlabel('Time')
plt.grid()


# In[10]:


# Create one entry per each year and month, 
# fill with mean value of each column over month
macro['YearMonth'] = macro['timestamp'].map(lambda x: 100*x.year + x.month)
year_month_group = macro.groupby(by='YearMonth')
macro_year_month = year_month_group.mean()

# Create a YearMonth attribute for the apartments as well
data['YearMonth'] = data['timestamp'].map(lambda x: 100*x.year + x.month)

# Now merge the data..
full_data = pd.merge(data, macro_year_month, how='left',                      left_on='YearMonth', right_index=True)
del full_data['timestamp']

# ..and split back into train/test set
last_train_row = train.shape[0]-1
train_proc = full_data.iloc[:last_train_row]
test_proc = full_data.iloc[last_train_row:]

# Move target price data into separate array
train_target_prices = train_proc['price_doc']
del train_proc['price_doc']
del test_proc['price_doc']


# In[11]:


# Set a random state for repeatability
random_state = 11

# Create a train/test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(     train_proc, train_target_prices, test_size=0.3, random_state=random_state)

# Create function for score metric, set any negatives to 0 to avoid math error
def rmsle(y_true, y_pred):
    negative_entries = y_pred[np.argwhere(np.isnan(np.log(y_pred+1)))]
    if (negative_entries):
        print(negative_entries)
        y_pred[y_pred < 0] = 0
    return np.sqrt(mean_squared_error(np.log(y_true+1), np.log(y_pred+1)))


# In[12]:


# RidgeCV
estimator = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 20.0, 100.0), 
                    fit_intercept=True, normalize=True, 
                    scoring='neg_mean_squared_error', cv=None, 
                    gcv_mode=None, store_cv_values=False)
estimator.fit(X_train, y_train)
print("Root mean square logarithmic error:", rmsle(y_test, estimator.predict(X_test)))
print("Best alpha", estimator.alpha_)


# In[13]:


print("\tActual price\t\tPredicted price")
for e in enumerate(zip(y_test[100:120], estimator.predict(X_test)[100:120])):
    print(e[0],"{:20,}".format(e[1][0]), "\t{:20,.0f}".format(e[1][1]))


# In[14]:


feature_weights = [x for x in zip(X_test.columns.values, estimator.coef_)]
feature_weights.sort(key=lambda x: x[1], reverse=True)
print("Strongest positive features")
for i in range(10):
    print(feature_weights[i])
print("\nStrongest negative features")
for i in range(10):
    print(feature_weights[-i-1])

