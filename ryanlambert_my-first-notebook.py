#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Packages

import numpy as np
import pandas as pd
import os
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# In[2]:


#Set up the paths of the files

train_path = '../input/new-york-city-taxi-fare-prediction/train.csv'
test_path = '../input/new-york-city-taxi-fare-prediction/test.csv'


# In[3]:


#Input the Training Data and Evaluate. Only going to take 1M rows out of the 55M to reduce processing time

train_data = pd.read_csv(train_path, nrows=5000000)
train_data.head()


# In[4]:


#Convert Pickup Datetime to a proper Datetime format

train_data['pickup_datetime'] =  pd.to_datetime(train_data['pickup_datetime'],format='%Y-%m-%d %H:%M:%S %Z')
train_data.head()


# In[5]:


#Build out date values into separate columns

train_data['pickup_year'] = train_data['pickup_datetime'].dt.year
train_data['pickup_quarter'] = train_data['pickup_datetime'].dt.quarter
train_data['pickup_month'] = train_data['pickup_datetime'].dt.month
train_data['pickup_day'] = train_data['pickup_datetime'].dt.day
train_data['pickup_hour'] = train_data['pickup_datetime'].dt.hour
train_data.head()


# In[6]:


#Drop columns we don't require

train_data.drop(['key', 'pickup_datetime'], axis=1, inplace=True)
train_data.head()


# In[7]:


#Create Cyclical Date Values

train_data['pickup_month_cos']=np.cos((train_data['pickup_month']-1)*(2*(np.pi/12)))
train_data['pickup_month_sin']=np.sin((train_data['pickup_month']-1)*(2*(np.pi/12)))
train_data['pickup_day_cos']=np.cos((train_data['pickup_day']-1)*(2*(np.pi/30)))
train_data['pickup_day_sin']=np.sin((train_data['pickup_day']-1)*(2*(np.pi/30)))
train_data['pickup_quarter_cos']=np.cos((train_data['pickup_quarter']-1)*(2*(np.pi/4)))
train_data['pickup_quarter_sin']=np.sin((train_data['pickup_quarter']-1)*(2*(np.pi/4)))
train_data['pickup_hour_cos']=np.cos((train_data['pickup_hour']-1)*(2*(np.pi/24)))
train_data['pickup_hour_sin']=np.sin((train_data['pickup_hour']-1)*(2*(np.pi/24)))
train_data.head()


# In[8]:


#Convert Year into Number of Years Historically from 2020.

train_data['pickup_year_age']=2020-train_data['pickup_year']
train_data.head()


# In[9]:


#Drop existing pickup date fields and use just cyclical ones going forward

train_data.drop(['pickup_year','pickup_quarter','pickup_month','pickup_day','pickup_hour'], axis=1, inplace=True)
train_data.head()


# In[10]:


#Now that most of the data is ready for modelling, let's confirm that the fare amount is distributed okay

train_data.hist(bins=10,column='fare_amount',figsize=(15,6))

#Will not scale this as it looks quite good as-is already. However there are negative values we must remove those


# In[11]:


#Filter based on analysis re-fares

train_data.dropna(inplace=True) #Modelling resulted in NaN errors, dropping these
train_data = train_data[(train_data.fare_amount > 0)]


# In[12]:


#Evaluate the Passenger Count

train_data.hist(column='passenger_count',figsize=(15,6))

#It looks like this is having values over 10 people, not sure how that is possible for a taxi, will remove


# In[13]:


#Ensure no negative passengers or where there are more than 10 passengers as most limos would have a 10 person maximum

train_data = train_data[(train_data.passenger_count > 0) & (train_data.passenger_count < 10)]


# In[14]:


#Next step is to build a distance function between the start and end points. Will use the Haversine distance calculation

def distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    radius = 6371
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude = map(np.radians,[pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude])
    distance_latitude = dropoff_latitude - pickup_latitude
    distance_longitude = dropoff_longitude - pickup_longitude
    calculation = np.sin(distance_latitude/2.0)**2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(distance_longitude/2.0)**2
    
    return 2 * radius * np.arcsin(np.sqrt(calculation))

train_data['distance'] = distance(train_data['pickup_latitude'], train_data['pickup_longitude'], train_data['dropoff_latitude'], train_data['dropoff_longitude'])


# In[15]:


# Workbook from https://www.kaggle.com/gunbl4d3/xgboost-ing-taxi-fares mentioned to filter out inappropriate locations outside of the range in NYC
train_data = train_data[(train_data.pickup_longitude > -80) & (train_data.pickup_longitude < -70) & (train_data.pickup_latitude > 35) & (train_data.pickup_latitude < 45) & (train_data.dropoff_longitude > -80) & (train_data.dropoff_longitude < -70) &
        (train_data.dropoff_latitude > 35) & (train_data.dropoff_latitude < 45)]


# In[16]:


#Split training data into training records and validation records. We cannot use test set as we do not know the outcome

X_train, X_val, y_train, y_val = train_test_split(train_data.iloc[:, 1:], train_data['fare_amount'], test_size=0.2, random_state=42)


# In[17]:


#Fit Model(s) here and create model shells/parameters

#linear_regression = linear_model.LinearRegression()
#ridge_regression = linear_model.Ridge(alpha=0.5)
#lasso_regression = linear_model.Lasso(alpha=0.1)
#random_forest = RandomForestRegressor(max_depth=4,n_estimators=250, random_state=0)
#gradient_boost = GradientBoostingRegressor()
xg_boost = XGBRegressor(objective='reg:squarederror')
#light_gbm = LGBMRegressor()


#linear_regression.fit(X_train,y_train)
#ridge_regression.fit(X_train,y_train)
#lasso_regression.fit(X_train,y_train)
#random_forest.fit(X_train,y_train)
#gradient_boost.fit(X_train,y_train)
xg_boost.fit(X_train,y_train)
#light_gbm.fit(X_train,y_train)


# In[18]:


#Predict validation result and measure it against actual

#y_pred_linear_regression = linear_regression.predict(X_val)
#y_pred_ridge_regression = ridge_regression.predict(X_val)
#y_pred_lasso_regression = lasso_regression.predict(X_val)
#y_pred_random_forest = random_forest.predict(X_val)
#y_pred_gradient_boost = gradient_boost.predict(X_val)
y_pred_xg_boost = xg_boost.predict(X_val)
#y_pred_light_gbm = light_gbm.predict(X_val)

#print('Linear Regression - Root Mean Squared Error: %.2f'
#      % math.sqrt(mean_squared_error(y_val, y_pred_linear_regression)))
#print('Ridge Regression - Root Mean Squared Error: %.2f'
#      % math.sqrt(mean_squared_error(y_val, y_pred_ridge_regression)))
#print('Lasso Regression - Root Mean Squared Error: %.2f'
#      % math.sqrt(mean_squared_error(y_val, y_pred_lasso_regression)))
#print('Random Forest - Root Mean Squared Error: %.2f'
#      % math.sqrt(mean_squared_error(y_val, y_pred_random_forest)))
#print('Gradient Boost - Root Mean Squared Error: %.2f'
#      % math.sqrt(mean_squared_error(y_val, y_pred_gradient_boost)))
print('XG Boost - Root Mean Squared Error: %.2f'
      % math.sqrt(mean_squared_error(y_val, y_pred_xg_boost)))
#print('Light GBM - Root Mean Squared Error: %.2f'
#      % math.sqrt(mean_squared_error(y_val, y_pred_light_gbm)))

#Result on 100k record sample shows XG Boost is the strongest performer by a good margin
#Result on 1M record sample shows XG Boost is still the winner and the RMSE is very similar to the 100k sample.


# In[19]:


#Bring in Test Data now and transform it appropriately

test_data = pd.read_csv(test_path)
test_data['pickup_datetime'] =  pd.to_datetime(test_data['pickup_datetime'],format='%Y-%m-%d %H:%M:%S %Z')
test_data['pickup_year'] = test_data['pickup_datetime'].dt.year
test_data['pickup_quarter'] = test_data['pickup_datetime'].dt.quarter
test_data['pickup_month'] = test_data['pickup_datetime'].dt.month
test_data['pickup_day'] = test_data['pickup_datetime'].dt.day
test_data['pickup_hour'] = test_data['pickup_datetime'].dt.hour
test_data['pickup_month_cos']=np.cos((test_data['pickup_month']-1)*(2*(np.pi/12)))
test_data['pickup_month_sin']=np.sin((test_data['pickup_month']-1)*(2*(np.pi/12)))
test_data['pickup_day_cos']=np.cos((test_data['pickup_day']-1)*(2*(np.pi/30)))
test_data['pickup_day_sin']=np.sin((test_data['pickup_day']-1)*(2*(np.pi/30)))
test_data['pickup_quarter_cos']=np.cos((test_data['pickup_quarter']-1)*(2*(np.pi/4)))
test_data['pickup_quarter_sin']=np.sin((test_data['pickup_quarter']-1)*(2*(np.pi/4)))
test_data['pickup_hour_cos']=np.cos((test_data['pickup_hour']-1)*(2*(np.pi/24)))
test_data['pickup_hour_sin']=np.sin((test_data['pickup_hour']-1)*(2*(np.pi/24)))
test_data['pickup_year_age']=2020-test_data['pickup_year']
test_data.drop(['pickup_year','pickup_quarter','pickup_month','pickup_day','pickup_hour','pickup_datetime'], axis=1, inplace=True)
test_data['distance'] = distance(test_data['pickup_latitude'], test_data['pickup_longitude'], test_data['dropoff_latitude'], test_data['dropoff_longitude'])
test_data.head()


# In[20]:


#Make predictions on Test Set

y_predictions = xg_boost.predict(test_data.iloc[:, 1:])


# In[21]:


#Send Test Set predictions to file for upload

submission = pd.DataFrame({'key': test_data['key'], 'fare_amount': y_predictions},columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

