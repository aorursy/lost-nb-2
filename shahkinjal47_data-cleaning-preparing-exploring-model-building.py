#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt 
#import googlemaps
import random
from math import cos, asin, sqrt
import numpy as np
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

#gmaps = googlemaps.Client(key='AIzaSyAm_TEkXHyBqBAKJXca9N_JrZyy6K7dpoU') 
TYPE = ""
#function to load dataset into dataframe, reducing the datatypes from float64 and int64 to float32 and uint8
def load_ny_taxi_data(type = TYPE):
    if(type == "train"):
        data_types = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

        cols = list(data_types.keys())
        file = "../input/train.csv"
    if(type == "test"):
        data_types = {'key': 'str',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}
        cols = list(data_types.keys())
        file = "../input/test.csv"
    return pd.read_csv(file, usecols=cols, dtype=data_types) 
    
train = load_ny_taxi_data("train")

train = train.sample(n=1000000)


# In[ ]:



train.describe() #take a look at the data
train.isnull().sum() #check for the null values
print("Minimum fare = " , train["fare_amount"].min())
print("Maximum fare = " , train["fare_amount"].max())
print("Negative fare rows = " , sum(train["fare_amount"] < 0))
print("0 fare rows = " , sum(train["fare_amount"] == 0))


# In[ ]:


#Data Cleaning

# Currently, minimum taxi fare in NYC is $2.5. Since, we have data from 2009, assuming that minimum fare 
# would be atleast $1, removing rows with fare < $1.5
# print("Rows with fare<1 are ", sum(train["fare_amount"] < 1.5)) #Number of rows with fare < $1.5
# print("Rows with fare>450 are ", sum(train["fare_amount"] > 450)) #Number of rows with fare > $450

#Maximum fare between two farthest points in NY city is $320 as per the current uber fare. 
#So considering all the data with fare>$450 and fare<1.5 as bad data, we will remove those values
train = train[(train["fare_amount"] > 1.5) & (train["fare_amount"] <= 450)]

#NY City latitude is between 40.4965, 40.9159 and longitude is between -74.25 , -73.7016
#Identify latitudes and lngitudes which does not belong to NY City, adding some grace distance

# print(train[(train["pickup_latitude"] <39.8) | (train["pickup_latitude"] > 41.3)].shape[0])
# print(train[(train["pickup_longitude"] < -75) | (train["pickup_longitude"] > -71.8)].shape[0])
# print(train[(train["dropoff_latitude"] <39.8) | (train["dropoff_latitude"] > 41.3)].shape[0])
# print(train[(train["dropoff_longitude"] < -75) | (train["dropoff_longitude"] > -71.8)].shape[0])

#Remove rows with bad latitude and longitude values
train = train[(train["pickup_latitude"] >39.8) & (train["pickup_latitude"] < 41.3)]
train = train[(train["pickup_longitude"] > -75) & (train["pickup_longitude"] < -71.8)]
train = train[(train["dropoff_latitude"] >39.8) & (train["dropoff_latitude"] < 41.3)]
train = train[(train["dropoff_longitude"] > -75) & (train["dropoff_longitude"] < -71.8)]

#Removing 195532 rows with passenger count more than 6 and less than 1
# print(train[(train["passenger_count"] > 6 ) | (train["passenger_count"] < 1)].shape[0])

train = train[(train["passenger_count"] <= 6 ) & (train["passenger_count"] >= 1)]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
train.hist(bins=50,figsize=(15,15)) #Look at the histograms
plt.show()


# In[ ]:


#Visualizing data
# train_sample = train.sample(n=100000) #take a sample of 100,000 randon rows
train.plot(kind="scatter", x="pickup_longitude", y="pickup_latitude", c="red", alpha=0.1)
train.plot(kind="scatter", x="dropoff_longitude", y="dropoff_latitude", c="blue", alpha=0.1)


# In[ ]:


#Dual axis plot
# fig, ax1 = plt.subplots()

# ax2 = ax1.twinx()
# ax1.plot(kind="scatter", x="pickup_longitude", y="pickup_latitude", c="red", alpha=0.1)
# ax2.plot(kind="scatter", x="dropoff_longitude", y="dropoff_latitude", c="blue", alpha=0.1)
# plt.show()
# ax1.set_xlabel(train_sample["pickup_longitude"])
# ax1.set_ylabel(train_sample["pickup_latitude"], color='g')
# ax2.set_ylabel(train_sample["dropoff_latitude"], color='b')


# In[ ]:


#Look for correlations anmong variables
corr = train_sample.corr()
corr["fare_amount"].sort_values(ascending=False)
from pandas.tools.plotting import scatter_matrix
scatter_matrix(train_sample,figsize=(12,8))


# In[ ]:


#Calculate distance and duration for given pickup and drop locations using Google's Distance Matrix API
#Could not use it because of the limitations on the number of free API calls and processing speed for 
#such large dataset 

# <--- !!!!  DO NOT EXECUTE THIS STEP  !!!! --->

# def calc_dist_dur(df):

#     pickup = df.pickup_latitude.astype(str) + ',' + df.pickup_longitude.astype(str)
#     drop = df.dropoff_latitude.astype(str) + ',' + df.dropoff_longitude.astype(str)

#     distance = []
#     duration = []
#     for i in range(len(df)):
#         dist = gmaps.distance_matrix(origins = pickup[i],destinations = drop[i])
#         dist = dist['rows'][0]
#         dist = dist['elements'][0]
#         dur = dist['duration']['value']
#         dist = dist['distance']['value']
#         distance.append(dist)
#         duration.append(dur)
    
#     return distance, duration

# distance, duration = calc_dist_dur(train)


# In[ ]:


# Define function to calculate distance between two points by latitude and longitude
# Faced some errors in this function as described below
# 1. lat1, lon1 etc were Panda series and so it won't allow its conversion to float.Typecasted to list when calling the function
# 2.Python won't allow operations between two list, so converted list to np.array
# 3.Math.cos,asin etc functions does not support np.array. So used np.cos, np.asin functions

def distance(lat1, lon1, lat2, lon2):
    lat1 = np.array(lat1)
    lon1 = np.array(lon1)
    lat2 = np.array(lat2)
    lon2 = np.array(lon2)
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return (7917.512 * np.arcsin(np.sqrt(a)))

# Assigning dummy value 1.1 to column "distance" to change its datatype to float64 from object
train["distance"] = 1.1

#Changing datatype of train.Distance from float64 to float16
train["distance"] = train["distance"].astype(np.float16)

#Calculating distance between pickup and drop points using Haversine formula
train['distance'] = distance(train.pickup_latitude.tolist(), train.pickup_longitude.tolist(),
                             train.dropoff_latitude.tolist(), train.dropoff_longitude.tolist())


# In[ ]:


#Since we have distance now, let's see its relation with fare amount
# We can see a clear linear relationship between distance and fare. Also, there are some trips with zero
# distance by non-zero fare amount. We can also see some stright lines around fare 40-60, may be fixed fare
# to airports

train.plot(kind="scatter", x="distance", y="fare_amount", c="red", alpha=0.1)
plt.show()


# In[ ]:


# Create three new variables for storing year,time and hour of pickup
import datetime
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["year"] = train["pickup_datetime"].dt.year
train["time"] = train["pickup_datetime"].dt.time
train["hour"] = train["pickup_datetime"].dt.hour


# In[ ]:


# train["weekday"].value_counts()


# In[ ]:


# Define Function to polulate new column "weekday" where 1 = weekday, 0 = weekend
def weekday(pickup_date):
    weekday = []
    for index,val in pickup_date.iteritems():
        val = pd.to_datetime(val)
        if(val.weekday() == 5 or val.weekday() == 6):
            weekday.append(0)
        else:
            weekday.append(1)
    return weekday

train["weekday"] = 0
train["weekday"] = weekday(train["pickup_datetime"])


# In[ ]:


train["hour"] = train["pickup_datetime"].dt.hour
#Histogram on train.hour to see when highest number of cabs are booked
plt.hist(train["hour"], bins=5)
plt.show()


# In[ ]:


# Let's categorize time data into morning, afternoon, evening and night in new column "part_of_day"

def time_in_range(start, end, x):
    # Return true if x is in the range [start, end]
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end
    
def assign_day_part(pickup_date):
    day_part = []
    # Morning = 0600-1000
    mornStart = datetime.time(6, 0, 1)
    mornEnd = datetime.time(10, 0, 0)

    # Midday = 1000-1600
    midStart = datetime.time(10, 0, 1)
    midEnd = datetime.time(16, 0, 0)

    # Evening = 1600-2000
    eveStart = datetime.time(16, 0, 1)
    eveEnd = datetime.time(20, 0, 0)
    
    # Night = 2000-0000
    nightStart = datetime.time(20, 0, 1)
    nightEnd = datetime.time(0, 0, 0)

    # Late Night = 0000-0600
    lateStart = datetime.time(0, 0, 1)
    lateEnd = datetime.time(6, 0, 0)
    
    for index,val in pickup_date.iteritems():
        if time_in_range(mornStart, mornEnd, val.time()):
            day_part.append("morning")
        elif time_in_range(midStart, midEnd, val.time()):
            day_part.append("midday")
        elif time_in_range(eveStart, eveEnd, val.time()):
            day_part.append("evening")
        elif time_in_range(nightStart, nightEnd, val.time()):
            day_part.append("night")
        elif time_in_range(lateStart, lateEnd, val.time()):
            day_part.append("lateNight")

    return day_part

train["part_of_day"] = assign_day_part(train["pickup_datetime"])

# We do not need time and hour variables now. So, dropping them
train = train.drop(["time", "hour"], axis=1)


# In[ ]:


#Since we have distance and classification of weekday now, let's see the correlation

train.plot(kind="scatter", x="distance", y="fare_amount", c="red", alpha=0.1)


# In[ ]:


# REMOVE COMMENTS

# #Since we have distance and classification of weekday now, let's see the correlation
# # We see that distance is highly related to fare amount positively. Year is also positively 
# # corelated to fare amount

# print(train.corr()["fare_amount"])


# In[ ]:


# We can see negative correlation of fare amount with pickup and drop off latitude and 
# positive correlation with both longitudes. To explore that further, let's check if fare varies with 
# the direction of the trip

def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])

            
train['diff_lon'] = train.pickup_longitude - train.dropoff_longitude
train['diff_lat'] = train.pickup_latitude - train.dropoff_latitude

# Select trips in Manhattan
BB_manhattan = (-74.025, -73.925, 40.7, 40.8)
idx_manhattan = select_within_boundingbox(train, BB_manhattan)

plt.figure(figsize=(14,8))
plt.scatter(train[idx_manhattan].diff_lon, train[idx_manhattan].diff_lat, s=0.5, alpha=1.0, 
            c=np.log1p(train[idx_manhattan].fare_amount), cmap='viridis')
plt.colorbar()
plt.xlabel('pickup_longitude - dropoff_longitude')
plt.ylabel('pickup_latitude - dropoff_latidue')
plt.title('log1p(fare_amount)');


# In[ ]:


# We can clearly see that direction has positive linear correlation with fare amount
# So let's calculate exact direction of a trip, from 180 to -180 degrees. Horizontal axes = 0 degrees

def calculate_direction(d_lon, d_lat):
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    return result

train['direction'] = calculate_direction(train.diff_lon, train.diff_lat)


# In[ ]:



train.plot(kind="scatter", x="direction", y="fare_amount", c="blue", alpha=0.1)


# In[ ]:


# Convert categorical varibale "part_of_day" from to numerical value using sklearn LabelBinarizer
# This will create 5 new columns - evening, lateNight, midday, morning, night

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb_results = lb.fit_transform(train["part_of_day"])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

train = pd.merge(train, lb_results_df, left_index=True, right_index=True) #Merge output with training set

train.drop(["part_of_day"], axis=1, inplace=True) # Dropping part_of_day variable


# In[ ]:


#Let us separate the response variable fare_amount before we standardize the training set

train_labels = train["fare_amount"]
train.drop(["fare_amount"], axis=1, inplace=True)

# Let us also drop pickup_datetime variable as we have included variables like year and part_of_day
train_std = train.drop(["pickup_datetime"], axis=1)


# In[ ]:


# Feature Scaling - Let's standardize our attributes so that they are in the same scales

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
train_std = std_scaler.fit_transform(train_std)
train_std


# In[ ]:


# Create and train a Linear Regression Model
train_no_std = train.drop(["pickup_datetime"], axis=1)

train_prepared = train_no_std.copy()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_prepared, train_labels)


# In[ ]:


#Let us test our preductions on first 3 rows
some_data = train_prepared.iloc[:3]
some_labels = train_labels.iloc[:3]
print("Predictions:\t", lin_reg.predict(some_data))
print("Labels", list(some_labels))


# In[ ]:


# Measuring this linear regression model's RMSE on the whole training set

from sklearn.metrics import mean_squared_error
fare_predictions = lin_reg.predict(train_prepared)
lin_mse = mean_squared_error(train_labels,fare_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse # We got RMSE of 4.964 which is not very good as mean of fare_amount is $11. So
# prediction error of $5 is not acceptable


# In[ ]:


# Function to display model scores

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[ ]:


# Let's train a new DecisionTreeRegressor model and compare the rmse with earlier model

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_prepared, train_labels)
fare_predictions_DTR = tree_reg.predict(train_prepared)
tree_mse = mean_squared_error(train_labels,fare_predictions_DTR)
tree_rmse = np.sqrt(tree_mse)
tree_rmse # This gives RMSE of 0.011 which means this model is overfitting

# Let's use cross-validation for the better evaluation of the model

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, train_prepared, train_labels, 
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

#Take a look at the DecisionTreeRegressor's score from cross-validation
    
display_scores(rmse_scores)

We get mean of 5.6 with SD of 0.35. The variation of $5.6+-0.35 is not a very good prediction model


# In[ ]:


# Checking score for Linear Regression model created earlier using cross-validation
scores_lin_reg = cross_val_score(lin_reg, train_prepared, train_labels, 
                         scoring="neg_mean_squared_error", cv=10)

lin_reg_rmse_scores = np.sqrt(-scores_lin_reg)

display_scores(lin_reg_rmse_scores)

#We get mean of 4.96 with sd of 0.24 which is better then DecisionTreeRegressor


# In[ ]:


# Let's train a RandomForestRegressor and check how well it fits the data
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train_prepared, train_labels)
fare_predictions_rfg = forest_reg.predict(train_prepared)
rfg_mse = mean_squared_error(train_labels,fare_predictions_rfg)
rfg_rmse = np.sqrt(rfg_mse)
rfg_rmse # RMSE = 1.83 which is quite good compared to the previous two models

# Let's train this model using Cross-validation 

scores_rfg = cross_val_score(forest_reg, train_prepared, train_labels, 
                         scoring="neg_mean_squared_error", cv=10)

rfg_rmse_scores = np.sqrt(-scores_rfg)

display_scores(rfg_rmse_scores) # We got mean error of 4.31 with sd of 0.18, not very good but better
                                # than linear regression model


# In[ ]:


# Let us fine tune the RandomForestRegressor model

from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(train_prepared, train_labels)


# In[ ]:


print("Best parameters combination: ",grid_search.best_params_)
print("Best estimator: ",grid_search.best_estimator_)
cvres = grid_search.cv_results_
print("Evaluation scores: ")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# We get below results:
# Best parameters combination:  {'max_features': 4, 'n_estimators': 30}
# Best estimator:  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features=4, max_leaf_nodes=None, min_impurity_decrease=0.0,
#            min_impurity_split=None, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
#            verbose=0, warm_start=False)
# Evaluation scores: 
# 4.844166768045483 {'max_features': 2, 'n_estimators': 3}
# 4.384999498554626 {'max_features': 2, 'n_estimators': 10}
# 4.210352917368231 {'max_features': 2, 'n_estimators': 30}
# 4.744536239308798 {'max_features': 4, 'n_estimators': 3}
# 4.285080545281859 {'max_features': 4, 'n_estimators': 10}
# 4.116223038942821 {'max_features': 4, 'n_estimators': 30} <--- Best solution
# 4.686922101415325 {'max_features': 6, 'n_estimators': 3}
# 4.278610842446552 {'max_features': 6, 'n_estimators': 10}
# 4.139147097263857 {'max_features': 6, 'n_estimators': 30}
# 4.6503502986524 {'max_features': 8, 'n_estimators': 3}
# 4.270118164505793 {'max_features': 8, 'n_estimators': 10}
# 4.145219647426304 {'max_features': 8, 'n_estimators': 30}
# 4.776605716185325 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
# 4.3499492989752495 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
# 4.794243834697975 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
# 4.307106350189692 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
# 4.706772084360594 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
# 4.29509148366207 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}    


# In[ ]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[ ]:


# Check the relative importance of each attribute for making accurate predictions
attributes = list(train_prepared)
sorted(zip(feature_importances, attributes), reverse=True)

# Output
# [(0.3841124284168225, 'distance'),
#  (0.17239442282118714, 'diff_lon'),
#  (0.10769124244350045, 'dropoff_longitude'),
#  (0.09648335922154312, 'diff_lat'),
#  (0.07543540514386286, 'pickup_longitude'),
#  (0.05130816936350101, 'pickup_latitude'),
#  (0.045930285019949633, 'dropoff_latitude'),
#  (0.025204936787226056, 'direction'),
#  (0.022248352241158183, 'year'),
#  (0.004941632952827758, 'passenger_count'),
#  (0.0026084883883504065, 'midday'),
#  (0.0025747338852137552, 'weekday'),
#  (0.0024459892828226667, 'evening'),
#  (0.0024339192827895037, 'morning'),
#  (0.0023701585993039075, 'night'),
#  (0.0018164761499410416, 'lateNight')]


# In[ ]:


# Let us try dropping some less useful features as per the output of last command

# train_prepared.drop(["direction","year","passenger_count","midday","weekday","evening","morning",
#                      "night","lateNight"], axis=1, inplace=True)

forest_reg2 = RandomForestRegressor()
forest_reg2.fit(train_prepared, train_labels)
fare_predictions_rfg2 = forest_reg2.predict(train_prepared)
rfg2_mse = mean_squared_error(train_labels,fare_predictions_rfg2)
rfg2_rmse = np.sqrt(rfg2_mse)
rfg2_rmse

scores_rfg2 = cross_val_score(forest_reg2, train_prepared, train_labels, 
                         scoring="neg_mean_squared_error", cv=10)

rfg2_rmse_scores = np.sqrt(-scores_rfg2)

display_scores(rfg2_rmse_scores)

train_prepared = train_no_std.copy() # Adding removed features back

# We get below scores after removing some features which is bad compared to earlier mean error of 4.3
# So we will go back to adding the removed features

# Scores: [4.09936639 4.53226684 4.5280771  4.60757191 4.69303811 4.67236903
#  4.97136344 4.31146722 4.52617438 4.58548337]
# Mean: 4.55271777832126
# Standard deviation: 0.21881778766341559


# In[ ]:


# Let's test our module on the given test data

# Read the test data

test = load_ny_taxi_data("test")

#Perform transformations on the test data

test["distance"] = 1.1

#Changing datatype of train.Distance from float64 to float16
test["distance"] = test["distance"].astype(np.float16)

#Calculating distance between pickup and drop points using Haversine formula
test["distance"] = distance(test.pickup_latitude.tolist(), test.pickup_longitude.tolist(),
                             test.dropoff_latitude.tolist(), test.dropoff_longitude.tolist())

test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])
test["year"] = test["pickup_datetime"].dt.year
test["time"] = test["pickup_datetime"].dt.time
test["hour"] = test["pickup_datetime"].dt.hour

test["weekday"] = 0
test["weekday"] = weekday(test["pickup_datetime"])

test["part_of_day"] = assign_day_part(test["pickup_datetime"])

# We do not need time and hour variables now. So, dropping them
test = test.drop(["time", "hour"], axis=1)

test['diff_lon'] = test.pickup_longitude - test.dropoff_longitude
test['diff_lat'] = test.pickup_latitude - test.dropoff_latitude

#Add variable "direction"
test['direction'] = calculate_direction(test.diff_lon, test.diff_lat)

test.drop(["pickup_datetime"], axis=1, inplace=True)

lb_results = lb.fit_transform(test["part_of_day"])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

test = pd.merge(test, lb_results_df, left_index=True, right_index=True) #Merge output with testing set

test.drop(["part_of_day"], axis=1, inplace=True)

# Removing extra "key" column in test data and storing it for later merging
test_keys = test["key"]
test.drop(["key"], axis=1, inplace=True)

test_keys = test_keys.to_frame()


# In[ ]:


# Since training and test set have same variables now, we can run test data 
# through our model to predict the fare

test_fare_predictions = forest_reg.predict(test)


# In[ ]:


type(test_fare_predictions)


# In[ ]:


# Merge test data keys and predictions for kaggle submission

test_keys["fare_amount"] = 0

for i in range(len(test_keys)):
    test_keys.loc[i,"fare_amount"] = test_fare_predictions[i]
    
test_keys = test_keys.round(2)


# In[ ]:


# This dataframe test_keys contains our final submission data. 

test_keys.to_csv("../output/submission.csv", index=False)

