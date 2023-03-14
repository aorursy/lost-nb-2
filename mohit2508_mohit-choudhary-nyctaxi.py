#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd 
import math
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import feather as fe
import os
from sklearn.ensemble import RandomForestRegressor as rf
plt.style.use('seaborn-whitegrid')

df = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv',nrows=1000000,low_memory=True)
df.to_feather('nycTaxi.feather')


# In[ ]:


df = pd.read_feather('nycTaxi.feather')


# In[ ]:


#Passenger Count >0
df=df[df.passenger_count>0]

# dropping rows with any geo co-ordinate = 0
df=df[df.dropoff_latitude!=0 ]
df=df[df.pickup_longitude!=0 ]
df=df[df.pickup_latitude!=0 ]
df=df[df.dropoff_longitude!=0]

#dropping rows in which Taxi fare <=2.5. Base fare of a NYC Taxi is 2.5 dollars
df= df[df.fare_amount>2]
df= df[df.fare_amount<100]

# finding missing values in each column
missing_values = df.isnull().sum()

#dropoff_latitude has 9 missing values. Removing rows where any value is missing -- In this case the 9 rows 
#where dropoff_latitude is 0
df=df.dropna()

#Splitting Columns - Pick up Date and Pickup Time to Hour and Year

df['year'], df['hour'] = df['pickup_datetime'].str.split(' ', 1).str
df['hour'] = df.hour.str[0:2]
df['year'] = df.year.str[:4]
df=df.dropna()


# In[ ]:


df[['year','hour']] = df[['year','hour']].apply(pd.to_numeric)
print(df.head())


# In[ ]:


def select_within_newYork(df, loc):
    return (df.pickup_longitude >= loc[0]) & (df.pickup_longitude <= loc[1]) &            (df.pickup_latitude >= loc[2]) & (df.pickup_latitude <= loc[3]) &            (df.dropoff_longitude >= loc[0]) & (df.dropoff_longitude <= loc[1]) &            (df.dropoff_latitude >= loc[2]) & (df.dropoff_latitude <= loc[3])
NYC = (-74.5, -72.8, 40.5, 41.8)

df = df[select_within_newYork(df, NYC)]


# In[ ]:


# Eculeadean Distance Of the Journey
# This function is based on the Haversine model to calculate distance
# calculate-distance-between-two-latitude-longitude-points-haversine-formula 
# return distance in miles

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2-lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
    return 6367 * 2 * np.arcsin(np.sqrt(a)) *0.62137

df['distance'] = haversine_np(df.pickup_longitude, df.pickup_latitude,df.dropoff_longitude,
                 df.dropoff_latitude)
print(df.head())


# In[ ]:


print('Co-relation b/w Fare and Distance')
print(st.pearsonr(df.distance, df.fare_amount))
df=df[df.distance<=30]


# In[ ]:


# Visualisation
fig, axs = plt.subplots(1, 2, figsize=(16,6))
con = (df.distance < 30)  & (df.distance>0.5) & (df.fare_amount>0) & (df.fare_amount <200) 
axs[0].scatter( df[con].fare_amount,df[con].distance, alpha=0.3)
axs[0].set_xlabel('Fare')
axs[0].set_ylabel('Distance')
axs[0].set_title('Distance vs Fare')


# In[ ]:


# The below code is referenced from discusison forums

df['fare-bin'] = pd.cut(df['fare_amount'], bins = list(range(0, 50, 5))).astype(str)
df.loc[df['fare-bin'] == 'nan', 'fare-bin'] = '[45+]'
df.loc[df['fare-bin'] == '(5, 10]', 'fare-bin'] = '(05, 10]'
df.groupby('fare-bin')['distance'].mean().sort_index().plot.bar(color = 'b');
plt.title('Average Distance vs Fair ');
plt.ylabel('Avg. Distance');
plt.xlabel('Fare Range');


# In[ ]:


print('corelation b/w Distance and Time of Day')
print(st.pearsonr(df.distance, df.hour))


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(16,6))
con = (df.distance<30) & (df.fare_amount>0) 
axs[0].scatter(df[con].hour, df[con].distance, alpha=0.2)
axs[0].set_xlabel('Time Of Day (hours)')
axs[0].set_ylabel('Distance')
axs[0].set_title('Time of Day vs Distance')


# In[ ]:


df.groupby('hour')['distance'].mean().sort_index().plot.bar(color = 'b');
plt.title('Average Distance vs Time of Day');
plt.ylabel('Mean Distance');


# In[ ]:


print('corelation b/w Fare and Time of Day')
print(st.pearsonr(df.fare_amount, df.hour))


# In[ ]:


df.groupby('hour')['fare_amount'].mean().sort_index().plot.bar(color = 'r');
plt.title('Average Fare amount vs Time of Day');
plt.ylabel('Mean Fare');


# In[ ]:


times_sq = (-73.985130,40.758896)
def loc_time(loc, name, dist=0.5):
    # select all datapoints with dropoff location within range of airport
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
   
    a = (haversine_np(df.pickup_longitude, df.pickup_latitude, loc[0], loc[1]) < dist)
    df[a].hour.hist(bins=100, ax=axs[0])
    axs[0].set_xlabel('Time')
    axs[0].set_title('Pickups within {} mile of {}'.format(dist, name))

    b = (haversine_np(df.dropoff_longitude, df.dropoff_latitude, loc[0], loc[1]) < dist)
    df[b].hour.hist(bins=100, ax=axs[1])
    axs[1].set_xlabel('Time')
    axs[1].set_title('Dropoffs within {} mile of {}'.format(dist, name));
    
loc_time(times_sq, 'Times Square - Manhattan')


# In[ ]:


df.hist(column='hour',bins=100)


# In[ ]:


df['diff_long'] = (df.dropoff_longitude - df.pickup_longitude).abs()
df['diff_lat'] = (df.dropoff_latitude - df.pickup_latitude).abs()


# In[ ]:


test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv',low_memory=True)
test['diff_lat'] = (test.dropoff_latitude-test.pickup_latitude).abs()
test['diff_long'] = (test.dropoff_longitude-test.pickup_longitude).abs()
test['distance'] = haversine_np(test.pickup_longitude, test.pickup_latitude,test.dropoff_longitude,
                 test.dropoff_latitude)
test['year'], test['hour'] = test['pickup_datetime'].str.split(' ', 1).str
test['hour'] = test.hour.str[0:2]
test['year'] = test.year.str[:4]
test[['year','hour']] = test[['year','hour']].apply(pd.to_numeric)
test_id = list(test.pop('key'))


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
p=lr.fit(df[['diff_lat','diff_long','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','distance','passenger_count']], df['fare_amount'])

print('Intercept', round(lr.intercept_, 4))
print('Lat diff coef: ', round(lr.coef_[0], 4), 
      '\tLong diff coef:', round(lr.coef_[1], 4),
      '\t Pikcup Latitude  coef', round(lr.coef_[2],4),
      '\t Pikcup Longitude  coef', round(lr.coef_[3],4),
      '\t Dropoff Latitude  coef', round(lr.coef_[4],4),
      '\t Dropoff Longitude  coef', round(lr.coef_[5],4),
      '\tDistance coef:', round(lr.coef_[6], 4))


# In[ ]:


preds_lr = lr.predict(test[['diff_lat', 'diff_long','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','distance','passenger_count']])
sub = pd.DataFrame({'key': test_id, 'fare_amount': preds_lr})
sub.to_csv('output_lr.csv', index = False)


# In[ ]:


Finidng RMSE


# In[ ]:


df.info()
from sklearn import metrics
from sklearn.model_selection import train_test_split
X = df.drop(['key','fare_amount','pickup_datetime', 'fare-bin'],1)
#X=df.drop('key',1)
X_train, X_test, y_train, y_test = train_test_split(X,df['fare_amount'], test_size=0.2)
lr.fit(X_train,y_train)
y_pred = lr.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, df['fare_amount']))
print (lrmse)


# In[ ]:


random_forest = rf(n_estimators = 10, max_depth = 10, max_features = None, oob_score = True, bootstrap = True, verbose = 1, n_jobs = -1)
random_forest.fit(df[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','distance', 'diff_lat', 'diff_long', 'passenger_count']],df['fare_amount'])


# In[ ]:


predictedFare = random_forest.predict(test[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','distance', 'diff_lat', 'diff_long', 'passenger_count']])
sub = pd.DataFrame({'key': test_id, 'fare_amount': predictedFare})
sub.to_csv('output_rf.csv', index = False)

