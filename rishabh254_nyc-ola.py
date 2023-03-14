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


data =  pd.read_csv('../input/train.csv', nrows = 15_000_000)


# In[ ]:


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(data)


# In[ ]:


print('Old size: %d' % len(data))
data = data.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(data))


# In[ ]:


from datetime import datetime
data['datetime_object'] = [datetime.strptime(date,'%Y-%m-%d %H:%M:%S %Z') for date in data['pickup_datetime']]


# In[ ]:


#print(data.describe())
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...

# add new column to dataframe with distance in miles
data['distance_miles'] = distance(data.pickup_latitude, data.pickup_longitude,                                       data.dropoff_latitude, data.dropoff_longitude)

(data.head())


# In[ ]:


print('Old size: %d' % len(data))
data = data[(data.abs_diff_longitude < 3.0) & (data.abs_diff_latitude < 3.0)]
data = data[(data.pickup_longitude >= -74.3) & (data.pickup_longitude <= -72.9)]  # nyc coordinates
data = data[(data.dropoff_longitude >= -74.3) & (data.dropoff_longitude <= -72.9)]
data = data[(data.pickup_latitude >= 40.5) & (data.pickup_latitude <= 41.8)]
data = data[(data.dropoff_latitude >= 40.5) & (data.dropoff_latitude <= 41.8)]
data = data[(data.fare_amount>=2) & (data.fare_amount<=500)]
data = data[(data.passenger_count>0) & (data.passenger_count <=6)]
data = data[(data.distance_miles<=100.0) & (data.distance_miles>0.05)]
nyc = (-74.0063889, 40.7141667)
data['distance_to_center'] = distance(nyc[1], nyc[0],data.dropoff_latitude, data.dropoff_longitude)
data = data[data.distance_to_center<15.0]
print('New size: %d' % len(data))


# In[ ]:


def late_night (row):
    if (row['hour'] <= 6) or (row['hour'] >= 20):
        return 1
    else:
        return 0


def night (row):
    if ((row['hour'] <= 20) and (row['hour'] >= 16)) and (row['weekday'] < 5):
        return 1
    else:
        return 0
    


#data.distance_miles.hist(bins=50, figsize=(12,4))
#plt.xlabel('distance miles')
#plt.title('Histogram ride distances in miles')
#data.groupby('passenger_count')['distance_miles', 'fare_amount'].mean()
#print("Average $USD/Mile : {:0.2f}".format(data.fare_amount.sum()/data.distance_miles.sum()))
#data['fare_per_mile'] = data.fare_amount / data.distance_miles
data['hour'] = [date.hour for date in data['datetime_object']]
data['year'] = [date.year for date in data['datetime_object']]
data['day'] = [date.day for date in data['datetime_object']]
data['weekday'] = data['datetime_object'].apply(lambda x: x.weekday())
data['night'] = data.apply (lambda x: night(x), axis=1)
data['late_night'] = data.apply (lambda x: late_night(x), axis=1)   
# There is a $1 surcharge from 4pm to 8pm on weekdays, excluding holidays.


# In[ ]:


rangeA = 1.5
rangeN = 20.0
rangeS = 48.7
rangeR = 14.1
rangeD = 28.7
rangeO = 29
rangeP = 15.7
rangeW = 20.0
jfk = (-73.7822222222, 40.6441666667) #JFK Airport
ewr = (-74.175, 40.69) # Newark Liberty International Airport
lgr = (-73.87, 40.77) # LaGuardia Airport

 # county
Nassau = (-73.5594, 40.6546)
Suffolk = (-72.6151, 40.9849)
Westchester = (-73.7949, 41.1220)
Rockland = (-73.9830, 41.1489)
Dutchess = (-73.7478, 41.7784)
Orange = (-74.3118, 41.3912)
Putnam = (-73.7949, 41.4351) 

data_air=data

def add_checkpoint(point, point_name,rangeA):
    data_air[point_name] = (distance(data.pickup_latitude, data.pickup_longitude, point[1], point[0]) <= rangeA) | ((distance(data.dropoff_latitude, data.dropoff_longitude, point[1], point[0]) <= rangeA))
    data_air[point_name].replace(False, 0, inplace=True)
    data_air[point_name] = data_air[point_name].astype(int)

add_checkpoint(jfk, 'jfk',rangeA)
add_checkpoint(ewr, 'ewr',rangeA)
add_checkpoint(lgr, 'lgr',rangeA)
add_checkpoint(Nassau, 'Nassau',rangeN)
add_checkpoint(Suffolk, 'Suffolk',rangeS)
add_checkpoint(Westchester, 'Westchester',rangeW)
add_checkpoint(Rockland, 'Rockland',rangeR )
add_checkpoint(Dutchess, 'Dutchess',rangeD)
add_checkpoint(Orange, 'Orange',rangeO)
add_checkpoint(Putnam, 'Putnam',rangeP)

data_air = data[(data_air.jfk | data_air.ewr | data_air.lgr | data_air.Nassau | data_air.Suffolk | data_air.Westchester | data_air.Rockland | data_air.Dutchess | data_air.Orange | data_air.Putnam)==1]
data_air['airport'] = (data_air.jfk | data_air.ewr | data_air.lgr )==1
data_air['airport'].replace(False, 0, inplace=True)
data_air['airport'] = data_air['airport'].astype(int)
data_air['county1'] = (data_air.jfk | data_air.ewr | data_air.lgr )==0
data_air['county1'] = (data_air.Nassau | data_air.Westchester)==1
data_air['county1'].replace(False, 0, inplace=True)
data_air['county1'] = data_air['county1'].astype(int)
data_air['county2'] = (data_air.jfk | data_air.ewr | data_air.lgr | data_air.Nassau | data_air.Westchester)==0
data_air['county2'].replace(False, 0, inplace=True)
data_air['county2'] = data_air['county2'].astype(int)
data = data[(data.jfk | data.ewr | data.lgr | data.Nassau | data.Suffolk | data.Westchester | data.Rockland | data.Dutchess | data.Orange | data.Putnam)==0]
data_air.describe()


# In[ ]:


'''import seaborn as sns
import matplotlib.pyplot as plt
corrmat = data_air.corr()
f, ax = plt.subplots(figsize=(12, 9))

k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'fare_amount')['fare_amount'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()'''


# In[ ]:




#data.year_2015.hist(bins=50, figsize=(12,4))
#plt.xlabel('distance miles')
#plt.title('Histogram ride hour')
plt.scatter(data['year'][:1000], data['fare_amount'][:1000])
plt.show()
data.head()


# In[ ]:


# Drop unwanted columns



dropped_columns_air = ['day','pickup_longitude','pickup_latitude','dropoff_latitude','dropoff_longitude',
                       'distance_to_center','passenger_count','Nassau','Westchester',
                  'datetime_object','abs_diff_longitude','abs_diff_latitude','key','pickup_datetime']

dropped_columns = ['day','pickup_longitude','pickup_latitude','dropoff_latitude','dropoff_longitude','distance_to_center',
                  'datetime_object','abs_diff_longitude','abs_diff_latitude','key','pickup_datetime',
                  'jfk','ewr','lgr', 'Nassau','Suffolk','Westchester','Rockland','Dutchess','Orange','Putnam'
                  ]
train_clean = data.drop(dropped_columns, axis=1)
train_air_clean = data_air.drop(dropped_columns_air, axis=1)
train_air_clean.head()
data_air.head()
#train_clean.head()
#test_clean = test.drop(dropped_columns + ['key', 'passenger_count'], axis=1)


# In[ ]:


# split data in train and validation (90% ~ 10%)
from sklearn.model_selection import train_test_split
train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

# Get labels
train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)


# In[ ]:


# split data in train and validation (90% ~ 10%)
train_air_df, validation_air_df = train_test_split(train_air_clean, test_size=0.10, random_state=1)

# Get labels
train_air_labels = train_air_df['fare_amount'].values
validation_air_labels = validation_air_df['fare_amount'].values
train_air_df = train_air_df.drop(['fare_amount'], axis=1)
validation_air_df = validation_air_df.drop(['fare_amount'], axis=1)


# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df['distance_miles'] = distance(test_df.pickup_latitude, test_df.pickup_longitude,                                       test_df.dropoff_latitude, test_df.dropoff_longitude)
test_df['datetime_object'] = [datetime.strptime(date,'%Y-%m-%d %H:%M:%S %Z') for date in test_df['pickup_datetime']]
test_df['hour'] = [date.hour for date in test_df['datetime_object']]
test_df['year'] = [date.year for date in test_df['datetime_object']]
test_df['day'] = [date.day for date in test_df['datetime_object']]
test_df['weekday'] = test_df['datetime_object'].apply(lambda x: x.weekday())
test_df['night'] = test_df.apply (lambda x: night(x), axis=1)
test_df['late_night'] = test_df.apply (lambda x: late_night(x), axis=1)

#test_df['distance_to_center'] = distance(nyc[1], nyc[0],test_df.dropoff_latitude, test_df.dropoff_longitude)

def add_checkpoint_test(point, point_name,rangeA):
    test_df[point_name] = (distance(test_df.pickup_latitude, test_df.pickup_longitude, point[1], point[0]) <= rangeA) | ((distance(test_df.dropoff_latitude, test_df.dropoff_longitude, point[1], point[0]) <= rangeA))
    test_df[point_name].replace(False, 0, inplace=True)
    test_df[point_name] = test_df[point_name].astype(int)

add_checkpoint_test(jfk, 'jfk',rangeA)
add_checkpoint_test(ewr, 'ewr',rangeA)
add_checkpoint_test(lgr, 'lgr',rangeA)
add_checkpoint_test(Nassau, 'Nassau',rangeN)
add_checkpoint_test(Suffolk, 'Suffolk',rangeS)
add_checkpoint_test(Westchester, 'Westchester',rangeW)
add_checkpoint_test(Rockland, 'Rockland',rangeR)
add_checkpoint_test(Dutchess, 'Dutchess',rangeD)
add_checkpoint_test(Orange, 'Orange',rangeO)
add_checkpoint_test(Putnam, 'Putnam',rangeP)



#test_df['euclidean'] = minkowski_distance(test_df['pickup_longitude'], test_df['dropoff_longitude'],
#                                       test_df['pickup_latitude'], test_df['dropoff_latitude'], 2)

test_air_df = test_df[(test_df.jfk | test_df.ewr | test_df.lgr | test_df.Nassau | test_df.Suffolk | test_df.Westchester | test_df.Rockland | test_df.Dutchess | test_df.Orange | test_df.Putnam)==1]
test_df = test_df[(test_df.jfk | test_df.ewr | test_df.lgr | test_df.Nassau | test_df.Suffolk | test_df.Westchester | test_df.Rockland | test_df.Dutchess | test_df.Orange | test_df.Putnam)==0]

dropped_columns_test = ['pickup_longitude', 'pickup_latitude', 'day','key',
                        'jfk','ewr','lgr','Nassau','Suffolk','Westchester','Rockland','Dutchess','Orange','Putnam',
                   'dropoff_longitude', 'dropoff_latitude' ,'datetime_object','pickup_datetime'
                  ]
test_clean = test_df.drop(dropped_columns_test, axis=1)
test_clean.head()

test_air_df['airport'] = (test_air_df.jfk | test_air_df.ewr | test_air_df.lgr )==1
test_air_df['airport'].replace(False, 0, inplace=True)
test_air_df['airport'] = test_air_df['airport'].astype(int)
test_air_df['county1'] = (test_air_df.jfk | test_air_df.ewr | test_air_df.lgr )==0 
test_air_df['county1'] = (test_air_df.Nassau | test_air_df.Westchester)==1
test_air_df['county1'].replace(False, 0, inplace=True)
test_air_df['county1'] = test_air_df['county1'].astype(int)
test_air_df['county2'] = (test_air_df.jfk | test_air_df.ewr | test_air_df.lgr | test_air_df.Nassau | test_air_df.Westchester)==0
test_air_df['county2'].replace(False, 0, inplace=True)
test_air_df['county2'] = test_air_df['county2'].astype(int)


dropped_columns_test_air = ['pickup_longitude', 'pickup_latitude', 'day','key','passenger_count','Nassau','Westchester',
                   'dropoff_longitude', 'dropoff_latitude' ,'datetime_object','pickup_datetime'
                  ]
test_air_clean = test_air_df.drop(dropped_columns_test_air, axis=1)
test_air_clean.describe()


# In[ ]:



import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU
from keras import optimizers
from keras import regularizers
# Scale data
# Note: im doing this here with sklearn scaler but, on the Coursera code the scaling is done with Dataflow and Tensorflow
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_clean)

train_air_df_scaled = scaler.fit_transform(train_air_df)
validation_air_df_scaled = scaler.transform(validation_air_df)
test_air_scaled = scaler.transform(test_air_clean)


# In[ ]:


BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.0001
DATASET_SIZE = 6000000

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=train_df_scaled.shape[1], activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer=adam, metrics=['mae'])
print(train_labels)


# In[ ]:


BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.0001
DATASET_SIZE = 6000000

model_air = Sequential()
model_air.add(Dense(256, activation='relu', input_dim=train_air_df_scaled.shape[1], activity_regularizer=regularizers.l1(0.01)))
model_air.add(BatchNormalization())
model_air.add(Dense(128, activation='relu'))
model_air.add(BatchNormalization())
model_air.add(Dense(64, activation='relu'))
model_air.add(BatchNormalization())
model_air.add(Dense(32, activation='relu'))
model_air.add(BatchNormalization())
model_air.add(Dense(16, activation='relu'))
model_air.add(BatchNormalization())
model_air.add(Dense(1))

adam = optimizers.adam(lr=LEARNING_RATE)
model_air.compile(loss='mse', optimizer=adam, metrics=['mae'])
print(train_air_labels)


# In[ ]:


print('Dataset size: %s' % DATASET_SIZE)
print('Epochs: %s' % EPOCHS)
print('Learning rate: %s' % LEARNING_RATE)
print('Batch size: %s' % BATCH_SIZE)
print('Input dimension: %s' % train_df_scaled.shape[1])
print('Features used: %s' % train_df.columns)
model.summary()
history = model.fit(x=train_df_scaled, y=train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                    verbose=1, validation_data=(validation_df_scaled, validation_labels), 
                    shuffle=True)


# In[ ]:


print('Dataset size: %s' % DATASET_SIZE)
print('Epochs: %s' % EPOCHS)
print('Learning rate: %s' % LEARNING_RATE)
print('Batch size: %s' % BATCH_SIZE)
print('Input dimension: %s' % train_air_df_scaled.shape[1])
print('Features used: %s' % train_air_df.columns)
model_air.summary()
history_air = model_air.fit(x=train_air_df_scaled, y=train_air_labels, batch_size=BATCH_SIZE, epochs=EPOCHS*2, 
                    verbose=1, validation_data=(validation_air_df_scaled, validation_air_labels), 
                    shuffle=True)


# In[ ]:


#plot_loss_accuracy(history)

SUBMISSION_NAME = 'submission.csv'
def output_submission(raw_test,prediction,  file_name):
    df = pd.DataFrame(prediction, columns=['fare_amount'])
    df['key'] = raw_test['key']
                  
    #raw_test = raw_test.drop(dropped_columns, axis=1)
    df[['key','fare_amount']].to_csv((file_name), index=False)
    
    #print(df)
    print('Output complete')
    print(df)
    
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
prediction_air = model_air.predict(test_air_scaled, batch_size=128, verbose=1)
#prediction_air = model_air.predict(test_air_scaled, num_iteration = model_air.best_iteration)
#print(prediction_air)
frames = [test_df, test_air_df]
test_final = pd.concat(frames)
frames = [prediction, prediction_air]
prediction_final = np.concatenate(frames)


#test_df = pd.read_csv('../input/test.csv')
result=dict()
print(len(test_final))
i=0
#print(test_df[0])
for index, row in test_final.iterrows():
    result[row['key']]=prediction_final[i]
    i=i+1
test_df1 = pd.read_csv('../input/test.csv')

pred=[]
for index, row in test_df1.iterrows():
    pred.append(result[row['key']])
    
test_df1.head()
output_submission(test_df1,pred, SUBMISSION_NAME)


#from sklearn.model_selection import train_test_split
#y = data.fare_amount
#X = data.drop('fare_amount', axis=1)
#train_df, val_df, train_y, val_y = train_test_split(X, y,test_size=0.2)
#train_df.dtypes


# In[ ]:


# Construct and return an Nx3 input matrix for our linear model
# using the travel vector, plus a 1.0 for a constant bias term.
def get_input_matrix(df):
    return np.column_stack((df.distance_miles, df.passenger_count,df.hour,df.year, np.ones(len(df))))

#train_X = get_input_matrix(train_df)
#train_y = np.array(train_df['fare_amount'])

#print(train_X.shape)
#print(train_y.shape)


# In[ ]:


# The lstsq function returns several things, and we only care about the actual weight vector w.
#(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)
#print(w)


# In[ ]:


#w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y)
#print(w_OLS)


# In[ ]:


#test_df = pd.read_csv('../input/test.csv')
#test_df['distance_miles'] = distance(test_df.pickup_latitude, test_df.pickup_longitude, \
                                      test_df.dropoff_latitude, test_df.dropoff_longitude)
#test_df['datetime_object'] = [datetime.strptime(date,'%Y-%m-%d %H:%M:%S %Z') for date in test_df['pickup_datetime']]
#test_df['hour'] = [date.hour for date in test_df['datetime_object']]
#test_df['year'] = [date.year for date in test_df['datetime_object']]

#val_df['distance_miles'] = distance(val_df.pickup_latitude, val_df.pickup_longitude, \
                                      val_df.dropoff_latitude, val_df.dropoff_longitude)
#val_df['datetime_object'] = [datetime.strptime(date,'%Y-%m-%d %H:%M:%S %Z') for date in val_df['pickup_datetime']]
#val_df['hour'] = [date.hour for date in val_df['datetime_object']]
#val_df['year'] = [date.year for date in val_df['datetime_object']]
test_df.dtypes
#val_df = pd.read_csv('../input/train.csv', nrows = 10000000)
#val_df.dtypes


# In[ ]:


# Reuse the above helper functions to add our features and generate the input matrix.
#add_travel_vector_features(test_df)
#test_X = get_input_matrix(test_df)
#add_travel_vector_features(val_df)
#val_df = val_df.dropna(how = 'any', axis = 'rows')
#val_df = val_df[(val_df.abs_diff_longitude < 5.0) & (val_df.abs_diff_latitude < 5.0)]
#val_X = get_input_matrix(val_df)
# Predict fare_amount on the test set using our model (w) trained on the training set.
#test_y_predictions = np.matmul(test_X, w).round(decimals = 2)
#val_y_predictions = np.matmul(val_X, w).round(decimals = 2)
#val_y = np.array(val_df['fare_amount'])

#from sklearn.metrics import mean_squared_error
#print(np.sqrt(mean_squared_error(val_y, val_y_predictions)))
# Write the predictions to a CSV file which we can submit to the competition.
#submission = pd.DataFrame(
#    {'key': test_df.key, 'fare_amount': test_y_predictions},
#    columns = ['key', 'fare_amount'])
#submission.to_csv('submission.csv', index = False)

#print(os.listdir('.'))

