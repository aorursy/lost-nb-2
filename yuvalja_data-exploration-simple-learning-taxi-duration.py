#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Get the data and save it in /data directory
# ! kg config -c nyc-taxi-trip-duration
# ! kg download
#  ! mkdir data
# ! mv test.zip data/test.zip
# ! mv train.zip data/train.zip
# ! mv sample_submission.zip data/sample_submission.zip
# ! unzip -q data/test.zip -d data/
# ! unzip -q data/train.zip -d data/
# ! unzip -q data/sample_submission.zip -d data/


# In[2]:


import os
import numpy as np
import pandas as pd


# In[3]:


def distance(pos1, pos2, r = 3958.75):
    pos1 = np.deg2rad(pos1)
    pos2 = np.deg2rad(pos2)
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


# In[4]:


# Examine the data we got
train_full = pd.read_csv('../input/train.csv') #pd.read_csv("data/train.csv")
test = pd.read_csv('../input/test.csv') #pd.read_csv("data/test.csv")

print('We have {} training rows and {} test rows.'.format(train_full.shape[0], test.shape[0]))

print('We have {} training columns and {} test columns.'.format(train_full.shape[1], test.shape[1]))
train_full.head(2)


# In[5]:


print('Id is unique.' if train_full.id.nunique() == train_full.shape[0] else 'oops')
print('Train and test sets are distinct.' if len(np.intersect1d(train_full.id.values, test.id.values))== 0 else 'oops')
print('We do not need to worry about missing values.' if train_full.count().min() == train_full.shape[0] and test.count().min() == test.shape[0] else 'oops')
print('The vendor_id has only two values {}.'.format(str(set(train_full.vendor_id.unique()) | set(test.vendor_id.unique()))))
print('The store_and_fwd_flag has only two values {}.'.format(str(set(train_full.store_and_fwd_flag.unique()) | set(test.store_and_fwd_flag.unique()))))


# In[6]:


def change_to_boolean(data):
    data = data.drop(['id'], axis=1)
    data['store_and_fwd_flag'] = pd.Series(
        np.where(data.store_and_fwd_flag.values == 'Y', 1, 0), data.index)
    data.vendor_id = pd.Series(np.where(data.vendor_id.values == 1, 1, 0), data.index)
    data = data.rename(columns = {'vendor_id' : 'is_vendor_1'})
    return data

train_full = change_to_boolean(train_full)
test = change_to_boolean(test)


# In[7]:


def create_distance_metric(data):
    pickup = np.column_stack((data.pickup_longitude.values, 
                              data.pickup_latitude.values))
    dropoff = np.column_stack((data.dropoff_longitude.values, 
                               data.dropoff_latitude.values))
    data['distance'] = distance(pickup, dropoff)
    return data

train_full = create_distance_metric(train_full)
test = create_distance_metric(test)


# In[8]:


def create_time_metric(data):
    datetime_pickup = pd.to_datetime(data.pickup_datetime, infer_datetime_format=True)
    data['pickup_time'] = datetime_pickup.dt.time.apply(lambda x: x.replace(second = 0))
    data['pickup_day'] = datetime_pickup.dt.weekday
    data['pickup_hour'] = datetime_pickup.dt.hour
    return data
    
    
train_full = create_time_metric(train_full)
test = create_time_metric(test)


# In[9]:


train_full['speed'] = np.round(train_full['distance'] / (train_full['trip_duration'] / (60*60)),2) # KM/H


# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(train_full.passenger_count, normed=True, bins=np.arange(1,np.max(train_full.passenger_count)) - 0.5)
#plt.xticks(range(0,train_full.passenger_count))
plt.title("passenger count")
plt.show()


# In[11]:


plt.hist(train_full.trip_duration, normed=True, 
         bins=range(int(np.percentile(train_full.trip_duration,1)),
               int(np.percentile(train_full.trip_duration,99)), 60*5))
plt.title("trip duration in (seconds)")
plt.show()


# In[12]:


plt.scatter(train_full.distance, train_full.trip_duration)
plt.xlabel("Distance in KM")
plt.ylabel("Trip duration in sec")
plt.show()


# In[13]:


distance_time_no_outliers = np.column_stack((train_full.distance, train_full.trip_duration, train_full.pickup_hour))
distance_time_no_outliers = distance_time_no_outliers[distance_time_no_outliers[:,1] < np.percentile(distance_time_no_outliers[:,1], 99.9)]
distance_time_no_outliers = distance_time_no_outliers[distance_time_no_outliers[:,0] < np.percentile(distance_time_no_outliers[:,0], 99.9)]

fig, ax = plt.subplots(ncols=2)
ax[0].scatter(distance_time_no_outliers[:,0], distance_time_no_outliers[:,1], s=1, alpha=0.1)
ax[0].set_xlabel("Distance in KM")
ax[0].set_ylabel("Trip duration in sec")
ax[1].scatter(distance_time_no_outliers[:,0], np.log(distance_time_no_outliers[:,1]), s=1, alpha=0.1,
              c=distance_time_no_outliers[:,2], cmap=plt.get_cmap('jet'))
#ax[1].set_xlabel("Distance in KM")
ax[1].set_ylabel("log Trip duration in sec")

plt.show()


# In[14]:


dates = pd.to_datetime(train_full.pickup_datetime, infer_datetime_format=True)

fig, ax = plt.subplots(ncols=2, sharey=True)
ax[0].hist(dates.dt.hour, np.arange(24) - 0.5, normed=True, color=['red'], lw=2)
ax[1].hist(dates.dt.weekday, np.arange(8) - 0.5, normed=True, color=['green'], lw=2)
ax[0].set_xticks(range(0,24,3))
ax[1].set_xticks(range(0,7))
ax[0].set_xlabel('hour')
ax[1].set_xlabel('week day')
ax[0].set_ylabel('count')

fig.show()


# In[15]:


plt.hist(train_full['speed'], range(0,int(np.ceil(np.percentile(train_full['speed'], 99.9)))),
         normed=True)
plt.title("Speed")
plt.show()


# In[16]:


# Try to find the connection between speed and time of day
# For that we need hour of week
train_full['pickup_week_hour'] = train_full['pickup_day'] * 24 + pd.to_datetime(train_full.pickup_datetime, infer_datetime_format=True).dt.hour

# Remove ourliers
no_ourliers = train_full[train_full['speed'] > 0]
no_ourliers = no_ourliers[no_ourliers['speed'] < np.percentile(no_ourliers['speed'], 99.9)]

fig, ax = plt.subplots(ncols=2, sharey=True)
ax[0].plot(train_full.groupby('pickup_hour').mean()['speed'], lw=2)
ax[0].set_ylabel("avg speed")
ax[0].set_xlabel("hour")
ax[1].plot(train_full.groupby('pickup_week_hour').mean()['speed'], lw=2)
ax[1].set_xlabel("week hour")

ax[0].set_xlim(0,24)
ax[1].set_xlim(0,7*24)
fig.show()


# In[17]:


# Let's look at the origin and destantion frequancy

def plot_places(longitude, latitude, title):
    #pickup_lat_bin = np.round(latitude, 3)
    #pickup_long_bin = np.round(longitude, 3)
    
    longitude_limits = (np.percentile(longitude, 0.1), np.percentile(longitude, 99.0))
    latitude_limits = (np.percentile(latitude, 0.1), np.percentile(latitude, 99.9))
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.set_title(title)
    ax.scatter(longitude, latitude, color='black', s=1, alpha=0.5)

    ax.set_xlim(longitude_limits)
    ax.set_ylim(latitude_limits)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()

plot_places(train_full.pickup_longitude.values, train_full.pickup_latitude.values, "Pickup locations")
plot_places(train_full.dropoff_longitude.values, train_full.dropoff_latitude.values, "Dropoff Locations")


# In[18]:


# Split the train data into train and valid
from sklearn.model_selection import train_test_split
train, valid = train_test_split(train_full,test_size=0.2)
print("We have {} train rows, and {} test rows.". format(train.shape[0], valid.shape[0]))


# In[19]:


# Look for linear regression between distance + time + day + origin ~ speed
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def linear_regresion(x,y):
    regr = linear_model.LinearRegression()
    regr.fit(train[x], train[y])

    y_pred = regr.predict(valid[x])

    coef = regr.coef_.tolist()
    coef.insert(0,regr.intercept_)
    x.insert(0,"intercept")
    print("X columns: {}\n y columns: {}\n\n".format(x,y))
    # The coefficients
    print(pd.DataFrame(list(zip(x, coef))))
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(valid[y], y_pred))
    # Explained variance score: 1 is perfect prediction
    print('R2 squre: %.2f' % r2_score(valid[y], y_pred))
    
x = ['pickup_day','pickup_hour', 'distance', 'pickup_latitude', 'pickup_longitude', 'dropoff_longitude', 'dropoff_latitude']
linear_regresion(x, 'speed')


# In[20]:


x = ['pickup_day','pickup_hour', 'distance']
linear_regresion(x, 'speed')


# In[21]:


# First group the trip duration into 10 groups (equal size) and tree to use a decision tree to predict 
groups = train_full['trip_duration'].quantile(np.arange(0.0, 1.0, 0.05))
def get_duration_group(data):
    data.loc[:, 'duration_group'] =data.loc[:, 'trip_duration'].apply(lambda x: np.where(x >= groups)[0][-1]) 
    return data

train = get_duration_group(train)
valid = get_duration_group(valid)


# In[22]:


# Create the tree and predict the validation set
from sklearn import tree

clf = tree.DecisionTreeClassifier()
x = ['pickup_day','pickup_hour', 'distance', 'pickup_latitude', 'pickup_longitude', 'dropoff_longitude', 'dropoff_latitude']
#x = ['pickup_day','pickup_hour', 'distance']
clf = clf.fit(train[x], train['duration_group'])
y_pred = clf.predict(valid[x])


# In[23]:


# Look at the results
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
print("The accuracy score is: {}". format(accuracy_score(valid['duration_group'], y_pred)))
print("The loss (MSE) for the groups is: {}". format(mean_squared_error(valid['duration_group'], y_pred)))

# For each group get the average duration and calcualte the loss
group_duration = [(groups.values[i] + groups.values[i+1]) / 2 for i in range(len(groups)-1)]
duration_pred = np.copy(y_pred)
for i in range(len(group_duration)):
    duration_pred[duration_pred == i] = group_duration[i]

print("The loss (MSE) for the trip duration: {}". format(mean_squared_error(valid['trip_duration'], duration_pred)))

cm = confusion_matrix(valid['duration_group'], y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.matshow(cm)
plt.colorbar()
plt.title('Confusion Matrix Normalized')
plt.show()


# In[24]:


clf.tree_.node_count

