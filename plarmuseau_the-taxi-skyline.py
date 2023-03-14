#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import decomposition
from scipy import stats
from sklearn import cluster

matplotlib.style.use('fivethirtyeight')
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (10,10)


# In[2]:


dataDir = '../input/'
taxiDB = pd.read_csv(dataDir + 'train.csv')

# remove obvious outliers
allLat  = np.array(list(taxiDB['pickup_latitude'])  + list(taxiDB['dropoff_latitude']))
allLong = np.array(list(taxiDB['pickup_longitude']) + list(taxiDB['dropoff_longitude']))

longLimits = [np.percentile(allLong, 0.3), np.percentile(allLong, 99.7)]
latLimits  = [np.percentile(allLat , 0.3), np.percentile(allLat , 99.7)]
durLimits  = [np.percentile(taxiDB['trip_duration'], 0.4), np.percentile(taxiDB['trip_duration'], 99.7)]

taxiDB = taxiDB[(taxiDB['pickup_latitude']   >= latLimits[0] ) & (taxiDB['pickup_latitude']   <= latLimits[1]) ]
taxiDB = taxiDB[(taxiDB['dropoff_latitude']  >= latLimits[0] ) & (taxiDB['dropoff_latitude']  <= latLimits[1]) ]
taxiDB = taxiDB[(taxiDB['pickup_longitude']  >= longLimits[0]) & (taxiDB['pickup_longitude']  <= longLimits[1])]
taxiDB = taxiDB[(taxiDB['dropoff_longitude'] >= longLimits[0]) & (taxiDB['dropoff_longitude'] <= longLimits[1])]
taxiDB = taxiDB[(taxiDB['trip_duration']     >= durLimits[0] ) & (taxiDB['trip_duration']     <= durLimits[1]) ]
taxiDB = taxiDB.reset_index(drop=True)

allLat  = np.array(list(taxiDB['pickup_latitude'])  + list(taxiDB['dropoff_latitude']))
allLong = np.array(list(taxiDB['pickup_longitude']) + list(taxiDB['dropoff_longitude']))

# convert fields to sensible units
medianLat  = np.percentile(allLat,50)
medianLong = np.percentile(allLong,50)

latMultiplier  = 111.32
longMultiplier = np.cos(medianLat*(np.pi/180.0)) * 111.32

taxiDB['duration [min]'] = taxiDB['trip_duration']/60.0
taxiDB['src lat [km]']   = latMultiplier  * (taxiDB['pickup_latitude']   - medianLat)
taxiDB['src long [km]']  = longMultiplier * (taxiDB['pickup_longitude']  - medianLong)
taxiDB['dst lat [km]']   = latMultiplier  * (taxiDB['dropoff_latitude']  - medianLat)
taxiDB['dst long [km]']  = longMultiplier * (taxiDB['dropoff_longitude'] - medianLong)

allLat  = np.array(list(taxiDB['src lat [km]'])  + list(taxiDB['dst lat [km]']))
allLong = np.array(list(taxiDB['src long [km]']) + list(taxiDB['dst long [km]']))
pLat  = np.array(list(taxiDB['src lat [km]']))
pLong = np.array(list(taxiDB['src long [km]']) )
dLat  = np.array( list(taxiDB['dst lat [km]']))
dLong = np.array( list(taxiDB['dst long [km]']))


# In[3]:


# make sure the ranges we chose are sensible
fig, axArray = plt.subplots(nrows=1,ncols=3,figsize=(13,4))
axArray[0].hist(taxiDB['duration [min]'],80); 
axArray[0].set_xlabel('trip duration [min]'); axArray[0].set_ylabel('counts')
axArray[1].hist(allLat ,80); axArray[1].set_xlabel('latitude [km]')
axArray[2].hist(allLong,80); axArray[2].set_xlabel('longitude [km]')
plt.show()


# In[4]:


# show the log density of pickup and dropoff locations
imageSize = (350,350)
longRange = [-5,19]
latRange = [-13,11]

allLatInds  = imageSize[0] - (imageSize[0] * (allLat  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)
allLongInds =                (imageSize[1] * (allLong - longRange[0]) / (longRange[1] - longRange[0])).astype(int)
pLatInds  = imageSize[0] - (imageSize[0] * (pLat  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)
pLongInds =                (imageSize[1] * (pLong - longRange[0]) / (longRange[1] - longRange[0])).astype(int)
dLatInds  = imageSize[0] - (imageSize[0] * (dLat  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)
dLongInds =                (imageSize[1] * (dLong - longRange[0]) / (longRange[1] - longRange[0])).astype(int)

locationDensityImage = np.zeros(imageSize)
for latInd, longInd in zip(allLatInds,allLongInds):
    locationDensityImage[latInd,longInd] += 1
plocationDensityImage = np.zeros(imageSize)
for latInd, longInd in zip(pLatInds,pLongInds):
    plocationDensityImage[latInd,longInd] += 1
dlocationDensityImage = np.zeros(imageSize)
for latInd, longInd in zip(dLatInds,dLongInds):
    dlocationDensityImage[latInd,longInd] += 1

print('total pickup dropoff heatmap')
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))
ax.imshow(np.log(locationDensityImage+1),cmap='hot')
ax.set_axis_off()
plt.show()
print('pickup heatmap: more in manhattan')
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))
ax.imshow(np.log(plocationDensityImage+1),cmap='hot')
ax.set_axis_off()
plt.show()
print ('dropoff heatmap more off the island')
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))
ax.imshow(np.log(dlocationDensityImage+1),cmap='hot')
ax.set_axis_off()
plt.show()


# In[5]:


from mpl_toolkits.mplot3d import Axes3D
# To getter a better understanding of interaction of the dimensions
print('pickup Skyline: its a logarithmic scale !')
fig = plt.figure(1, figsize=(12, 12))
ax = Axes3D(fig, elev=30, azim=-41)
col=np.log(locationDensityImage[pLatInds,pLongInds]+1).round()*100
ax.scatter(pLatInds ,pLongInds, np.log(plocationDensityImage[pLatInds,pLongInds]+1), c=col,cmap=plt.cm.Paired)
plt.show()
#---------------------
# To getter a better understanding of interaction of the dimensions
print('dropoff Skyline: ')
fig = plt.figure(1, figsize=(12, 12))
ax = Axes3D(fig, elev=30, azim=-41)
dcol=np.log(locationDensityImage[dLatInds,dLongInds]+1).round()*100
ax.scatter(dLatInds , dLongInds, np.log(plocationDensityImage[dLatInds,dLongInds]+1), c=dcol,cmap=plt.cm.Paired)
plt.show()
#---------------------    

