#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',index_col=0)
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',index_col=0)


# In[2]:


latmin = 40.48
lonmin = -74.28
latmax = 40.93
lonmax = -73.65
ratio = np.cos(40.7 * np.pi/180) * (lonmax-lonmin) /(latmax-latmin)
from matplotlib.colors import LogNorm
fig = plt.figure(1, figsize=(8,ratio*8) )
hist = plt.hist2d(train.pickup_longitude,train.pickup_latitude,bins=199,range=[[lonmin,lonmax],[latmin,latmax]],norm=LogNorm())
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Pickup Locations')
plt.colorbar(label='Number')
plt.show()


# In[3]:


fig = plt.figure(1, figsize=(8,ratio*8) )
hist = plt.hist2d(train.dropoff_longitude,train.dropoff_latitude,bins=199,range=[[lonmin,lonmax],[latmin,latmax]],norm=LogNorm())
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Dropoff Locations')
plt.colorbar(label='Number')
plt.show()


# In[4]:


def get_census_data():
    blocks = pd.read_csv('../input/new-york-city-census-data/census_block_loc.csv')
    #blocks = blocks[blocks.County.isin(['Bronx','Kings','New York','Queens','Richmond'])]
    census = pd.read_csv('../input/new-york-city-census-data/nyc_census_tracts.csv',index_col=0)
    blocks['Tract'] = blocks.BlockCode // 10000
    blocks = blocks.merge(census,how='left',left_on='Tract',right_index=True)
    #blocks = blocks.dropna(subset=['Borough'],axis=0)
    return blocks,census

def convert_to_2d(lats,lons,values):
    latmin = 40.48
    lonmin = -74.28
    latmax = 40.93
    lonmax = -73.65
    lon_vals = np.mgrid[lonmin:lonmax:200j]
    lat_vals = np.mgrid[latmin:latmax:200j]
    map_values = np.zeros([200,200],'l')
    dlat = lat_vals[1] - lat_vals[0]
    dlon = lon_vals[1] - lon_vals[0]
    for lat,lon,value in zip(lats,lons,values):
        lat_idx = int(np.rint((lat - latmin) / dlat))
        lon_idx = int(np.rint((lon-lonmin) / dlon ))        
        if not np.isnan(value):
            map_values[lon_idx,lat_idx] = value
    return lat_vals,lon_vals,map_values

blocks,census = get_census_data()
blocks_tmp = blocks[blocks.County_x.isin(['Bronx','Kings','New York','Queens','Richmond'])]
map_lats, map_lons,map_tracts_nyc = convert_to_2d(blocks_tmp.Latitude,blocks_tmp.Longitude,blocks_tmp.Tract)
map_lats, map_lons,map_tracts = convert_to_2d(blocks.Latitude,blocks.Longitude,blocks.Tract)


# In[5]:


def get_tract(lat,lon):
    latmin = 40.48
    lonmin = -74.28
    latmax = 40.93
    lonmax = -73.65
    dlat = (latmax-latmin) / 199
    dlon = (lonmax-lonmin) / 199
    if (latmin<lat<latmax) and (lonmin<lon<lonmax):
        lat_idx = int(np.rint((lat - latmin) / dlat))
        lon_idx = int(np.rint((lon-lonmin) / dlon )) 
        return map_tracts[lon_idx,lat_idx]
    return 0


# In[6]:


train.info()


# In[7]:


train['pu_tracts'] = np.array([get_tract(lat,lon) for lat,lon in zip(train.pickup_latitude,train.pickup_longitude)])
train['do_tracts'] = np.array([get_tract(lat,lon) for lat,lon in zip(train.dropoff_latitude,train.dropoff_longitude)])

test['pu_tracts'] = np.array([get_tract(lat,lon) for lat,lon in zip(test.pickup_latitude,test.pickup_longitude)])
test['do_tracts'] = np.array([get_tract(lat,lon) for lat,lon in zip(test.dropoff_latitude,test.dropoff_longitude)])


# In[8]:


pickups = train['pu_tracts'].value_counts()
dropoffs = train['do_tracts'].value_counts()


# In[9]:


top_tracts = [x for x in pickups.index.values[0:5]]
top_tracts.reverse()
values = 0.250*(1-np.isin(map_tracts_nyc,top_tracts+[0]))

for i in range(len(top_tracts)):
    values += (i+7)*(map_tracts_nyc==top_tracts[i])/11

fig = plt.figure(1,figsize=[7,7])
im = plt.imshow(values.T,origin='lower',cmap='jet')
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Most Common Pickup Points in NYC Limits')
plt.colorbar(im,fraction=0.045, pad=0.04)
plt.show()


# In[10]:


top_tracts = [x for x in dropoffs.index.values[0:5]]
top_tracts.reverse()
values = 0.250*(1-np.isin(map_tracts_nyc,top_tracts+[0]))

for i in range(len(top_tracts)):
    values += (i+7)*(map_tracts_nyc==top_tracts[i])/11

fig = plt.figure(1,figsize=[7,7])
im = plt.imshow(values.T,origin='lower',cmap='jet')
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Most Common Dropoff Points in NYC Limits')
plt.colorbar(im,fraction=0.045, pad=0.04)
plt.show()


# In[11]:


areas = blocks.Tract.value_counts()


# In[12]:


pu_area_norm = pickups
do_area_norm = dropoffs

pu_area_norm = pd.concat([pu_area_norm,areas],join='inner',axis=1)
do_area_norm = pd.concat([do_area_norm,areas],join='inner',axis=1)

pu_area_norm['areas'] = pu_area_norm.pu_tracts/pu_area_norm.Tract
do_area_norm['areas'] = do_area_norm.do_tracts/do_area_norm.Tract

pu_areas = pu_area_norm.areas.sort_values(ascending=False)
do_areas = do_area_norm.areas.sort_values(ascending=False)


# In[13]:


top_tracts = [x for x in pu_areas.index.values[0:10]]
top_tracts.reverse()
values = 0.250*(1-np.isin(map_tracts_nyc,top_tracts+[0]))

for i in range(len(top_tracts)):
    values += (i+7)*(map_tracts_nyc==top_tracts[i])/16
    

fig = plt.figure(1,figsize=[7,7])
im = plt.imshow(values.T,origin='lower',cmap='jet')
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Densest Pickup Points in NYC Limits')
plt.colorbar(im,fraction=0.045, pad=0.04)
plt.show()


# In[14]:


top_tracts = [x for x in do_areas.index.values[0:10]]
top_tracts.reverse()
values = 0.250*(1-np.isin(map_tracts_nyc,top_tracts+[0]))

for i in range(len(top_tracts)):
    values += (i+7)*(map_tracts_nyc==top_tracts[i])/16

fig = plt.figure(1,figsize=[7,7])
im = plt.imshow(values.T,origin='lower',cmap='jet')
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Densest Dropoff Points in NYC Limits')
plt.colorbar(im,fraction=0.045, pad=0.04)
plt.show()


# In[15]:


train['PU_Location'] =  train.pu_tracts//1000000
fips_codes = {36061:'Manhattan',36081:'Queens',36047:'Brooklyn',
              36005:'Bronx',36085:'Staten Island',36059:'Nassau',36119:'Westchester',
              34017:'NJ',34013:'NJ',34003:'NJ',34039:'NJ',
              34031:'NJ',34023:'NJ',34025:'NJ',0:'Unknown'
             }
train.PU_Location = train.PU_Location.map(fips_codes)

test['PU_Location'] = test.pu_tracts//1000000
test.PU_Location = test.PU_Location.map(fips_codes)

train.PU_Location.value_counts()


# In[16]:


train['DO_Location'] =  train.do_tracts//1000000
train.DO_Location = train.DO_Location.map(fips_codes)
test['DO_Location'] =  test.do_tracts//1000000
test.DO_Location = test.DO_Location.map(fips_codes)
train.DO_Location.value_counts()


# In[17]:


pd.set_option('display.max_rows', 200)
print(train.groupby(['PU_Location','DO_Location'])['trip_duration'].describe()[['count','25%','50%','75%']])
pd.reset_option('display.max_rows')


# In[18]:


loc_map = {"Manhattan":0,"Queens":1,"Brooklyn":2,"Bronx":3,
           "NJ":4,"Unknown":5,"Staten Island":6,"Nassau":7,
           "Westchester":8}

train.PU_Location = train.PU_Location.map(loc_map)
train.DO_Location = train.DO_Location.map(loc_map)

test.PU_Location = test.PU_Location.map(loc_map)
test.DO_Location = test.DO_Location.map(loc_map)


# In[19]:


targets = np.log(train.trip_duration+1)
Xall = train[['PU_Location',"DO_Location"]]

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
nsplits = 10
tree_model = DecisionTreeRegressor()
kf = KFold(n_splits = nsplits,random_state=999)

err_train_ave = 0 
err_test_ave = 0
err_train_std = 0
err_test_std = 0
counter = 0
for train_index, test_index in kf.split(Xall):
    X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
    y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]
    tree_model.fit(X_train,y_train)
    pred_train = tree_model.predict(X_train)
    pred_test = tree_model.predict(X_test)
    err_train = np.sqrt(mean_squared_error(y_train,pred_train))
    err_test = np.sqrt(mean_squared_error(y_test,pred_test))
    err_train_ave += err_train
    err_test_ave += err_test
    err_train_std += err_train*err_train
    err_test_std += err_test * err_test
    print('%i Train Err: %f, Validation Err: %f'%(counter,err_train,err_test))
    counter+=1

err_train_ave /= nsplits
err_test_ave /= nsplits
err_train_std /= nsplits
err_test_std /= nsplits
err_train_std = np.sqrt((err_train_std - err_train_ave*err_train_ave)*nsplits / (nsplits-1))
err_test_std = np.sqrt((err_test_std - err_test_ave*err_test_ave)*nsplits / (nsplits-1))

print("\nTrain Ave: %f Std. Dev: %f" %(err_train_ave,err_train_std))
print("Validation Ave: %f Std. Dev: %f"%(err_test_ave,err_test_std))


# In[20]:


train.head()


# In[21]:


train['dlon'] = (train.dropoff_longitude-train.pickup_longitude) * np.pi/180 *                np.cos((train.dropoff_latitude+train.pickup_latitude) * 0.5 * np.pi/180)
train['dlat'] = (train.dropoff_latitude-train.pickup_latitude) * np.pi/180
Re = 6371 # Earth radius in km

train['dist'] = Re*np.hypot(train.dlon,train.dlat)
train['pu_do_code'] = train.PU_Location + 10 * train.DO_Location

test['dlon'] = (test.dropoff_longitude-test.pickup_longitude) * np.pi/180 *                np.cos((test.dropoff_latitude+test.pickup_latitude) * 0.5 * np.pi/180)
test['dlat'] = (test.dropoff_latitude-test.pickup_latitude) * np.pi/180
test['dist'] = Re*np.hypot(test.dlon,test.dlat)
# Encode Pickup/Dropoff location into a single number
test['pu_do_code'] = test.PU_Location + 10 * test.DO_Location


train.head()


# In[22]:


plt.hist(train.dist,bins=100,range=[0,50])
plt.yscale('log')
plt.show()


# In[23]:


train['ldist'] = np.log(train.dist + 0.01)
train['d2'] = train.dist*train.dist
train['d1_2'] = np.sqrt(train.dist)
train['d3_2'] = train.dist * train.d1_2

test['ldist'] = np.log(test.dist + 0.01)
test['d2'] = test.dist*test.dist
test['d1_2'] = np.sqrt(test.dist)
test['d3_2'] = test.dist * test.d1_2

X_all = train[['dist','ldist','d2','d1_2','d3_2']]
X_test = test[['dist','ldist','d2','d1_2','d3_2']]

#for i in range(9):
#    for j in range(9):
#        code = 1.0*(train['pu_do_code'] == (10*i+j))
#        if code.sum() > 10:
#            X_train['PU_DO_%i%i'%(i,j)] = code

y_all = np.log(train['trip_duration']+1)


# In[24]:


X_all.head()


# In[25]:


from sklearn.linear_model import LinearRegression

nsplits = 10
lin_model = LinearRegression()
kf = KFold(n_splits = nsplits,random_state=999)
err_train_ave = 0 
err_test_ave = 0
err_train_std = 0
err_test_std = 0
counter = 0
for train_index, test_index in kf.split(X_all):
    X_train, X_val = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_val = y_all.iloc[train_index], y_all.iloc[test_index]
    lin_model.fit(X_train,y_train)
    pred_train = lin_model.predict(X_train)
    pred_test = lin_model.predict(X_val)
    err_train = np.sqrt(mean_squared_error(y_train,pred_train))
    err_test = np.sqrt(mean_squared_error(y_val,pred_test))
    err_train_ave += err_train
    err_test_ave += err_test
    err_train_std += err_train*err_train
    err_test_std += err_test * err_test
    print('%i Train Err: %f, Validation Err: %f'%(counter,err_train,err_test))
    counter+=1
err_train_ave /= nsplits
err_test_ave /= nsplits
err_train_std /= nsplits
err_test_std /= nsplits
err_train_std = np.sqrt((err_train_std - err_train_ave*err_train_ave)*nsplits / (nsplits-1))
err_test_std = np.sqrt((err_test_std - err_test_ave*err_test_ave)*nsplits / (nsplits-1))

print("\nTrain Ave: %f Std. Dev: %f" %(err_train_ave,err_train_std))
print("Validation Ave: %f Std. Dev: %f"%(err_test_ave,err_test_std))


# In[26]:


lin_model.fit(X_all,y_all)
#print(lin_model.coef_)
#print(lin_model.intercept_)
#print(X_test[X_test.isnull().any(axis=1)])
log_test_pred = lin_model.predict(X_test)

test_pred = np.exp(log_test_pred) - 1
#print(test_pred)
test['pred'] = [ np.max(x,0) for x in test_pred ]
X_out = test[['pred']]
X_out.columns.values[0] = 'trip_duration'
#print(X_out)
#X_out['trip_duration'] = test_pred
X_out.to_csv('LinearModel.csv')

