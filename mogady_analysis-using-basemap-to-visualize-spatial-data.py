#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
from mpl_toolkits.basemap import Basemap
from statsmodels.formula.api import ols


# In[ ]:


train=pd.read_csv('../input/train.csv',nrows=5000000)
test=pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.isna().sum()


# In[ ]:


train=train.dropna()


# In[ ]:


train['passenger_count'].unique()


# In[ ]:


test['passenger_count'].unique()


# In[ ]:


train=train[(train['passenger_count']<=6)&(train['passenger_count']>0)]


# In[ ]:


len(train)


# In[ ]:


train['fare_amount'].describe()


# In[ ]:


#remove negative faer value
train=train[train['fare_amount']>0]


# In[ ]:


sns.boxplot(train['fare_amount'])


# In[ ]:


train[train['fare_amount']>600]


# In[ ]:


train=train[train['fare_amount']<600]


# In[ ]:


plot=sns.distplot(train['fare_amount'])


# In[ ]:


sns.catplot(x='passenger_count',y='fare_amount',data=train)


# In[ ]:


model=ols('fare_amount~passenger_count',data=train).fit()
model.summary()


# In[ ]:


train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].describe()


# In[ ]:


train=train[(train['pickup_longitude']>-180) & (train['pickup_longitude']<180)]


# In[ ]:


train=train[(train['pickup_latitude']>-90) & (train['pickup_latitude']<90)]


# In[ ]:


train=train[(train['dropoff_longitude']>-180) & (train['dropoff_longitude']<180)]


# In[ ]:


train=train[(train['dropoff_latitude']>-90) & (train['dropoff_latitude']<90)]


# In[ ]:


train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].describe()


# In[ ]:


fares=train.fare_amount.values


# In[ ]:


#draw pick up points
fig = plt.figure(figsize=(30, 10))
m = Basemap(projection='cyl',llcrnrlat=40,urcrnrlat=42,            llcrnrlon=-75, urcrnrlon=-72, resolution='h', area_thresh=70, lat_0=40.78, lon_0=-73.96)

lons = train['pickup_longitude'].values
lats = train['pickup_latitude'].values
x,y=m(lons,lats)
### there is a problem with basemap version here that causes drawmapboundry to color everything without this bug u could color the sea with anyother color
#m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgrey',zorder=0)
m.drawstates(color='black')
z=m.scatter(x,y,c=fares,cmap='Reds')
cbar=m.colorbar(z)
cbar.set_label('fares')


# In[ ]:


#draw dropoff points
fig = plt.figure(figsize=(30, 10))
m = Basemap(projection='cyl',llcrnrlat=40,urcrnrlat=42,            llcrnrlon=-75, urcrnrlon=-72, resolution='h', area_thresh=70, lat_0=40.78, lon_0=-73.96)

lons = train['dropoff_longitude'].values
lats = train['dropoff_latitude'].values
x,y=m(lons,lats)
### there is a problem with basemap version here that causes drawmapboundry to color everything without this bug u could color the sea with anyother color
#m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgrey',zorder=0)
m.drawstates(color='black')
z=m.scatter(x,y,c=fares,cmap='Reds')
cbar=m.colorbar(z)
cbar.set_label('fares')


# In[ ]:


#draw pick up for test data
fig = plt.figure(figsize=(30, 10))
m = Basemap(projection='cyl',llcrnrlat=40,urcrnrlat=42,            llcrnrlon=-75, urcrnrlon=-72, resolution='h', area_thresh=70, lat_0=40.78, lon_0=-73.96)

lons = test['pickup_longitude'].values
lats = test['pickup_latitude'].values
x,y=m(lons,lats)
### there is a problem with basemap version here that causes drawmapboundry to color everything without this bug u could color the sea with anyother color
#m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgrey',zorder=0)
m.drawstates(color='black')
m.scatter(x,y)


# In[ ]:


#draw drofoff for test data
fig = plt.figure(figsize=(30, 10))
m = Basemap(projection='cyl',llcrnrlat=40,urcrnrlat=42,            llcrnrlon=-75, urcrnrlon=-72, resolution='h', area_thresh=70, lat_0=40.78, lon_0=-73.96)

lons = test['dropoff_longitude'].values
lats = test['dropoff_latitude'].values
x,y=m(lons,lats)
### there is a problem with basemap version here that causes drawmapboundry to color everything without this bug u could color the sea with anyother color
#m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgrey',zorder=0)
m.drawstates(color='black')
m.scatter(x,y)


# In[ ]:


train['harv_distance']=6371 * 2 * np.arcsin(np.sqrt(np.sin((np.radians(train['dropoff_latitude']) -np.radians(train['pickup_latitude']))/2)**2 + np.cos(np.radians(train['pickup_latitude'])) * np.cos(np.radians(train['dropoff_latitude'])) * np.sin((np.radians(train['dropoff_longitude']) - np.radians(train['pickup_longitude']))/2)**2))


# In[ ]:


train.harv_distance.describe()


# In[ ]:


train.harv_distance.max()


# In[ ]:


train.sort_values('harv_distance', ascending=False)


# In[ ]:


sns.scatterplot(x='harv_distance',y='fare_amount',data=train)


# In[ ]:


train=train[(train['pickup_longitude']!=train['dropoff_longitude']) & (train['pickup_latitude']!=train['dropoff_latitude'])]


# In[ ]:


train[(train['harv_distance']>500)&(train['harv_distance']<6000)].sort_values(['harv_distance'],ascending=True)


# In[ ]:


fig = plt.figure(figsize=(30, 10))
m = Basemap(projection='cyl',llcrnrlat=7,urcrnrlat=90,            llcrnrlon=-90, urcrnrlon=-4, resolution='h', area_thresh=50)

lons1 = -73.99503
lats1 = 40.744945
x1,y1=m(lons1,lats1)

lons2 = -7.98664
lats2 = 40.729937
x2,y2=m(lons2,lats2)
### there is a problem with basemap version here that causes drawmapboundry to color everything without this bug u could color the sea with anyother color
#m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgrey',zorder=0)
m.drawstates(color='black')
m.scatter(x1,y1,color='blue',marker='x')
m.scatter(x2,y2,color='red',marker='o')
plt.arrow(x1,y1,x2-x1,y2-y1,color='black')


# In[ ]:


fig = plt.figure(figsize=(30, 10))
m = Basemap(projection='cyl',llcrnrlat=40,urcrnrlat=45,            llcrnrlon=-80, urcrnrlon=-70, resolution='h', area_thresh=50)

lons1 = -73.977917
lats1 = 40.752368
x1,y1=m(lons1,lats1)

lons2 = -79.090594
lats2 = 43.178567
x2,y2=m(lons2,lats2)
### there is a problem with basemap version here that causes drawmapboundry to color everything without this bug u could color the sea with anyother color
#m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgrey',zorder=0)
m.drawstates(color='black')
m.scatter(x1,y1,color='blue',marker='x')
m.scatter(x2,y2,color='red',marker='o')


# In[ ]:


train=train[(train['pickup_latitude']>=40) & (train['pickup_latitude']<=42)&            (train['dropoff_latitude']>=40) & (train['dropoff_latitude']<=42)&            (train['pickup_longitude']>-75) & (train['pickup_longitude']<=-72)&           (train['dropoff_longitude']>-75) & (train['dropoff_longitude']<=-72)]


# In[ ]:


train['harv_distance']=6371 * 2 * np.arcsin(np.sqrt(np.sin((np.radians(train['dropoff_latitude']) -np.radians(train['pickup_latitude']))/2)**2 + np.cos(np.radians(train['pickup_latitude'])) * np.cos(np.radians(train['dropoff_latitude'])) * np.sin((np.radians(train['dropoff_longitude']) - np.radians(train['pickup_longitude']))/2)**2))


# In[ ]:


sns.scatterplot(x='harv_distance',y='fare_amount',data=train)


# In[ ]:


train.harv_distance.describe()


# In[ ]:


groups=train.groupby('passenger_count')


# In[ ]:


g=sns.FacetGrid(train,col='passenger_count')
g.map(plt.scatter,'harv_distance','fare_amount')


# In[ ]:


dup = train[train.duplicated(subset=['dropoff_longitude','dropoff_latitude'], keep=False)]


# In[ ]:


repeated=dup.groupby(['dropoff_longitude','dropoff_latitude'])


# In[ ]:


df=pd.DataFrame()
for g in repeated.groups:
    df=df.append({'long': g[0], 'lat': g[1], 'avg_fare': repeated.get_group(g)['fare_amount'].mean(),'avg_distance':repeated.get_group(g)['harv_distance'].mean(),'counts':repeated.get_group(g).count()['key']}, ignore_index=True)


# In[ ]:


fig = plt.figure(figsize=(30, 10))
m = Basemap(projection='cyl',llcrnrlat=40,urcrnrlat=42,            llcrnrlon=-75, urcrnrlon=-72, resolution='h', area_thresh=70, lat_0=40.78, lon_0=-73.96)

counts=df.counts.values
lons=df.long.values
lans=df.lat.values
fares=df.avg_fare.values
x,y=m(lons,lans)
### there is a problem with basemap version here that causes drawmapboundry to color everything without this bug u could color the sea with anyother color
#m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgrey',zorder=0)
m.drawstates(color='black')
z=m.scatter(x,y,s=counts,c=fares,cmap='Reds')
cbar=m.colorbar(z)
cbar.set_label('fares')


# In[ ]:


sns.pairplot(df[['avg_fare','avg_distance','counts']])


# In[ ]:


df[df['counts']==df['counts'].max()]


# In[ ]:


train[(train['dropoff_longitude']==-74.1771926879883)&(train['dropoff_latitude']==40.6949462890625)]


# In[ ]:


dup = train[train.duplicated(subset=['pickup_longitude','pickup_latitude'], keep=False)]
repeated=dup.groupby(['pickup_longitude','pickup_latitude'])
dt=pd.DataFrame()
for g in repeated.groups:
    dt=dt.append({'long': g[0], 'lat': g[1], 'avg_fare': repeated.get_group(g)['fare_amount'].mean(),'avg_distance':repeated.get_group(g)['harv_distance'].mean(),'counts':repeated.get_group(g).count()['key']}, ignore_index=True)


# In[ ]:


fig = plt.figure(figsize=(30, 10))
m = Basemap(projection='cyl',llcrnrlat=40,urcrnrlat=42,            llcrnrlon=-75, urcrnrlon=-72, resolution='h', area_thresh=70, lat_0=40.78, lon_0=-73.96)

counts=dt.counts.values
lons=dt.long.values
lans=dt.lat.values
fares=dt.avg_fare.values
x,y=m(lons,lans)
### there is a problem with basemap version here that causes drawmapboundry to color everything without this bug u could color the sea with anyother color
#m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgrey',zorder=0)
m.drawstates(color='black')
z=m.scatter(x,y,s=counts,c=fares,cmap='Reds')
cbar=m.colorbar(z)
cbar.set_label('fares')


# In[ ]:


sns.pairplot(dt[['avg_fare','avg_distance','counts']])


# In[ ]:


dt.sort_values('counts',ascending=False).head()


# In[ ]:


dt.sort_values('avg_fare',ascending=False).head()


# In[ ]:


#### Now lets look at the datetime column

train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'],infer_datetime_format=True)

train['year']=train.pickup_datetime.apply(lambda x:x.year)

train['day']=train.pickup_datetime.apply(lambda x:x.weekday())

train['month']=train.pickup_datetime.apply(lambda x:x.month)

train['hour']=train.pickup_datetime.apply(lambda x:x.hour)


# In[ ]:


train.head()


# In[ ]:


groups=train.groupby(['day','hour'])
data=groups.mean()['fare_amount'].reset_index()


# In[ ]:


g=sns.FacetGrid(data,col='day',col_wrap=4)
g.map(sns.pointplot,'hour','fare_amount')


# In[ ]:


sns.distplot(train['hour'],kde=False)


# In[ ]:


sns.catplot(x='day',y='fare_amount',data=train)


# In[ ]:


pd.crosstab(train['fare_amount'],train['day']).plot(kind='kde',xlim=[-1000,1000])


# In[ ]:


groups=train.groupby(['year','hour'])
data=groups.mean()['fare_amount'].reset_index()
g=sns.FacetGrid(data,col='year',col_wrap=4)
g.map(sns.pointplot,'hour','fare_amount')


# In[ ]:


test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'],infer_datetime_format=True)
test['year']=test.pickup_datetime.apply(lambda x:x.year)
test['day']=test.pickup_datetime.apply(lambda x:x.weekday())
test['hour']=test.pickup_datetime.apply(lambda x:x.hour)
test['harv_distance']=6371 * 2 * np.arcsin(np.sqrt(np.sin((np.radians(test['dropoff_latitude']) -np.radians(test['pickup_latitude']))/2)**2 + np.cos(np.radians(test['pickup_latitude'])) * np.cos(np.radians(test['dropoff_latitude'])) * np.sin((np.radians(test['dropoff_longitude']) - np.radians(test['pickup_longitude']))/2)**2))


# In[ ]:


test.head()


# In[ ]:


train['xpick']=np.cos(train['pickup_latitude'])*np.cos(train['pickup_longitude'])
train['ypick']=np.cos(train['pickup_latitude'])*np.sin(train['pickup_longitude'])
train['zpick']=np.sin(train['pickup_latitude'])
train['xdrop']=np.cos(train['dropoff_latitude'])*np.cos(train['dropoff_longitude'])
train['ydrop']=np.cos(train['dropoff_latitude'])*np.cos(train['dropoff_longitude'])
train['zdrop']=np.sin(train['dropoff_latitude'])
test['xpick']=np.cos(test['pickup_latitude'])*np.cos(test['pickup_longitude'])
test['ypick']=np.cos(test['pickup_latitude'])*np.sin(test['pickup_longitude'])
test['zpick']=np.sin(test['pickup_latitude'])
test['xdrop']=np.cos(test['dropoff_latitude'])*np.cos(test['dropoff_longitude'])
test['ydrop']=np.cos(test['dropoff_latitude'])*np.cos(test['dropoff_longitude'])
test['zdrop']=np.sin(test['dropoff_latitude'])


# In[ ]:


features = ['year', 'hour', 'harv_distance','day', 'passenger_count','xpick','ypick','zpick','xdrop','ydrop','zdrop']
X = train[features].values
y = train['fare_amount'].values


# In[ ]:


X.shape, y.shape


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Create the random forest
random_forest = RandomForestRegressor(n_estimators = 20, max_depth = 20, 
                                      max_features = None, oob_score = True, 
                                      bootstrap = True, verbose = 1, n_jobs = -1)

# Train on data
random_forest.fit(X_train, y_train)


# In[ ]:


y_pred = random_forest.predict(X_test)
fig, axs = plt.subplots(1,1,figsize=(10,4))
axs.scatter(y_test, y_pred)
print("Mean Squared Error: %.4f"
      % np.sqrt(np.mean((y_pred - y_test) ** 2)))


# In[ ]:


XTEST = test[features].values
filename = './output/rf_model'

y_pred_final = random_forest.predict(XTEST)

submission = pd.DataFrame(
    {'key': test.key, 'fare_amount': y_pred_final},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission1.csv', index = False)

