#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[2]:


plt.rcParams["figure.figsize"] = (10, 6) # (w, h)


# In[ ]:





# In[ ]:





# In[3]:


ls ../input/


# In[4]:


data = pd.read_csv('../input/train.csv', parse_dates = [0]) # convert to datetime object 


# In[5]:


data.shape


# In[6]:


data.columns


# In[7]:


# renaming columns to non pandas "reserved" object attributes (to avoid mistakes / confusion)
data.rename(columns={'datetime':'logged',
                          'count':'cnt'}, 
                 inplace=True)


# In[ ]:





# In[8]:


# ordinal value which represents different levels of precipitation. rename to more appropriate name 
data.rename(columns=
            {'weather':'precipitation'          
            },
            inplace=True)


# In[9]:


data.head(20)


# In[10]:


data.describe()


# In[ ]:





# In[11]:


#check for any missing values
data.isnull().values.any()


# In[12]:


# nice!


# In[13]:


#however, windspeed has many zeros. will check that out later 


# In[ ]:





# In[14]:


# making sure a holiday is not a working day
len(data[((data.holiday & data.workingday) == 1)])


# In[15]:


# todo: still there might be correlation between holiday and workingday. will check this out 


# In[ ]:





# In[16]:


data.info()


# In[17]:


'''
logged : datetime 
holiday : categorical
workingday : categorical
precipitation : ordinal . represents the "level" of precipitation
season : ordinal . one season is "greater" than another
temp : numerical
atemp : numerical
humidity : numerical
windspeed : numerical
casual : numerical
registered :numerical
cnt : numerical
'''


# In[ ]:





# In[18]:


data.info()


# In[ ]:





# In[19]:


dataOrig = data.copy() # deep copy
#data = dataOrig.copy()


# In[ ]:





# In[20]:


colsToDrop = []


# In[ ]:





# In[21]:


# unravel date time data and add them as as features 
data['hour'] = data.logged.dt.hour # hour in day
data['day'] = data.logged.dt.day # day in month
data['dayofweek'] = data.logged.dt.dayofweek # day of week 
data['month'] = data.logged.dt.month # month in year
data['year'] = data.logged.dt.year # year within the two year timespan 


# In[ ]:





# In[22]:


# drop unraveled datetime object
colsToDrop.append('logged')


# In[ ]:





# In[23]:


data.head(20)


# In[ ]:





# In[24]:


data.hist(figsize = (25,22), bins=25);


# In[25]:


for col in data.columns:
    print(data[col].value_counts().head(20))
    print()


# In[ ]:





# In[26]:


sns.boxplot(data['cnt']).set_title('Cnt distribution')


# In[27]:


# cnt has many outlier datapoints beyond the outer quartile 


# In[ ]:





# In[28]:


sns.boxplot(data['season'],data['cnt']).set_title('Cnt distribution across season')


# In[29]:


# season 1 (spring) has a significant drop in count 


# In[ ]:





# In[30]:


sns.boxplot(data['precipitation'],data['cnt']).set_title('Cnt distribution across precipitation')


# In[31]:


# amount of rentals drop as it gets more rainy . 


# In[ ]:





# In[32]:


sns.boxplot(data['hour'],data['cnt']).set_title('Cnt distribution across hour')


# In[33]:


# peak hours are during commuting hours: between 7-8 in the morning, and 17-18 in evening


# In[ ]:





# In[34]:


dateCols = ['hour','day','dayofweek','month','year']
nonNumCols = ['season','holiday', 'workingday', 'precipitation']
continuousCols = ['temp','atemp','humidity','windspeed']
targetCols = ['casual','registered','cnt']


# In[ ]:





# In[ ]:





# In[35]:


for col in continuousCols:
    sns.boxplot(col, data=data,orient='v') 
    plt.show()


# In[36]:


'''
temp :  distributed normal
atemp : distributed normal
windspeed : has many outliers 
humidity : few outliers 
'''


# In[ ]:





# In[37]:


# windspeed count plot 


# In[38]:


sns.countplot(data['windspeed']) 


# In[39]:


'''
there are many zero values

logically, if I were renting a bike, the wind levels won't matter so much (as long there are not tornado / hurricane winds)
and it wont really wouldn't be part of my decision to rent a boke 

will need to check for correlation to decide if this feature is kept or not 

'''


# In[ ]:





# In[ ]:





# In[40]:


# bar plot with sum estimator 
for col in dateCols + nonNumCols:
    sns.barplot(x=col, y='cnt', data=data, estimator = sum)
    plt.show()


# In[41]:


# bar plot with mean estimator
# with mean can relatively compare between classes despite high imbalances

for col in dateCols + nonNumCols:
    sns.barplot(x=col, y='cnt', data=data) #estimator = mean (default param)
    plt.show()


# In[42]:


'''
1. summer months had the highest amount of rentals  
2. cnt varies well with: hour, month, year, season, precipitation levels 


todo: check if day in month is correlated to day in week 
'''


# In[ ]:





# In[43]:


fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.set_title('Weekday - Time of Day')

days = {0: 'sunday', 1:'monday', 2:'tuesday',3:'wednesday',4:'thursday',5:'friday',6:'saturday'}

for day in range(7): 
    datetimeGroup = data[data['dayofweek'] == day].groupby(['hour']).mean().reset_index() # reset index turns index back into col after groupby turned it into an index 
    ax.plot(datetimeGroup.hour,datetimeGroup.cnt, label = days[day])

ax.legend(loc=2)
fig.show()


# In[44]:


'''
we now see that the high usage during commuting hours is not during friday+saturday

friday and saturday had high usages during the morning-afternoon (10-16 oclock)
'''


# In[ ]:





# In[ ]:





# In[45]:


fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.set_title('Season - Time of Day')

seasons = {0: 'spring', 1:'summer', 2:'fall',3:'winter'}


for season in range(4): 
    datetimeGroup = data[data['season'] == season].groupby(['hour']).mean().reset_index() # reset index turns index back into col after groupby turned it into an index 
    ax.plot(datetimeGroup.hour,datetimeGroup.cnt, label = seasons[season])

ax.legend(loc=2)
fig.show()


# In[46]:


'''
we see the same  amounts of rental peaks during those certain hours of the day, just a less amount of rentals in the summer season
'''


# In[ ]:





# In[47]:


fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.set_title('Customer Type - Time of Day')

customers = {0: 'casual', 1:'registered'}

for customer in range(len(customers)): 
    datetimeGroup = data.groupby(['hour']).mean().reset_index() # reset index turns index back into col after groupby turned it into an index 
    ax.plot(datetimeGroup.hour,datetimeGroup[customers[customer]], label = customers[customer])

ax.legend(loc=2)
fig.show()


# In[48]:


# registered users alone are the ones who ride during the peak commuting hours, not casual users
# specifally, the casual users ride on the off hours of the peak users.
# could be that the registered users are registered since they use this only to commute, as the more ideal tiem to ride
# seems to be the middle of the day

#also, most riding is done during the second peak (evening)


# In[ ]:





# In[ ]:





# In[49]:


# 24 hour window 


# In[50]:


data['cnt'].rolling(24).mean().plot()
plt.show()


# In[51]:


# can see the spikes which are the changes during the different days of the week as discussed . 


# In[52]:


#zooming out 


# In[53]:


# one month window


# In[54]:


daysPerMonth = len(data.day.unique())


# In[55]:


data['cnt'].rolling(24*daysPerMonth).mean().plot()
plt.show()


# In[ ]:





# In[56]:


#each season (each 3 months)
# https://www.google.com/search?q=how+many+months+a+season&oq=how+many+months+a+season


# In[57]:


#there are 19 days a month


# In[58]:


data['cnt'].rolling(24*daysPerMonth*3).mean().plot()
plt.show()


# In[59]:


# as seen from the daily, monthly, and seasonal rolling windows, the count grows over each season of time


# In[60]:


# since season already correlates to the unraveled datetime features, no need to keep as a feature, it will already be incorporated 


# In[ ]:





# In[61]:


corrMatrix = data.corr(method='spearman')

fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(corrMatrix,square = False,annot =True,cmap='Spectral', ax=ax)

plt.show()


# In[62]:


# pearson
corrMatrix = data.corr(method='pearson')
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(corrMatrix,square = False,annot =True,cmap='Spectral', ax=ax)

plt.show()


# In[63]:


# windspeed has a very low correlation with cnt
colsToDrop.append('windspeed')


# In[ ]:





# In[64]:


#logically there is a high correlation between month and season 


# In[65]:


colsToDrop.append('season')


# In[ ]:





# In[66]:


# hifh corr between cnt, registered and casual. makes sense since they are leakage variables! 


# In[67]:


(data['cnt'] - data['registered']).equals(data['casual'])


# In[68]:


#bingo!
#  drop registered and casual
colsToDrop.append('registered')
colsToDrop.append('casual')


# In[ ]:





# In[69]:


# temperature (both temp and atemp) has a corr with cnt. not very high  but still significant enough


# In[70]:


# humidiy has a neg corr with count. not very high  but still significant enough


# In[ ]:





# In[71]:


# temp is also hightly correlated with atemp


# In[72]:


tempDiff = (data['atemp'] - data['temp'])


# In[73]:


print('mean: ' + str(tempDiff.mean()))
print('median: ' + str(tempDiff.median()))
print('mode: ' + str(tempDiff.mode()[0]))


# In[74]:


tempDiff.hist(bins=50)


# In[75]:


tempDiff.plot()


# In[76]:


tempDiff.plot(kind='box')


# In[77]:


data['atemp'].max()


# In[78]:


data['temp'].max()


# In[79]:


# temp diff does have a few outliers due to atemp having a few spikes, yet temp and atemp are highly correleated and very close in value. so will simply just drop one of them
# decided to drop atemp due to the outliers  


# In[80]:


colsToDrop.append('atemp')


# In[ ]:





# In[81]:


#highly neg correlated with day of week and workingday
# working day: whether the day is neither a weekend nor holiday. that is already derived from day of week and holiday
colsToDrop.append('workingday')


# In[ ]:





# In[82]:


colsToDrop


# In[ ]:





# In[83]:


data.columns


# In[ ]:





# In[84]:


#todo: if have time, explore methods to remove corrlelated dimensions such as pca, knn+pca, lasso, tsne...


# In[ ]:





# In[85]:


# setting the target variable: given an order, predict the amount of rentals there will be that day


# In[86]:


data.head()


# In[87]:


years = data.logged.dt.year.unique()
months = data.logged.dt.month.unique()
days = data.logged.dt.day.unique()


# In[88]:


years


# In[89]:


months


# In[90]:


days


# In[ ]:





# In[91]:


data.head(10)


# In[92]:


#dataOrig = data.copy()


# In[ ]:





# In[93]:


# alrady accomplished this with the datagrouping / aggregation 

'''
data["cntDay"] = np.nan

for y in years:
    for m in months:
        for d in days:
            selection = (data['year'] == y) & (data['month'] == m) & (data['day'] == d)
            data.loc[selection, 'cntDay'] = (data[ selection ])['cnt'].sum()
  '''          


# In[94]:


#making sure data is sorted properly 
data.sort_index(ascending=True).equals(data)


# In[95]:


dataset = data.copy()


# In[96]:


dataset.head(2)


# In[97]:


dataset.columns


# In[98]:


colsToDrop


# In[99]:


dataset = dataset.drop(colsToDrop,axis=1)


# In[100]:


dataset.columns


# In[101]:


dataset = dataset.groupby(['year','month','day']).agg({'precipitation':'mean','temp':'mean','holiday':'mean','humidity':'mean','cnt':'sum'}).reset_index()


# In[102]:


dataset.head()


# In[103]:


dataset['precipitation'] = round(dataset['precipitation'])
dataset['humidity'] = round(dataset['humidity'])
dataset['temp'] = round(dataset['temp'],2)


# In[104]:


dataset.head(10)


# In[105]:


dataset['tomorrowCnt'] = dataset['cnt'].shift(-1)


# In[106]:


dataset.head(10)


# In[107]:


#making sure the only thing na is the last col due to the upshift 


# In[108]:


dataset.isna().sum() #


# In[109]:


dataset = dataset.dropna()


# In[110]:


dataset.isna().sum() #


# In[ ]:





# In[111]:


dataset.head(20)


# In[112]:


target = dataset.tomorrowCnt


# In[113]:


tomorrowCntMean = target.mean()


# In[114]:


tomorrowCntMean


# In[115]:


dataset = dataset.drop(['cnt','tomorrowCnt'],axis=1)


# In[116]:


dataset.columns


# In[ ]:





# In[ ]:





# In[117]:


dataset.head()


# In[118]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


# In[119]:


dataBeforeNorm = dataset.copy()


# In[120]:


#dataset = dataBeforeNorm.copy()


# In[121]:


#commenting out since got better results not scaling 

#dataset = pd.DataFrame(min_max_scaler.fit_transform(dataset), columns=dataset.columns, index=dataset.index)
#dataset.head()  


# In[122]:


for col in dataset.columns:
    print(dataset[col].value_counts().head())
    print()


# In[ ]:





# In[ ]:





# In[123]:


from sklearn.model_selection import train_test_split


# In[124]:


import xgboost as xgb


# In[ ]:





# In[125]:


# we should split this into train, validation, test but i understood from the problem to only split it into train and test 


# In[126]:


#For training, use 90% of the data and 10% for test
X_train, X_test, y_train, y_test  = train_test_split(dataset, target, test_size=0.1, random_state=26)
dm_train = xgb.DMatrix(X_train, label=y_train)
dm_test = xgb.DMatrix(X_test, label=y_test)


# In[127]:


'''
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    assert len(preds) == len(labels)
    #labels = labels.tolist()
    #preds = preds.tolist()

    return 'error', mean_squared_error(preds)/tomorrowCntMean > 1
'''


# In[128]:


clParams = {}
clParams['objective'] = 'reg:linear'
#clParams['eval_metric'] = 'mae' # 'error'

clParams['eta'] = 0.1
clParams['max_depth'] = 8

watchlist = [(dm_train, 'train'), (dm_test, 'test')]

#cl = xgb.train(clParams, dm_train, 10000, watchlist, feval = evalerror, early_stopping_rounds=500, maximize=False, verbose_eval=10)
cl = xgb.train(clParams, dm_train, 10000,  watchlist, early_stopping_rounds=500, maximize=False, verbose_eval=10)


# In[ ]:





# In[129]:


xgb.plot_importance(cl)


# In[ ]:





# In[ ]:





# In[130]:


# Prediction on test 


# In[131]:


def MeasurePerf(preds,y_test):
    
    predsMse = mean_squared_error(y_test,preds)
    meanPreds = np.full(len(preds),tomorrowCntMean)
    meanMse = mean_squared_error(y_test,meanPreds)
    
    res = predsMse / meanMse
    print('predMse/meanMse: '+ str(res))
    
    if (res>=1): # if it equals 1, why train a whole model when u can use average 
        print('\tbetter to use average')
    else:
        print('\tgood prediction')
        
    
    


# In[132]:


def CalculateCompanyGain(ordered,actual): 
    
    extraOrders =  ordered.sum() - actual.sum()
    print("extra orders: " + str(extraOrders))
    
    gain = actual.sum()*15 - ordered.sum()*10  + (extraOrders<0)*extraOrders*100 # supplier charges 10 per vehicle and registered pays 15

    gain = round(gain)
    print('gain: ' + str(gain))    
    return gain


# In[ ]:





# In[133]:


from sklearn.metrics import mean_squared_error


# In[134]:


preds = cl.predict(dm_test)


# In[ ]:





# In[135]:


MeasurePerf(preds,y_test)


# In[ ]:





# In[136]:


# company gains with forcasted 
gainsWithForcasted = CalculateCompanyGain(preds,y_test)


# In[137]:


# company gains with mean
gainsWithAverage = CalculateCompanyGain(np.full(len(preds),tomorrowCntMean),y_test)


# In[138]:


# model doesn't seem to put enough of a penalty on ordering less than required. this is bad since it's more expensive to lose
# a customer then to order extra. the loss function is root mean squared error which substracts the difference between them and squares it
# this gives it equal penalties to under and over ordering. 


# In[ ]:





# In[ ]:





# In[139]:


# did we make more money with forcased or average?

gainsWithForcasted - gainsWithAverage > 0


# In[ ]:





# In[140]:


# sample test 


# In[141]:


sampleX = X_train.head(1)
sampleY = y_train.head(1)


# In[142]:


dm_sample = xgb.DMatrix(sampleX, label=sampleY)


# In[143]:


samplePreds = cl.predict(dm_sample)


# In[144]:


MeasurePerf(samplePreds,sampleY)


# In[ ]:





# In[ ]:





# In[145]:


'''

I would have some threshold defined which marks when we are running low on bikes.

Once that threshold is hit, if a registered customer orders, i will supply him. 

However if a casual user orders, I would need to check the time in the day.
If it's past the second commuting peak, I will grant the casual customer his order.
If it's before the second commuting peak, I will deny the casual customer's order, since I am expecting a large amount of orders
from the registered customers during the next peak. 
(Although the casual users pay more money and their peak time is between the two commuting peaks,
based of the graphs it seems the high volumes of the registered users during their peak hours can generate a higher profit.)

Optionally, more tresholds can be set defining the policy of before the first commuting peak or in between peaks
'''


# In[ ]:





# In[146]:



'''

I would use a custom loss function instead of mse. This was trained on rmse which 
gives an loss of an equal balance to more bikes predicted or less bikes predicted

Instead, we need to give a higher penalty loss when less than the needed amount of bikes are perdicted, 
(since being short of bikes causes the company to suffer an expensive fee of losing customers),
and will penalize less for ordering too many bikes

I'd try a loss of: (actual.sum()*15 - forcasted.sum()*10  + (extraOrders<0)*extraOrders*100 )^2
it is squared so that it will be convex function and have a min 


'''

