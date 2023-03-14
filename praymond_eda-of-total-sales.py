#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np                 # linear algebra
import pandas as pd                # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime as dt
from sklearn import linear_model
from pandas.tseries.holiday import USFederalHolidayCalendar as cal1
import calendar


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# let's remove some of the warnings we get from the libraries


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[ ]:


MyTrain = pd.read_csv('../input/train.csv',parse_dates=['date'], index_col=['date'])
MyTest  = pd.read_csv('../input/test.csv',parse_dates=['date'], index_col=['date'])

print("Number of training samples read : ", len(MyTrain) )
print("        Shape is (rows, columns): ", MyTrain.shape)
print("Number of testing samples read  : ", len(MyTest)  )
print("        Shape is (rows, columns): ", MyTest.shape )


# In[ ]:


cal = cal1()
holidays = cal.holidays(start=MyTrain.index.min(), end=MyTrain.index.max())


# In[ ]:


MyTrain.describe()


# In[ ]:


ZeroSales = MyTrain[MyTrain.sales == 0]
ZeroSales


# In[ ]:


g = sns.FacetGrid(data=MyTrain,row='item',col='store')
g = g.map(plt.plot,'sales') 


# In[ ]:


MyTrain.groupby(MyTrain.index).sum()['sales'].plot(style='.',figsize=(20,5))


# In[ ]:


MyTrain.groupby(MyTrain.index).sum()['sales'][0:160].plot(style='.',figsize=(20,5))


# In[ ]:


MyTrain['year']  = MyTrain.index.year
MyTrain['month'] = MyTrain.index.month
MyTrain['DoM']   = MyTrain.index.day
MyTrain['DoW']   = MyTrain.index.dayofweek # Mondays are 0
MyTrain['DoY']   = MyTrain.index.dayofyear

MyTrain['Holiday'] = MyTrain.index.isin(holidays)
MyTrainHolidays = MyTrain[MyTrain.Holiday==True]

plt.figure(figsize=(20,14))

plt.subplot(221)
plt.title('Sales per individual store/item for each day-of-week')
sns.boxplot(x=MyTrain.DoW,y=MyTrain.sales)

y = MyTrain.groupby(MyTrain.index).sum().sales
x = MyTrain.groupby(MyTrain.index).min().DoW

plt.subplot(222)
plt.title('Aggregated sales over all stores/items for each day-of-week')
sns.boxplot(x=x,y=y)



plt.subplot(223)
plt.title('Holiday Sales per individual store/item for each day-of-week')
x=MyTrainHolidays.DoW
y=MyTrainHolidays.sales
sns.boxplot(x=x,y=y)


plt.subplot(224)
plt.title('Holiday Aggregates Sales over all store/item for each day-of-week')
x=MyTrainHolidays.groupby(MyTrainHolidays.index).min().DoW
y=MyTrainHolidays.groupby(MyTrainHolidays.index).sum().sales
sns.boxplot(x=x,y=y)


# In[ ]:


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets

# Take total sale for each day and index days with ordinal values.
GroupedMyTrain = MyTrain.groupby(MyTrain.index).sum()['sales'].reset_index()
GroupedMyTrain['OrdinalDate'] = GroupedMyTrain['date'].map(dt.datetime.toordinal)

# prepare x and y data frames
x = GroupedMyTrain[['OrdinalDate']] 
y = GroupedMyTrain[['sales']]

fit = regr.fit(x,y)
P   = regr.predict(x) 

plt.figure(figsize=(20,8))
plt.annotate('Coef: {}'.format(regr.coef_) , xy=(x.iloc[750].OrdinalDate,P[[750]]),  xycoords='data',
            xytext=(0.35, 0.85), textcoords='axes fraction',
            arrowprops=dict(facecolor='green', shrink=1),
            horizontalalignment='right', verticalalignment='top',
            )

plt.plot(GroupedMyTrain[['date']], y,  color='black', marker='.'  , ls=' ' , markersize=2 )
plt.plot(GroupedMyTrain[['date']], P,  color='green' , linewidth=4 )


# In[ ]:


yCor = y - P
yCor.plot(figsize=(20,5),marker='.',markersize=2,ls=' ')


# In[ ]:


yCor = y * P[0]/P 
yCor.plot(figsize=(20,5),marker='.',markersize=2,ls=' ')


# In[ ]:


plt.figure(figsize=(20,5))

plt.subplot(121)
plt.plot(
    x.OrdinalDate[0:140],
    MyTrain.reset_index().groupby(MyTrain.index).sum()['sales'][0:140],
    marker='.',ls=' ',color='blue')
plt.plot(
    x.OrdinalDate[0:140],
    yCor[0:140],
    color='red',marker='.',ls=' ')

plt.subplot(122)
plt.plot(
    x.OrdinalDate[-140:],
    MyTrain.groupby(MyTrain.index).sum()['sales'][-140:],
    marker='.',ls=' ',color='blue')
plt.plot(
    x.OrdinalDate[-140:],
    yCor[-140:],
    color='red',marker='.',ls=' ')


# In[ ]:


yS = yCor
yS['date']  = GroupedMyTrain.date
yS['DoY']   = yS.date.dt.dayofyear
yS['year']  = yS.date.dt.year
yS['DoW']   = yS.date.dt.dayofweek
yS['month'] = yS.date.dt.month


pattern = yS.groupby(yS.DoY).sum()['sales'] / 10 /50 / 5


# In[ ]:


pattern.plot(marker='.',ls=' ',markersize=2,figsize=(20,5))


# In[ ]:


DoWFactor = yS.groupby(yS.DoW).sum().sales
DoWFactor = DoWFactor / DoWFactor.mean()
plt.bar(x=DoWFactor.index, height=DoWFactor)


# In[ ]:


yS2 = MyTrain.groupby([MyTrain.index,MyTrain.DoW]).sum()['sales'].reset_index()
yS2['DoY']   = yS2.date.dt.dayofyear
yS2['month'] = yS2.date.dt.month
yS2['year'] = yS.date.dt.year
yS2['DiM']   = yS2.date.dt.day
yS2['DoW']   = yS2.date.dt.dayofweek
yS2['DoWFactor']   = yS2.groupby(yS2.index).min()['DoW'].map(lambda x: DoWFactor[x])
yS2['GFactor'] = P[0] / P
yS2['CorrSales'] = yS2.sales * yS2.GFactor /yS2.DoWFactor

yS2 = yS2.set_index('date')

pattern = yS2.groupby('DoY').sum()['CorrSales'] / 10 /50 / 5
pattern.plot(marker='.',ls=' ',markersize=2,figsize=(20,5))


# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(121)
pattern[90:120].plot(marker='.',ls=' ',markersize=10)
plt.subplot(122)
sns.distplot(pattern[90:120])
plt.show()


# In[ ]:


pattern = yS2.groupby(['month','DiM']).sum()['CorrSales'] / 10 /50 / 5
pattern.plot(marker='.',ls=' ',markersize=3,figsize=(20,5))


# In[ ]:


MonthFactor = yS2.groupby(yS2.month).sum().sales
MonthFactor = MonthFactor / MonthFactor.mean()
plt.bar(x=MonthFactor.index, height=MonthFactor)


# In[ ]:


GroupedMyTrain['DoWFactor']   = MyTrain.groupby(MyTrain.index).min().reset_index()['DoW'].map(lambda x: DoWFactor[x])
GroupedMyTrain['MonthFactor'] = MyTrain.groupby(MyTrain.index).min().reset_index().date.dt.month.map(lambda x: MonthFactor[x])
GroupedMyTrain['GeomFactor']  = P /P[0] # make sure that those P values still come from the Geom calculation


# In[ ]:


RefVal = GroupedMyTrain.sales[0] / GroupedMyTrain['DoWFactor'][0] /GroupedMyTrain['MonthFactor'][0]
GroupedMyTrain['Predicted'] =           RefVal                        *     GroupedMyTrain['DoWFactor']   *     GroupedMyTrain['MonthFactor'] *     GroupedMyTrain['GeomFactor']


# In[ ]:


GroupedMyTrain[['Predicted','sales']].plot(figsize=(20,5),marker='.',markersize=2,ls=' ')


# In[ ]:


GroupedMyTrain['Delta'] = GroupedMyTrain['Predicted'] - GroupedMyTrain['sales']
GroupedMyTrain.plot(x='date',y='Delta',figsize=(20,5),marker='.',markersize=2,ls=' ',grid=True)


# In[ ]:


import calendar

def DiM(MyDate) :
    MyYear = MyDate.year 
    MyMonth = MyDate.month
    return calendar.monthrange(MyYear,MyMonth)[1]

GroupedMyTrain['year'] = GroupedMyTrain.date.dt.year
GroupedMyTrain['month'] = GroupedMyTrain.date.dt.month
GroupedMyTrain['DayInMonth'] = GroupedMyTrain.date.map(lambda x: DiM(x))
GroupedMyTrain['DiMFactor']  = 31 / GroupedMyTrain['DayInMonth'] 


GeomFactor2 = GroupedMyTrain.groupby('year').sum()['sales']
GeomFactor2 =  GeomFactor2/GeomFactor2[2013]

GroupedMyTrain['GeomFactor2'] = GroupedMyTrain.year.map(lambda x: GeomFactor2[x])


# In[ ]:


RefVal = GroupedMyTrain.sales[0] / GroupedMyTrain['DoWFactor'][0] /GroupedMyTrain['MonthFactor'][0]
GroupedMyTrain['Predicted'] =           RefVal                        *     GroupedMyTrain['DoWFactor']   *     GroupedMyTrain['MonthFactor'] *     GroupedMyTrain['GeomFactor2'] *     GroupedMyTrain['DiMFactor']


# In[ ]:


GroupedMyTrain['Delta'] = GroupedMyTrain['Predicted'] - GroupedMyTrain['sales']
GroupedMyTrain.plot(x='date',y='Delta',figsize=(20,5),marker='.',markersize=2,ls=' ')


# In[ ]:


GroupedMyTrain['DiMFactor'][(GroupedMyTrain.year==2016) & (GroupedMyTrain.month==2)] = 31.0 / 28
GroupedMyTrain['Predicted'] =          RefVal                        *    GroupedMyTrain['DoWFactor']   *    GroupedMyTrain['MonthFactor'] *    GroupedMyTrain['GeomFactor2'] *    GroupedMyTrain['DiMFactor']
GroupedMyTrain['Delta'] = GroupedMyTrain['Predicted'] - GroupedMyTrain['sales']
GroupedMyTrain.plot(x='date',y='Delta',figsize=(20,5),marker='.',markersize=2,ls=' ')



# In[ ]:



from scipy import stats
z,p = stats.normaltest(GroupedMyTrain['Delta'])
sns.distplot(GroupedMyTrain['Delta'])
plt.title("z={:4.2e}    p={:4.2e}".format(z,p))


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
Seasonal = MyTrain.groupby(MyTrain.index).sum()['sales']
SDW = seasonal_decompose(Seasonal, model='additive',freq=7)
SDA = seasonal_decompose(SDW.trend.dropna(), model='additive',freq=365)
fig = plt.figure()  
fig = SDW.plot()
fig = SDA.plot()

