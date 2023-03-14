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


import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn import preprocessing
import os #mod
import glob #mod
color = sns.color_palette()
import sys
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 999


# In[ ]:


os.getcwd() 


# In[ ]:


print(glob.glob("/kaggle/working/*.*"))


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)


# In[ ]:


#we create some new fields to easy manipulate
#forecasting probably should be at item-store because demand pattens could vary much dep. items and store 
train_df['weekday']=pd.DatetimeIndex(train_df['date']).weekday
train_df['month']=pd.DatetimeIndex(train_df['date']).month 
train_df['year']=pd.DatetimeIndex(train_df['date']).year
train_df['itemstore']=train_df.item.astype(str)+"-"+train_df.store.astype(str)


# In[ ]:


#overview of data
print("number of different items: %i" %(len(np.unique(train_df.item))))
print("number of different stores: %i" %(len(np.unique(train_df.store))))
print("number of different dates: %i" %(len(np.unique(train_df.date))))
print("maximun date in data: %s" %(max(train_df.date)))
print("minimum date in data: %s" %(min(train_df.date)))
print("number of different itemstore: %i" %(len(np.unique(train_df.itemstore))))


# In[ ]:


#plot values to see range 
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.sales.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('sales', fontsize=12)
plt.show()


# In[ ]:


#create some lists to see range of unique values
stores = list(set(train_df.store))
item = list(set(train_df.item))
itemstore = list(set(train_df.itemstore))


# In[ ]:


#we check anual sales profile comparing stores
c=train_df.groupby(['year','store']).sum()
plt.figure(figsize=(15,10))
d=c.unstack()
d.plot(y='sales')


# In[ ]:


#we check seasonal sales profile comparing stores
c=train_df.groupby(['month', 'store']).sum()
plt.figure(figsize=(15,10))
d=c.unstack()
d.plot(y='sales')


# In[ ]:


#we check seasonal sales profile comparing stores
c=train_df.groupby(['weekday', 'store']).sum()
plt.figure(figsize=(15,10))
d=c.unstack()
d.plot(y='sales')


# In[ ]:


#we evaluate increase in anual sales at itemstore level
b =train_df.drop(columns=['store', 'item','weekday','date','month'])
c=b.groupby(['year', 'itemstore']).sum()
d=c.unstack()
sales_itemstore_year=d.T
sales_itemstore_year['delta_2014/2013']=((sales_itemstore_year[2014]-sales_itemstore_year[2013])/sales_itemstore_year[2013])*100
sales_itemstore_year['delta_2015/2014']=((sales_itemstore_year[2015]-sales_itemstore_year[2014])/sales_itemstore_year[2014])*100
sales_itemstore_year['delta_2016/2015']=((sales_itemstore_year[2016]-sales_itemstore_year[2015])/sales_itemstore_year[2015])*100
sales_itemstore_year['delta_2017/2016']=((sales_itemstore_year[2017]-sales_itemstore_year[2016])/sales_itemstore_year[2016])*100
sales_itemstore_year_deltas =sales_itemstore_year.drop(columns=[2013, 2014, 2015, 2016, 2017], axis=1)


# In[ ]:


sales_itemstore_year_deltas =sales_itemstore_year.drop(columns=[2013, 2014, 2015, 2016, 2017], axis=1)


# In[ ]:


#heat-maps to compare deltas anual and bet. itemstore each year
sales_itemstore_year_deltas=sales_itemstore_year_deltas.sort_values('delta_2014/2013')
plt.figure(figsize=(8,10))
sns.heatmap(sales_itemstore_year_deltas)
plt.title("Percentage variation sales-itemstore. Sort 2014/2013", fontsize=15)
plt.show()


# In[ ]:


sales_itemstore_year_deltas=sales_itemstore_year_deltas.sort_values('delta_2015/2014')
plt.figure(figsize=(8,10))
sns.heatmap(sales_itemstore_year_deltas)
plt.title("Percentage variation sales-itemstore. Sort 2015/2014", fontsize=15)
plt.show()


# In[ ]:


sales_itemstore_year_deltas=sales_itemstore_year_deltas.sort_values('delta_2016/2015')
plt.figure(figsize=(8,10))
sns.heatmap(sales_itemstore_year_deltas)
plt.title("Percentage variation sales-itemstore. Sort 2016/2015", fontsize=15)
plt.show()


# In[ ]:


sales_itemstore_year_deltas=sales_itemstore_year_deltas.sort_values('delta_2017/2016')
plt.figure(figsize=(8,10))
sns.heatmap(sales_itemstore_year_deltas)
plt.title("Percentage variation sales-itemstore. Sort 2017/2016", fontsize=15)
plt.show()


# In[ ]:


#we pivot, group to weeks
train_df['date'] = pd.to_datetime(train_df['date'])
train_df_train=train_df.pivot(index='date', columns='itemstore', values='sales')
train_df_train=train_df_train.resample('W').sum()
train_df_train = train_df_train[:-1]


# In[ ]:


#we will forescat last 3 months so we keep a last test set to check weekly to dayly deaggregation
train_df_train_V1 = train_df_train[:'2017-09-30']
train_df_test_V1 = train_df_train['2017-10-01':]


# In[ ]:


len(train_df_train_V1[t]*0.7)


# In[ ]:


# we search ARIMA parameters for item 1 store 1 with 52 weeks differentation for stationary hip
import warnings
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas.core import datetools

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
# prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.7)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
    # difference data
        weeks_in_year = 52
        diff = difference(history, weeks_in_year)
        model = ARIMA(diff, order=arima_order)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        yhat = inverse_difference(history, yhat, weeks_in_year)
        predictions.append(yhat)
        history.append(test[t])
        # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# In[ ]:


#evaluate models
p_values = range(0, 6)
d_values = range(0, 2)
q_values = range(0, 6)
t = '1-1'

warnings.filterwarnings("ignore")

evaluate_models(train_df_train_V1[t].values, p_values, d_values, q_values)


# In[ ]:


t = '1-1'
X = train_df_train_V1[t].values
X = X.astype('float32')

# difference data
weeks_in_year = 52
diff = difference(X, weeks_in_year)
model = ARIMA(diff, order=(0,0,3))
model_fit = model.fit(trend='nc', disp=0)
# bias constant, could be calculated from in-sample mean residual
bias = -0

# save model
model_fit.save('model.pkl')
np.save('model_bias.npy', [bias])


# In[ ]:


# load and evaluate the finalized model on the validation dataset with a 3 month prediction
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt


# load and prepare datasets
X = train_df_train_V1[t].values.astype('float32')
history = [x for x in X]
weeks_in_year = 52
y = train_df_test_V1[t].values.astype('float32')

# load model
model_fit = ARIMAResults.load('model.pkl')
bias = np.load('model_bias.npy')

# multi-step out-of-sample forecast
predictions = list()
forecast = model_fit.forecast(steps=13)[0]
for yhat in forecast:
    yhat = bias + inverse_difference(history, yhat, weeks_in_year)
    history.append(yhat)
    predictions.append(yhat)
    print('>Predicted=%.3f' %(yhat))

# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)
plt.figure(figsize=(15,10))
plt.plot(y)
plt.plot(predictions, color='red')
plt.show()


# In[ ]:


#Calculate weekly pattern for each itemstore and apply to weekly prediction
train_df_test_V1=train_df_test_V1.reset_index()
predictions = pd.DataFrame(predictions)
train_df_test_V1_pred = pd.concat([train_df_test_V1['date'], train_df_test_V1[t], predictions], axis=1)
train_df_test_V1_pred['date'] = pd.to_datetime(train_df_test_V1_pred['date'])


# In[ ]:


train_df_test_V1_pred=train_df_test_V1_pred.set_index('date')
new_dates = pd.date_range('2017-10-01', '2017-12-31', name='date')
train_df_test_V1_pred_daily = train_df_test_V1_pred.reindex(new_dates, method='ffill')


# In[ ]:


train_df_test_V1_pred_daily


# In[ ]:


train_df = train_df.set_index('date')


# In[ ]:


#We predict at item-store and day level so we will de-compose week sales prediction for each itemstore into day sales prediction 
#for each itemstore

#First we asign in a dictionary for each item-store the de-composition of sales for SUN-MON-TUE.......-SAT-SUMA
dictionary_week_sales_itemstore={}
dictionary_week_sales_itemstore_reparto={}
for i in range (len(itemstore)):
    dictionary_week_sales_itemstore.update({itemstore[i]:[0, 0, 0, 0, 0, 0, 0, 0]})
    dictionary_week_sales_itemstore_reparto.update({itemstore[i]:[0, 0, 0, 0, 0, 0, 0, 0]})

#Now we group sales at item-store level and week-day    
#train_df=train_df.set_index('date')
train_sales_weekday=train_df[:'30-09-2017'].groupby(['weekday', 'itemstore']).sum()


# In[ ]:


#def update_dictionary_week_sales_itemstore(itemstore, train_sales_weekday)
for i in range (len(itemstore)):
    for j in range (0,7):
        dictionary_week_sales_itemstore[itemstore[i]][j]= train_sales_weekday.loc[(j, itemstore[i]),['sales']][0]
    dictionary_week_sales_itemstore[itemstore[i]][7]= sum(dictionary_week_sales_itemstore[itemstore[i]][0:7])   
    
#Now we update second dictionary dictionary_week_sales_itemstore_reparto={}
for i in range (len(itemstore)):
    for j in range (0,7):
        dictionary_week_sales_itemstore_reparto[itemstore[i]][j]= (dictionary_week_sales_itemstore[itemstore[i]][j]/               dictionary_week_sales_itemstore[itemstore[i]][7])


# In[ ]:


dictionary_week_sales_itemstore_reparto['1-1']


# In[ ]:


for i in range (13):
    for j in range (7):
        train_df_test_V1_pred_daily[0][(i*7)+j] = train_df_test_V1_pred_daily[0][(i*7)+j]*              dictionary_week_sales_itemstore_reparto[t][j]


# In[ ]:


train_df_test_V1_pred_daily[0]


# In[ ]:


train_df_test_V1_pred_daily[0]=round(train_df_test_V1_pred_daily[0])


# In[ ]:


train_df_test_V1_pred_daily[0]


# In[ ]:


train_df=train_df.reset_index()
train_df_2017=train_df.pivot(index='date', columns='itemstore', values='sales')
y_d = train_df_2017['1-1']['2017-10-01':].values


# In[ ]:


predictions_d = train_df_test_V1_pred_daily[0].values


# In[ ]:


predictions_d[91]=21


# In[ ]:


# report performance
rmse = sqrt(mean_squared_error(y_d, predictions_d))
print('RMSE: %.3f' % rmse)
plt.figure(figsize=(15,10))
plt.plot(y_d)
plt.plot(predictions_d, color='red')
plt.show()


# In[ ]:




