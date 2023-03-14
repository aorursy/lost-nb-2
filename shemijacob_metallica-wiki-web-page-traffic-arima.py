#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import statsmodels.api as sm

import matplotlib.pyplot as plt # plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # plotting

import warnings
warnings.filterwarnings("ignore")


# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
     for filename in filenames:
         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.


# In[2]:


base_url = '/kaggle/input/web-traffic-time-series-forecasting/'

train_1 = pd.read_csv(base_url+'train_1.csv')
# train_2 = pd.read_csv(base_url+'train_2.csv')


# In[3]:


train_1.shape


# In[4]:


train_1.head()


# In[5]:


trainT = train_1.drop('Page', axis=1).T
trainT.columns = train_1.Page.values
trainT.head()


# In[6]:


metallica = pd.DataFrame(trainT['Metallica_es.wikipedia.org_all-access_all-agents'])
metallica.head()


# In[7]:


print (metallica.shape)


# In[8]:



print (metallica.isnull().sum())


# In[9]:


metallica.plot(figsize=(15, 6))
plt.show()


# In[10]:


def statistics(x): 
    # Determining rolling statistics
    rolmean = x.rolling(window=22, center=False).mean()
    rolstd = x.rolling(window=12, center=False).std() 
    
    # Plot rolling statistics
    orig = plt.plot(x.values, color='blue', label='Original') 
    mean = plt.plot(rolmean.values, color='red', label='Rolling Mean') 
    std = plt.plot(rolstd.values, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


# In[11]:


statistics(metallica)


# In[12]:


metallica.index  = pd.to_datetime(metallica.index)
metallica.tail()


# In[13]:


from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(metallica)

adf_test

print "ADF = " + str(adf_test[0])
print "p-value = " +str(adf_test[1])


# In[14]:


metallica_log = np.log1p(metallica)
plt.plot(metallica_log.values, color='green')
plt.show()


# In[15]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose( metallica_log.values, model="multiplicative", freq=7 )
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.title('Observed = Trend + Seasonality + Residuals')
plt.plot( metallica_log.values, label='Observed' )
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='seasonal')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='residual')
plt.legend(loc='best')

plt.tight_layout()
plt.show()


# In[16]:


metallica_log_diff = metallica_log - metallica_log.shift()
plt.plot( metallica_log_diff.values )
plt.show()


# In[17]:


metallica_log_diff.tail()


# In[18]:


metallica_log_diff.dropna(inplace=True)
statistics(metallica_log_diff)


# In[19]:


from statsmodels.tsa.stattools import acf, pacf 
from statsmodels.tsa.arima_model import ARIMA
import itertools


# In[20]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[21]:


results_aic_min = 10000

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(metallica_log,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            
            if results.aic < results_aic_min:
                results_aic_min = results.aic

            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            
                      
        except:
            continue

print(results_aic_min)


# In[22]:


mod = sm.tsa.statespace.SARIMAX(metallica_log,
                                order=(1, 1, 1),
                                seasonal_order=(0, 0, 0, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[23]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[24]:


metallica_log.index  = pd.to_datetime(metallica_log.index)
metallica_log.tail()


# In[25]:


pred = results.get_prediction(start=pd.to_datetime('2016-10-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = metallica_log['2015':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Metallica Wikipedia Article Visits')
plt.legend()
plt.show()


# In[26]:


y_forecasted = pred.predicted_mean
y_truth = metallica_log['2016-10-31':]
print(y_forecasted.shape, y_truth.iloc[:,0].shape)


# In[27]:


mse = ((y_forecasted - y_truth.iloc[:,0]) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[28]:


pred_uc = results.get_forecast(steps=60)
pred_ci = pred_uc.conf_int()
ax = metallica_log.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()


# In[ ]:




