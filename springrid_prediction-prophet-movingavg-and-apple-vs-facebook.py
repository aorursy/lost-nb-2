#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import re
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime
from collections import Counter
from scipy.fftpack import fft

from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA
import warnings

kaggle_on = True

if kaggle_on:
    path = '../input/'
else:
    path = 'data/'

df_train = pd.read_csv(path + 'train_1.csv', nrows=150000).fillna(0)
print('Len of data: ', len(df_train.index))


# In[2]:


# Convert page views to integers
for col in df_train.columns[1:]:
    df_train[col] = pd.to_numeric(df_train[col], downcast='integer')


# In[3]:


# Get the language of an article
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        return res.group(0)[0:2]
    return 'na'

df_train['lang'] = df_train.Page.map(get_language)
languages = df_train.lang.unique()
print(Counter(df_train.lang))


# In[4]:


# Analyze language feature
lang_sets = {}
for language in languages:
    lang_sets[language] = df_train[df_train.lang == language].iloc[:, 0:-1]

sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:, 1:].sum(axis=0) / lang_sets[key].shape[0]
    
days = [r for r in range(sums['fr'].shape[0])]
fig = plt.figure(1, figsize=[10, 10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels = {'en': 'English', 'ja': 'Japanese', 'de': 'German',
          'na': 'Media', 'fr': 'French', 'zh': 'Chinese',
          'ru': 'Russian', 'es': 'Spanish'}
for key in sums:
    plt.plot(days, sums[key], label=labels[key])
plt.legend()
plt.show()


# In[5]:


def plot_with_fft(key):
    fig = plt.figure(1, figsize=[15, 5])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.title(labels[key])
    plt.plot(days, sums[key], label=labels[key])

    fig = plt.figure(2, figsize=[15,5])
    fft_complex = fft(sums[key])
    fft_mag = [np.sqrt(np.real(x)*np.real(x)+np.imag(x)*np.imag(x)) for x in fft_complex]
    fft_xvals = [day / days[-1] for day in days]
    npts = len(fft_xvals) // 2 + 1
    fft_mag = fft_mag[:npts]
    fft_xvals = fft_xvals[:npts]

    plt.ylabel('FFT Magnitude')
    plt.xlabel(r"Frequency [days]$^{-1}$")
    plt.title('Fourier Transform')
    plt.plot(fft_xvals[1:], fft_mag[1:], label=labels[key])
    # Draw lines at 1, 1/2, and 1/3 week periods
    plt.axvline(x=1./7, color='red', alpha=0.3)
    plt.axvline(x=2./7, color='red', alpha=0.3)
    plt.axvline(x=3./7, color='red', alpha=0.3)

    plt.show()

for key in sums:
    plot_with_fft(key)


# In[6]:


# For each language get highest few pages
npages = 5
top_pages = {}
for key in lang_sets:
    # print(key)
    sum_set = pd.DataFrame(lang_sets[key][['Page']])
    sum_set['total'] = lang_sets[key].sum(axis=1)
    sum_set = sum_set.sort_values('total',ascending=False)
    # print(sum_set.head(10))
    top_pages[key] = sum_set.index[0]
    # print('\n\n')


# In[7]:


cols = df_train.columns[1:-1]

def filter_df(df, word):
    df_new = df[df['Page'].str.contains(word)]
    apple_pages = df_new.Page.values
    df_new = df_new[cols].transpose()
    df_new[word] = df_new.values.sum(axis=1)
    return df_new[[word]]

word_to_filter_by = ['Apple_Inc', 'Microsoft', 'Facebook', 'Google']

df_companies = pd.DataFrame()
for word in word_to_filter_by:
    df_tmp = filter_df(df_train, word)
    df_companies = pd.concat([df_companies, df_tmp], axis=1)

print(df_companies.idxmax(axis=0))
df_companies.plot()

# mark Apple releases and other important dates during time period
if False:
    holidays = ['2015-11-26', '2015-12-25']
    stock_dates = ['2016-08-10']
    apple_dates = ['2015-07-15', '2015-09-09', '2015-09-25', '2015-10-13', '2015-10-26', '2015-11-11',
                  '2016-03-31', '2016-04-19', '2016-09-16', '2016-10-27', '2016-12-19']
    for date in holidays + stock_dates:
        plt.axvline(df_companies.index.get_loc(date), color='black', linestyle='solid')


# In[8]:


## 3 Predictive Modelling - Apple vs. Others
We will try a few different simple models to see how they perform.

### 3.1 Moving Average
Simple moving average approach over one week.


# In[9]:


def moving_average_approach(df):
    moving_avg = df.rolling(window=7).mean()
    # + df.rolling(window=28).mean()/2 # + df.rolling(window=56).mean()/4
    # moving_std = df.rolling(window=3).std()
    return moving_avg

moving_average_approach(df_companies).plot()
df_companies.plot()


# In[10]:


# https://github.com/facebookincubator/prophet/issues/223
# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


# In[11]:


from fbprophet import Prophet
sns.set(font_scale=1) 

def make_forecast_with_prophet(df, cols):
    values = df.values
    df_prophet = pd.DataFrame(columns=['ds', 'y'])
    df_prophet['ds'] = cols
    df_prophet = df_prophet.set_index('ds')
    df_prophet['y'] = values
    df_prophet.reset_index(drop=False,inplace=True)
    

    m = Prophet(yearly_seasonality=True).fit(df_prophet)
    future = m.make_future_dataframe(periods=days_to_forecast,freq='D', include_history=True)
    forecast = m.predict(future)
    return forecast, m

if False:
    days_to_forecast = 31+28
    plot_on = False

    cols = df_train.columns[1:-1]
    for key in top_pages[0]:
        df_tmp = df_train.loc[top_pages[key], cols].copy()
        forecast, m = make_forecast_with_prophet(df_tmp)

        if plot_on:
            plt.figure(figsize=(10, 10))
            fig = m.plot(forecast)
            fig = m.plot_components(forecast)


# In[12]:


from statsmodels.tsa.arima_model import ARIMA
import warnings

n_cols = len(df_train.columns) - 1
n_cols_train = round(n_cols / 10*7)
cols_train = df_train.columns[1:n_cols_train]
cols_predict = df_train.columns[n_cols_train:-1]

for key in top_pages:
    data = np.array(df_train.loc[top_pages[key], df_train.columns[1:-1]],'f')
    data_train = np.array(df_train.loc[top_pages[key], cols_train],'f')
    data_predict = np.array(df_train.loc[top_pages[key], cols_predict],'f')
    result = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            arima = ARIMA(data_train, [2,1,4])
            result = arima.fit(disp=False)
        except:
            try:
                arima = ARIMA(data_train, [2,1,2])
                result = arima.fit(disp=False)
            except:
                print(df_train.loc[top_pages[key],'Page'])
                print('\tARIMA failed')
    stop = 599
    pred = result.forecast(steps=100)[0]
    x = [i for i in range(len(data))]
    x_train = [i for i in range(n_cols_train-1)]
    x_pred = [i for i in range(n_cols_train, n_cols_train+100)]
    i=0
    
    plt.plot(x, data, label='Data')
    plt.plot(x_train, data_train, label='Train Data')
    plt.plot(x_pred, pred,label='ARIMA Model')
    plt.title(df_train.loc[top_pages[key], 'Page'])
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.legend()
    plt.show()


# In[13]:


def batch_process_with_prophet(df):
    submission = pd.DataFrame(columns=['Id', 'Visits'])
    results_forecasts = pd.DataFrame(columns=['ds', 'yhat'])

    cols = df.columns[1:-1]
    # i = 0

    # Comment out this to be able to run the rest :)
    for index, row in df.iterrows():
        df_tmp = df.loc[index, cols].copy()

        # Workaround for handle pages where all visists are zero.
        if df_tmp.sum() == 0:
            dates_index = pd.date_range(start="2017-01-01", end="2017-02-28", freq="D")
            forecast = pd.DataFrame(columns=['ds', 'yhat'])
            forecast['ds'] = dates_index
            forecast = forecast.set_index(forecast.ds)
            forecast['yhat'] = 0
        else:
            forecast, m = make_forecast_with_prophet(df_tmp, cols)
            forecast = forecast[['ds', 'yhat']]
            
        forecast['ds'] = row.Page + '_' + forecast.ds.apply(lambda x: x.strftime('%Y-%m-%d'))
        results_forecasts = results_forecasts.append(forecast.tail(days_to_forecast))
        
        # if i % 10 == 0:
        #     print(i)
        # i += 1
        
    return results_forecasts


# In[14]:


CHUNKSIZE = 100

df_key = pd.read_csv(path + 'key_1.csv')
reader = pd.read_csv(path + 'train_1.csv", chunksize=CHUNKSIZE)
                     
pool = mp.Pool(4) # use 4 processes

funclist = []
i = 0

for df in reader:
    # process each data frame
    f = pool.apply_async(batch_process_with_prophet,[df])
    funclist.append(f)
    i += 1
    if i > 10:
        break

result = []
with suppress_stdout_stderr():
    for f in funclist:
        result.append(f.get(timeout=60*60)) # timeout in 300 seconds = 60 mins

# combine chunks with transformed data into a single training set
training = pd.concat(result)

training.to_csv('sub.csv')

if False:
    results_forecasts = training.sort_values('ds')
    df_key = df_key.sort_values('Page')
    submission['Id'] = df_key.Id.values
    submission['Visits'] = results_forecasts.yhat.values
    submission.to_csv('submissions/' + datetime.now().strftime("%Y%m%d-%I%M%p") + '.csv')

print(training.head())
print(training.describe())


# In[15]:


# from: https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/37232
from numba import jit
import math

@jit
def smape_fast(y_true, y_pred):
    out = 0
    for i in range(y_true.shape[0]):
        a = y_true[i]
        b = y_pred[i]
        c = a+b
        if c == 0:
            continue
        out += math.fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out


# In[16]:




