#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Some basic libraries 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv('./input/train_1.csv').fillna(0)
df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group(0)[0:2]
    return 'na'


# In[ ]:


df_train['lang'] = df_train.Page.map(get_language)


# In[ ]:


df_train[df_train.Page == 'Noël_fr.wikipedia.org_all-access_all-agents']


# In[ ]:


lang_sets = {}
lang_sets['en'] = df_train[df_train.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = df_train[df_train.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = df_train[df_train.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = df_train[df_train.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = df_train[df_train.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = df_train[df_train.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = df_train[df_train.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = df_train[df_train.lang=='es'].iloc[:,0:-1]

sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]


# In[ ]:


days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1,figsize=[18,5])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels={'en':'English','ja':'Japanese','de':'German',
        'na':'Media','fr':'French','zh':'Chinese',
        'ru':'Russian','es':'Spanish'
       }

for key in sums:
    plt.plot(days,sums[key],label = labels[key] )
    
plt.legend()
plt.show()


# In[ ]:


from scipy.fftpack import fft
def plot_with_fft(key):
    f, ax = plt.subplots(2, figsize=(18,8))
    ax[0].set_ylabel('Views per Page')
    ax[0].set_xlabel('Day')
    ax[0].set_title(labels[key])
    ax[0].plot(days,sums[key],label = labels[key] )

    fft_complex = fft(sums[key])
    fft_mag = [np.sqrt(np.real(x)*np.real(x)+np.imag(x)*np.imag(x)) for x in fft_complex]
    fft_xvals = [day / float(days[-1]) for day in days]
    npts = len(fft_xvals) // 2 + 1
    fft_mag = fft_mag[:npts]
    fft_xvals = fft_xvals[:npts]
        
    ax[1].set_ylabel('FFT Magnitude')
    ax[1].set_xlabel(r"Frequency [days]$^{-1}$")
    ax[1].set_title('Fourier Transform')
    ax[1].plot(fft_xvals[1:],fft_mag[1:],label = labels[key] )
    ax[1].axvline(x=1./7,color='red',alpha=0.3)
    ax[1].axvline(x=2./7,color='red',alpha=0.3)
    ax[1].axvline(x=3./7,color='red',alpha=0.3)

    plt.show()

for key in sums:
    plot_with_fft(key)


# In[ ]:


def plot_entry(key,idx, ax):
    data = lang_sets[key].iloc[idx,1:]
    ax.plot(days,data, label=df_train.iloc[lang_sets[key].index[idx],0])
    ax.set_xlabel('day')
    ax.set_ylabel('views')


# In[ ]:


f, ax = plt.subplots(figsize=(18,4))
plot_entry(key='en', idx=4, ax=ax )
plot_entry(key='en', idx=5, ax=ax )
plot_entry(key='en', idx=6, ax=ax )
plot_entry(key='en', idx=7, ax=ax )
plot_entry(key='en', idx=8, ax=ax )
ax.legend()


# In[ ]:


npages = 5
top_pages = {}
for key in lang_sets:
    print(key)
    sum_set = pd.DataFrame(lang_sets[key][['Page']])
    sum_set['total'] = lang_sets[key].sum(axis=1)
    sum_set = sum_set.sort_values('total',ascending=False)
    print(sum_set.head(10))
    top_pages[key] = sum_set.index[0]
    print('\n\n')


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA 


# In[ ]:


import warnings

cols = df_train.columns[1:-1]
for key in top_pages:
    data = np.array(df_train.loc[top_pages[key],cols],'f')
    result = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            arima = ARIMA(data,[2,1,4])
            result = arima.fit(disp=False)
        except:
            try:
                arima = ARIMA(data,[2,1,2])
                result = arima.fit(disp=False)
            except:
                print(df_train.loc[top_pages[key],'Page'])
                print('\tARIMA failed')
    #print(result.params)
    pred = result.predict(2,599,typ='levels')
    x = [i for i in range(600)]
    i=0
    
    f, ax = plt.subplots(figsize=(18,4))
    ax.plot(x[2:len(data)],data[2:] ,label='Data')
    ax.plot(x[2:],pred,label='ARIMA Model')
    print str(df_train.loc[top_pages[key],'Page'])
    ax.set_title(str(df_train.loc[top_pages[key],'Page']).decode('utf-8'))
    ax.set_xlabel('Days')
    ax.set_ylabel('Views')
    ax.legend()
    plt.show()


# In[ ]:


df_train = df_train.drop('Page',axis = 1)
df_train.head()


# In[ ]:


#Packages for pre processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

 # Importing the Keras libraries and packages for LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[ ]:


def keras_regressor():
    regressor = Sequential()
    # Adding the input layerand the LSTM layer
    regressor.add(LSTM(units = 8, activation = 'relu', input_shape = (None, 1)))
    # Adding the output layer
    regressor.add(Dense(units = 1))
    # Compiling the RNN
    regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
    return regressor

for key in sums:
    f, ax = plt.subplots(figsize=(18,4))
    
    row = [0]*sums[key].shape[0]
    for i in range(sums[key].shape[0]):
        row[i] = sums[key][i]

    #Using Data From Random Row for Training and Testing
    X = row[0:549]
    y = row[1:550]
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    # Feature Scaling
    sc = MinMaxScaler()
    X_train = np.reshape(X_train,(-1,1))
    y_train = np.reshape(y_train,(-1,1))
    X_train = sc.fit_transform(X_train)
    y_train = sc.fit_transform(y_train)
    #Reshaping Array
    X_train = np.reshape(X_train, (384,1,1))

    # Initialising the RNN
    regressor = keras_regressor()

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose = 0)

    # Getting the predicted Web View
    inputs = X
    inputs = np.reshape(inputs,(-1,1))
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (549,1,1))
    y_pred = regressor.predict(inputs)
    y_pred = sc.inverse_transform(y_pred)

    print(key)
    #Visualising Result
    ax.plot(y, color = 'red', label = 'Real Web View')
    ax.plot(y_pred, color = 'blue', label = 'Predicted Web View')
    ax.set_title ('Web View Forecasting: %s' % key)
    ax.set_xlabel('Number of Days from Start')
    ax.set_ylabel('Web View')
    ax.legend()
    plt.show()

