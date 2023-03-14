#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # to separate pages based on language (regular expression)
import matplotlib.pyplot as plt # to visualize data
from pandas.tools.plotting import autocorrelation_plot # to visualize and configure the parameters of ARIMA model
from statsmodels.tsa.arima_model import ARIMA # to make an ARIMA model that fits the data


# In[2]:


train_df = pd.read_csv('../input/train_1.csv').fillna(0)
train_df.head()


# In[3]:


train_df.info()


# In[4]:


def find_language(url):
    res = re.search('[a-z][a-z].wikipedia.org',url)
    if res:
        return res[0][0:2]
    return 'na'

train_df['lang'] = train_df.Page.map(find_language)


# In[5]:


lang_sets = {}
lang_sets['en'] = train_df[train_df.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = train_df[train_df.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = train_df[train_df.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = train_df[train_df.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = train_df[train_df.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = train_df[train_df.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = train_df[train_df.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = train_df[train_df.lang=='es'].iloc[:,0:-1]

sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]


# In[6]:


days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1,figsize=[10,10])
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


# In[7]:


from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

for key in sums:
    fig = plt.figure(1,figsize=[10,5])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    data = np.array(sums[key])
    autocorr = acf(data)
    pac = pacf(data)

    x = [x for x in range(len(pac))]
    ax1.plot(x[1:],autocorr[1:])

    ax2.plot(x[1:],pac[1:])
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')

    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Autocorrelation')
    print(key)
    plt.show()


# In[8]:


params = {'en': [4,1,0], 'ja': [7,1,1], 'de': [7,1,1], 'na': [4,1,0], 'fr': [4,1,0], 'zh': [7,1,1], 'ru': [4,1,0], 'es': [7,1,1]}

for key in sums:
    data = np.array(sums[key])
    result = None
    arima = ARIMA(data,params[key])
    result = arima.fit(disp=False)
    #print(result.params)
    pred = result.predict(2,599,typ='levels')
    x = [i for i in range(600)]
    i=0
    
    print(key)
    plt.plot(x[2:len(data)],data[2:] ,label='Data')
    plt.plot(x[2:],pred,label='ARIMA Model')
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.legend()
    plt.show()


# In[9]:


train_df.head()


# In[10]:


train_df = train_df.drop('Page',axis = 1)
train_df.shape


# In[11]:


#Packages for pre processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

 # Importing the Keras libraries and packages for LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[12]:


for key in sums:
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


   #Training LSTM

   #Reshaping Array
   X_train = np.reshape(X_train, (384,1,1))

   # Initialising the RNN
   regressor = Sequential()

   # Adding the input layerand the LSTM layer
   regressor.add(LSTM(units = 8, activation = 'relu', input_shape = (None, 1)))


   # Adding the output layer
   regressor.add(Dense(units = 1))

   # Compiling the RNN
   regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

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
   plt.figure
   plt.plot(y, color = 'red', label = 'Real Web View')
   plt.plot(y_pred, color = 'blue', label = 'Predicted Web View')
   plt.title('Web View Forecasting')
   plt.xlabel('Number of Days from Start')
   plt.ylabel('Web View')
   plt.legend()
   plt.show()

