#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.simplefilter('ignore')


# In[2]:


train=pd.read_csv(r"../input/covid19-global-forecasting-week-4/train.csv")
test=pd.read_csv(r"../input/covid19-global-forecasting-week-4/test.csv")


# In[3]:


train.sample(6)


# In[4]:


test.sample(6)


# In[5]:


df = train.fillna('NA').groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum()                           .groupby(['Country_Region','Province_State']).max().sort_values()                           .groupby(['Country_Region']).sum().sort_values(ascending = False)

top10 = pd.DataFrame(df).head(10)
top10


# In[6]:


df_by_date = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'India') &(df_by_date.Date >= '2020-03-01')].sort_values('ConfirmedCases',ascending = False), 
             x='Date', y='ConfirmedCases', color="ConfirmedCases")
fig.update_layout(title_text='Confirmed COVID-19 cases per day in India')
fig.show()


# In[7]:


df_by_date = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'US') &(df_by_date.Date >= '2020-03-01')].sort_values('ConfirmedCases',ascending = False), 
             x='Date', y='ConfirmedCases', color="ConfirmedCases")
fig.update_layout(title_text='Confirmed COVID-19 cases per day in US')
fig.show()


# In[8]:


df_by_date = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'China') &(df_by_date.Date >= '2020-01-01')].sort_values('ConfirmedCases',ascending = False), 
             x='Date', y='ConfirmedCases', color="ConfirmedCases")
fig.update_layout(title_text='Confirmed COVID-19 cases per day in China')
fig.show()


# In[9]:


df_by_date = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'Spain') &(df_by_date.Date >= '2020-03-01')].sort_values('ConfirmedCases',ascending = False), 
             x='Date', y='ConfirmedCases', color="ConfirmedCases")
fig.update_layout(title_text='Confirmed COVID-19 cases per day in Spain')
fig.show()


# In[10]:


df_by_date = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'Germany') &(df_by_date.Date >= '2020-03-01')].sort_values('ConfirmedCases',ascending = False), 
             x='Date', y='ConfirmedCases', color="ConfirmedCases")
fig.update_layout(title_text='Confirmed COVID-19 cases per day in Germany')
fig.show()


# In[11]:


df=train.groupby(['Date','Country_Region']).agg('sum').reset_index()
df.tail(5)


# In[12]:


def pltCountry_cases(ConfirmedCases,*argv):
    f, ax=plt.subplots(figsize=(16,5))
    labels=argv
    for a in argv: 
        country=df.loc[(df['Country_Region']==a)]
        plt.plot(country['Date'],country['ConfirmedCases'],linewidth=3)
        plt.xticks(rotation=40)
        plt.legend(labels)
        ax.set(title='Evolution of the number of cases' )


# In[13]:


def pltCountry_fatalities(Fatalities,*argv):
    f, ax=plt.subplots(figsize=(16,5))
    labels=argv
    for a in argv: 
        country=df.loc[(df['Country_Region']==a)]
        plt.plot(country['Date'],country['Fatalities'],linewidth=3)
        plt.xticks(rotation=40)
        plt.legend(labels)
        ax.set(title='Evolution of the number of fatalities' )


# In[14]:


pltCountry_cases('ConfirmedCases','India')
pltCountry_fatalities('Fatalities','India')


# In[15]:


pltCountry_cases('ConfirmedCases', 'Germany','Spain','China','US')
pltCountry_fatalities('Fatilities','Germany','Spain','China','US')


# In[16]:


test['Date'] = pd.to_datetime(test['Date'])
train['Date'] = pd.to_datetime(train['Date'])


# In[17]:


case='ConfirmedCases'
def timeCompare(time,*argv):
    Coun1=argv[0]
    Coun2=argv[1]
    f,ax=plt.subplots(figsize=(16,5))
    labels=argv  
    country=df.loc[(df['Country_Region']==Coun1)]
    plt.plot(country['Date'],country[case],linewidth=2)
    plt.xticks([])
    plt.legend(labels)
    ax.set(title=' Evolution of actual cases',ylabel='Number of cases' )

    country2=df.loc[df['Country_Region']==Coun2]
    plt.plot(country2['Date'],country2[case],linewidth=2)
    plt.legend(labels)
    ax.set(title=' Cases in India Vs Cases in %s '%argv[1] ,ylabel='Number of %s cases'%case, xlabel='Time' )


# In[18]:


timeCompare(7,'India','China')
timeCompare(7,'India','Spain')
timeCompare(7,'India','Germany')
timeCompare(7,'India','US')


# In[19]:


case='Fatalities'
def timeCompare_f(time,*argv):
    Coun1=argv[0]
    Coun2=argv[1]
    f,ax=plt.subplots(figsize=(16,5))
    labels=argv  
    country=df.loc[(df['Country_Region']==Coun1)]
    plt.plot(country['Date'],country[case],linewidth=2)
    plt.xticks([])
    plt.legend(labels)
    ax.set(title=' Evolution of actual cases',ylabel='Number of cases' )

    country2=df.loc[df['Country_Region']==Coun2]
    #country2['Date']=country2['Date']-datetime.timedelta(days=time)
    plt.plot(country2['Date'],country2[case],linewidth=2)
    #plt.xticks([])
    plt.legend(labels)
    ax.set(title=' Fatalities in India Vs Fatalities in %s '%argv[1] ,ylabel='Number of %s cases'%case, xlabel='Time' )


# In[20]:


timeCompare_f(7,'India','Spain')
timeCompare_f(7,'India','Germany')
timeCompare_f(7,'India','US')
timeCompare_f(7,'India','China')


# In[21]:


def roll(country,case='ConfirmedCases'):
    ts=df.loc[(df['Country_Region']==country)]  
    ts=ts[['Date',case]]
    ts=ts.set_index('Date')
    ts.astype('int64')
    a=len(ts.loc[(ts['ConfirmedCases']>=10)])
    ts=ts[-a:]
    return (ts.rolling(window=4,center=False).mean().dropna())


def rollPlot(country, case='ConfirmedCases'):
    ts=df.loc[(df['Country_Region']==country)]  
    ts=ts[['Date',case]]
    ts=ts.set_index('Date')
    ts.astype('int64')
    a=len(ts.loc[(ts['ConfirmedCases']>=10)])
    ts=ts[-a:]
    plt.figure(figsize=(16,6))
    plt.plot(ts.rolling(window=7,center=False).mean().dropna(),label='Rolling Mean')
    plt.plot(ts[case])
    plt.plot(ts.rolling(window=7,center=False).std(),label='Rolling std')
    plt.legend()
    plt.title('Cases distribution in %s with rolling mean and standard' %country)
    plt.xticks([])


# In[22]:


tsC1=roll('China')
rollPlot('China')


# In[23]:


tsC2=roll('US')
rollPlot('US')


# In[24]:


tsC3=roll('Italy')
rollPlot('Italy')


# In[25]:


tsC4=roll('Spain')
rollPlot('Spain')


# In[26]:


tsC5=roll('Germany')
rollPlot('Germany')


# In[27]:


tsC6=roll('India')
rollPlot('India')


# In[28]:


fig=sm.tsa.seasonal_decompose(tsC1.values,freq=7).plot()


# In[29]:


fig=sm.tsa.seasonal_decompose(tsC2.values,freq=7).plot()


# In[30]:


fig=sm.tsa.seasonal_decompose(tsC3.values,freq=7).plot()


# In[31]:


fig=sm.tsa.seasonal_decompose(tsC4.values,freq=7).plot()


# In[32]:


fig=sm.tsa.seasonal_decompose(tsC5.values,freq=7).plot()


# In[33]:


fig=sm.tsa.seasonal_decompose(tsC6.values,freq=7).plot()


# In[34]:


def stationarity(ts):
    print('Results of Dickey-Fuller Test:')
    test = adfuller(ts, autolag='AIC')
    results = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for i,val in test[4].items():
        results['Critical Value (%s)'%i] = val
    print (results)

#For China
tsC=tsC1['ConfirmedCases'].values
stationarity(tsC)


# In[35]:


tsC


# In[36]:


#For US
tsC=tsC2['ConfirmedCases'].values
stationarity(tsC)


# In[37]:


#For Italy
tsC=tsC3['ConfirmedCases'].values
stationarity(tsC)


# In[38]:


#For Spain
tsC=tsC4['ConfirmedCases'].values
stationarity(tsC)


# In[39]:


#For Germany
tsC=tsC5['ConfirmedCases'].values
stationarity(tsC)


# In[40]:


#For INdia
tsC=tsC6['ConfirmedCases'].values
stationarity(tsC)


# In[41]:


tsC7=tsC6['ConfirmedCases'].values


# In[42]:


def corr(ts):
    plot_acf(ts,lags=12,title="ACF")
    plot_pacf(ts,lags=12,title="PACF")
    

#For China
corr(tsC1)


# In[43]:


#For US
corr(tsC2)


# In[44]:


#For Italy
corr(tsC3)


# In[45]:


#For Spain
corr(tsC4)


# In[46]:


#For Germany
corr(tsC5)


# In[47]:


#For India
corr(tsC6)


# In[48]:


#test['Date'] = pd.to_datetime(test['Date'])
#train['Date'] = pd.to_datetime(train['Date'])
train = train.set_index(['Date'])
test = test.set_index(['Date'])


# In[ ]:





# In[49]:


train.shape


# In[50]:



def create_features(df,label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['Date'] = df.index
    df['hour'] = df['Date'].dt.hour
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
   
    return X


# In[51]:


train_features=pd.DataFrame(create_features(train))
test_features=pd.DataFrame(create_features(test))
features_and_target_train = pd.concat([train,train_features], axis=1)
features_and_target_test = pd.concat([test,test_features], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def FunLabelEncoder(df):
    for c in df.columns:
        if df.dtypes[c] == object:
            le.fit(df[c].astype(str))
            df[c] = le.transform(df[c].astype(str))
    return df
features_and_target_train= FunLabelEncoder(features_and_target_train)


# In[52]:


x_train= features_and_target_train[['Country_Region','month', 'dayofyear', 'dayofmonth' , 'weekofyear']]
y1 = features_and_target_train[['ConfirmedCases']]
y2 =features_and_target_train[['Fatalities']]
x_test = features_and_target_test[['Country_Region', 'month', 'dayofyear', 'dayofmonth' , 'weekofyear']]


# In[53]:


tsC


# In[54]:


dd2=pd.DataFrame(columns=['MAPE','MSE','RMSE'])
dd2


# In[55]:


#Mean absolute percentage error

from sklearn.metrics import r2_score
def r2score(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return r2_score(y1, y_pred)

def mape(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100
def mse(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean((y1 - y_pred)**2)

def rmse(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.sqrt(np.mean((y1 - y_pred)**2))




def split(ts):
    size = int(len(ts) * 0.85)
    train= ts[:size]
    test = ts[size:]
    return(train,test)

def arima(ts,test):
    p=d=q=range(0,6)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(ts, order=var)
            result = model.fit()

            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
            
    #Modeling
    model = ARIMA(ts, order=param)
    result = model.fit()
    result.plot_predict(start=int(len(ts) * 0.7), end=int(len(ts) * 1.2))
    pred=result.forecast(steps=len(test))[0]
    #Plotting results
    f,ax=plt.subplots()
    plt.plot(pred,c='green', label= 'predictions')
    plt.plot(test, c='red',label='real values')
    plt.legend()
    plt.title('True vs predicted values')
    #Printing the error metrics
    print(result.summary())        
    
    print('\nMean absolute percentage error: %f'%mape(test,pred))
    print("Mean Square Error error using ARIMA: ",mse(test,pred))
    print("Root Mean Square error using ARIMA: ",rmse(test,pred))
    print("R2 Score  using ARIMA: ",r2score(test,pred))
    lis=[mape(test,pred),mse(test,pred), rmse(test,pred)]
    dd2=pd.DataFrame({'MAPE': [mape(test,pred)], 'MSE': [mse(test,pred)], 'RMSE': [rmse(test,pred)]})
    return (pred,dd2)

train,test=split(tsC)
pred,dd2=arima(train,test)


# In[56]:


dd2.rename({0:'ARIMA'})


# In[57]:


from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 5
n_features = 1
NtsC=np.reshape(train,(-1,1))
test2=np.reshape(test,(-1,1))

generator = TimeseriesGenerator(NtsC, NtsC, length=n_input, batch_size=1)


# In[58]:


test2.shape


# In[59]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[60]:


model.summary()


# In[61]:


model.fit_generator(generator,epochs=25)
loss_per_epoch = model.history.history['loss']
fig = plt.figure(dpi = 120,figsize = (8,4))
ax = plt.axes()
ax.set(xlabel = 'Number of Epochs',ylabel = 'MSE Loss',title = 'Loss Curve of RNN LSTM')
plt.plot(range(len(loss_per_epoch)),loss_per_epoch,lw = 2)


# In[62]:


test_predictions = []

first_eval_batch = NtsC[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test2)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)     
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[63]:


test_predictions=np.ravel(test_predictions)


# In[64]:


pred=pd.DataFrame(columns={"current","prediction"})


# In[65]:


pred["current"]=test
pred["prediction"]=test_predictions


# In[66]:


pred


# In[67]:


dd3=pd.DataFrame(columns=['MAPE','MSE','RMSE'])
dd3


# In[68]:


def mape(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100

def mse(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean((y1 - y_pred)**2)

def rmse(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.sqrt(np.mean((y1 - y_pred)**2))
def r2score(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return r2_score(y1, y_pred)

print("Mean absolute % error using LSTM: ",mape(pred["current"],pred["prediction"]))
print("Mean Square Error error using LSTM: ",mse(pred["current"],pred["prediction"]))
print("Root Mean Square error using LSTM: ",rmse(pred["current"],pred["prediction"]))
print("R2 score using LSTM: ",r2score(pred["current"],pred["prediction"]))
dd3=pd.DataFrame({'MAPE': [mape(pred["current"],pred["prediction"])], 'MSE': [mse(pred["current"],pred["prediction"])], 'RMSE': [rmse(pred["current"],pred["prediction"])]}, index=[1])


# In[69]:


dd3.rename({1:'LSTM'})


# In[70]:


dd4=pd.concat([dd2,dd3])


# In[71]:


dd4.rename({0:'ARIMA', 1:'LSTM'})


# In[72]:


f,ax=plt.subplots()
plt.plot(test_predictions,c='green', label= 'predictions')

plt.plot(test, c='red',label='real values')
plt.legend()
plt.title('True vs predicted values')


# In[73]:


from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 5
n_features = 1
NtsC=np.reshape(tsC,(-1,1))

generator = TimeseriesGenerator(NtsC, NtsC, length=n_input, batch_size=1)

model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit_generator(generator,epochs=25)
loss_per_epoch = model.history.history['loss']
fig = plt.figure(dpi = 120,figsize = (8,4))
ax = plt.axes()
ax.set(xlabel = 'Number of Epochs',ylabel = 'MSE Loss',title = 'Loss Curve of RNN LSTM')
plt.plot(range(len(loss_per_epoch)),loss_per_epoch,lw = 2)


# In[74]:


forecast = []

first_eval_batch = NtsC[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(15):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[75]:


forecast = pd.DataFrame({'Forecast':np.ravel(forecast)})
forecast.index = np.arange('2020-05-15',15,dtype='datetime64[D]')
forecast


# In[ ]:





# In[ ]:




