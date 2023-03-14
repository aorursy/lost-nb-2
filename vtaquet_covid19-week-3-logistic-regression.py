#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary modules
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import norm, skew, probplot 

warnings.filterwarnings('ignore')


# In[2]:


dftrain = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv', parse_dates=['Date']).sort_values(by=['Country_Region', 'Date']).fillna('None')
dftest = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv', parse_dates=['Date']).sort_values(by=['Country_Region', 'Date']).fillna('None')
dfsubm = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')#, parse_dates=['Date']).sort_values(by=['Country_Region', 'Date'])


# In[3]:


dftrain.head(2)


# In[4]:


dftrain.tail(2)


# In[5]:


dftest.head(2)


# In[6]:


dftest.tail(2)


# In[7]:


dfsubm.head(2)


# In[8]:


starttest = dftest['Date'].min() ; endtest = dftest['Date'].max() ; datetest = dftest['Date'].unique()


# In[9]:


confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv').sort_values(by='Country/Region')#.set_index('Country/Region')
deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')#.set_index('Country/Region')
recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')#.set_index('Country/Region')


# In[10]:


def change_countryprovince(df):
    df['Country_Region'] = df['Country/Region']
    df['Province_State'] = df['Province/State']
    df = df.drop(['Country/Region', 'Province/State'], axis=1)
    return df


# In[11]:


confirmed = change_countryprovince(confirmed)
deaths = change_countryprovince(deaths)
recovered = change_countryprovince(recovered)
#
csse_data = pd.DataFrame(confirmed.set_index(['Province_State','Country_Region']).drop(['Lat','Long'],axis=1).stack()).reset_index()
csse_data.columns = ['Province_State', 'Country_Region', 'Date', 'ConfirmedCases']
#
csse_data['Fatalities'] = deaths.set_index(['Province_State','Country_Region']).drop(['Lat','Long'],axis=1).stack().tolist()
csse_data['Date'] = pd.to_datetime(csse_data['Date'])
csse_data.head()
#confirmed.head()


# In[12]:


len(csse_data['Province_State'].unique())


# In[13]:


len(dftrain['Province_State'].unique())


# In[14]:


dftrain = dftrain.join(confirmed[['Country_Region', 'Province_State', 'Lat', 'Long']].set_index(['Province_State', 'Country_Region']), on=['Province_State', 'Country_Region'])#, how='outer')#.set_index(['Province_State', 'Country_Region']))


# In[15]:


dftrain['Dayofyear'] = dftrain['Date'].dt.dayofyear
dftest['Dayofyear'] = dftest['Date'].dt.dayofyear


# In[16]:


def transpose_df(df):
    df = df.drop(['Lat','Long'],axis=1).groupby('Country_Region').sum().T
    df.index = pd.to_datetime(df.index)#.date
    return df


# In[ ]:





# In[17]:


confirmedT = transpose_df(confirmed)
deathsT = transpose_df(deaths)
recoveredT = transpose_df(recovered)
mortalityT = deathsT/confirmedT


# In[18]:


def add_day(df):
    df['Date'] = df.index
    df['Dayofyear'] = df['Date'].dt.dayofyear
    return df


# In[19]:


confirmedT, deathsT, recoveredT, mortalityT = add_day(confirmedT), add_day(deathsT), add_day(recoveredT), add_day(mortalityT)


# In[20]:


allcountries_ordered = confirmed.set_index(['Country_Region']).iloc[:,-2].sort_values(ascending=False).index.tolist()


# In[21]:


confirmed.set_index(['Country_Region']).iloc[:,-2].sort_values(ascending=False).to_csv('confirmed_countries.csv')


# In[22]:


#
def df_day1(df, confirmed):
    df_day1 = pd.DataFrame({'Days since 100 cases' : np.arange(1000)}).set_index('Days since 100 cases')
    countries_df = df.columns.tolist()[:-2]
    countries_conf = confirmed.columns.tolist()[:-2]
    #print(len(countries_df), len(confirmed.columns.tolist()[:-2]))
    for ic, country in enumerate(countries_df):
        for ic2, country2 in enumerate(countries_conf):
            if country == country2:
                dfsub = df[confirmed[country] > 100.][country]
                df_day1[country] = np.nan
                df_day1.loc[:len(dfsub)-1,country] = (dfsub).tolist()
        #try:
        #except:
        #    pass
    df_day1 = df_day1.dropna(how='all')
    #df_day1 = df_day1.fillna(0.)
    return df_day1


# In[23]:


confirmed_day1 = df_day1(confirmedT, confirmedT)
deaths_day1 = df_day1(deathsT, confirmedT)
recovered_day1 = df_day1(recoveredT, confirmedT)
mortality_day1 = df_day1(mortalityT, confirmedT)
confirmednorm_day1 = confirmed_day1/confirmed_day1.loc[0,:]
maxday = confirmed_day1.shape[0]


# In[24]:


date_day1 = confirmedT.copy()
for column in date_day1:
    date_day1[column] = confirmedT.index.tolist()
date_day1 = df_day1(date_day1, confirmedT)


# In[25]:


date_day1.T.sort_values(by=0).head()


# In[26]:


allcountries = dftrain['Country_Region'].unique().tolist()
#allcountries_confirmed = confirmed['Country/Region'].unique().tolist()
#allcountries_deaths = deaths['Country/Region'].unique().tolist()
#allcountries_recovered = confirmed['Country/Region'].unique().tolist()
#countries = ['France', 'Italy', 'Spain', 'US', 'Germany', 'United Kingdom', 'China']


# In[27]:


def logistic_curve(x, k, x_0, ymax):
    return ymax / (1 + np.exp(-k*(x-x_0)))


# In[28]:


def logistic_curve2(x, k1, k2, x_0, ymax):
    return ymax / (1 + np.exp(-k1*(x-x_0)) + np.exp(-k2*(x-x_0)))


# In[29]:


#
list_countries = dftrain[dftrain['Date'] == '2020-01-22']['Country_Region'].tolist()
list_states = dftrain[dftrain['Date'] == '2020-01-22']['Province_State'].tolist()
datenow = datetime.now()


# In[30]:


list_date_pand = [] ; list_maxcases = []; list_maxfat = []
for country, state in list(zip(list_countries, list_states)):
    df2 = dftrain.loc[(dftrain['Country_Region'] == country) & (dftrain['Province_State'] == state)].fillna('None')
    maxcases, maxfat = df2['ConfirmedCases'].max(), df2['Fatalities'].max()
    date_pand2 = []
    date_pand = df2[df2['ConfirmedCases'] > 100.]['Date'].tolist()#[0]
    try:
        list_date_pand.append(pd.to_datetime(date_pand[0]))
    except:
        list_date_pand.append(pd.to_datetime(datenow))
    list_maxcases.append(maxcases) ; list_maxfat.append(maxfat)
#
dfstartpand = pd.DataFrame(np.array([list_countries, list_states, list_date_pand, list_maxcases, list_maxfat]).T, 
                           columns=['Country_Region', 'Province_State', 'Start_Pandemic', 'ConfirmedCases', 'Fatalities'])
dfstartpand['Start_Pandemic'] = dfstartpand['Start_Pandemic'].dt.date


# In[31]:


#
dfstartpand_ordered = dfstartpand.sort_values(by=['Start_Pandemic', 'ConfirmedCases', 'Fatalities'], ascending=[True, False, False])#.head(5)
country_state_ordered = list(zip(dfstartpand_ordered['Country_Region'].tolist(), dfstartpand_ordered['Province_State']))
#country_state_ordered = list(zip(dfstartpand_ordered[['Country_Region', 'Province_State']]))
#
datetrain = dftrain['Date'].unique()#.dt.date
datetest = dftest['Date'].unique()#.dt.date


# In[32]:


starttest


# In[33]:


from scipy.optimize import curve_fit
import math
#
dftest['ConfirmedCases_logreg'] = 0.0 ; dftrain['ConfirmedCases_logreg'] = 0.0
dftest['Fatalities_logreg'] = 0.0 ; dftrain['Fatalities_logreg'] = 0.0
p0 = 1
#
for country, state in country_state_ordered:
    #print(country+' '+state)
    #country2plot = 'Italy' ; state2plot = 'None'
    masktrain = (dftrain['Country_Region'] == country) & (dftrain['Province_State'] == state)
    masktrain2 = (dftrain['Country_Region'] == country) & (dftrain['Province_State'] == state) &                  (dftrain['Date'] <= '2020-04-07') & (dftrain['Date'] >= starttest) 
    masktest = (dftest['Country_Region'] == country) & (dftest['Province_State'] == state)
    masktest2 = (dftest['Country_Region'] == country) & (dftest['Province_State'] == state) &                 (dftest['Date'] <= '2020-04-07')
    df2plot = dftrain[masktrain].set_index('Date')
    #print(len(dftrain[masktrain2]))
    #
    X = np.arange(len(df2plot))
    X_test = (np.timedelta64(datetest[0]-datetrain[0], 'D')).astype(float)+np.arange(0,len(datetest))
    #
    try:
        y = df2plot['ConfirmedCases']
        p0_cases = [1/(len(X)/2.), X[-1], y.max()]
        popt, pcov = curve_fit(logistic_curve, X, y, p0=p0_cases,bounds=([0,0,0],np.inf), maxfev=1000)
        k_cases, x_0_cases, ymax_cases = popt
        cases_train_fc = pd.Series(logistic_curve(X, k_cases, x_0_cases, ymax_cases),index=df2plot.index)
        cases_test_fc = pd.Series(logistic_curve(X_test, k_cases, x_0_cases, ymax_cases),index=datetest)
        #
        dftest.loc[masktest,'ConfirmedCases_logreg'] = cases_test_fc.tolist()
        dftrain.loc[masktrain,'ConfirmedCases_logreg'] = cases_train_fc.tolist()
    except:
        print(country+' '+state+' Unable to fit the confirmed cases')
        dftest.loc[masktest,'ConfirmedCases_logreg'] = dftrain.loc[masktrain,'ConfirmedCases'].iloc[-1]#cases_test_fc.tolist()
        dftrain.loc[masktrain,'ConfirmedCases_logreg'] = dftrain.loc[masktrain,'ConfirmedCases']
    try:
        y = df2plot['Fatalities']#.rolling(3).mean()
        p0_deaths = [1/(len(X)/2.), X[-1], y.max()]
        popt, pcov = curve_fit(logistic_curve, X, y, p0=p0_deaths,bounds=([0,0,0],np.inf), maxfev=1000)
        k_deaths, x_0_deaths, ymax_deaths = popt
        deaths_train_fc = pd.Series(logistic_curve(X, k_deaths, x_0_deaths, ymax_deaths),index=datetrain)
        deaths_test_fc = pd.Series(logistic_curve(X_test, k_deaths, x_0_deaths, ymax_deaths),index=datetest)
        #
        dftest.loc[masktest,'Fatalities_logreg'] = deaths_test_fc.tolist()
        dftrain.loc[masktrain,'Fatalities_logreg'] = deaths_train_fc.tolist()
    except:
        print(country+' '+state+' Unable to fit the fatalities')
        dftest.loc[masktest,'Fatalities_logreg'] = dftrain.loc[masktrain,'Fatalities'].iloc[-1] #deaths_test_fc.tolist()
        dftrain.loc[masktrain,'Fatalities_logreg'] = dftrain.loc[masktrain,'Fatalities']
    dftest.loc[masktest2,'ConfirmedCases_logreg'] = dftrain.loc[masktrain2,'ConfirmedCases'].tolist()
    dftest.loc[masktest2,'Fatalities_logreg'] = dftrain.loc[masktrain2,'Fatalities'].tolist()


# In[34]:


country2plot = 'Japan' ; state2plot = 'None'
masktrain = (dftrain['Country_Region'] == country2plot) & (dftrain['Province_State'] == state2plot)
masktest = (dftest['Country_Region'] == country2plot) & (dftest['Province_State'] == state2plot)
train2plot = dftrain[masktrain].set_index('Date')
test2plot = dftest[masktest].set_index('Date')
#
fig = plt.figure(0,figsize=[20,5])
fig.subplots_adjust(wspace=0.25, hspace=0.45)
ax = fig.add_subplot(1,4,1)
ax.set_ylabel('Total number of confirmed cases')
ax = train2plot['ConfirmedCases'].rolling(1).mean().plot()
ax = train2plot['ConfirmedCases_logreg'].plot()
ax = test2plot['ConfirmedCases_logreg'].plot()
#
ax = fig.add_subplot(1,4,2)
ax.set_ylabel('Total number of fatalities')
ax = train2plot['Fatalities'].rolling(1).mean().plot()
ax = train2plot['Fatalities_logreg'].plot()
ax = test2plot['Fatalities_logreg'].plot()
#
ax = fig.add_subplot(1,4,3)
ax.set_ylabel('Confirmed cases per day')
ax = train2plot['ConfirmedCases'].diff().rolling(3).mean().plot()
ax = train2plot['ConfirmedCases_logreg'].diff().plot()
ax = test2plot['ConfirmedCases_logreg'].diff().plot()
#
ax = fig.add_subplot(1,4,4)
ax.set_ylabel('Fatalities per day')
ax = train2plot['Fatalities'].diff().rolling(3).mean().plot()
ax = train2plot['Fatalities_logreg'].diff().plot()
ax = test2plot['Fatalities_logreg'].diff().plot()


# In[35]:


dfsubm['ConfirmedCases'] = dftest['ConfirmedCases_logreg']#.tol
dfsubm['Fatalities'] = dftest['Fatalities_logreg']#.tolist()


# In[36]:


dfsubm.head(5)


# In[37]:


dfsubm.to_csv('submission.csv', index=False)

