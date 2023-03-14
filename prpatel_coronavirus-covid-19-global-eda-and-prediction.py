#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
import datetime
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# In[20]:


# Fetch data

confirmed_df  = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df     = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

confirmed_df        = confirmed_df.groupby(['Country/Region']).sum()
recoveries_df       = recoveries_df.groupby(['Country/Region']).sum()
deaths_df           = deaths_df.groupby(['Country/Region']).sum()

dates               = np.array([dt[:-3] for dt in confirmed_df.columns[2:]])

date_ticks     = np.arange(0, len(dates), 7) # interval of 7 days
date_labels    = dates[date_ticks]

print('Data available till:', confirmed_df.columns[-1])


# In[21]:


def sir_model_fitting(country, cluster_population=50000000, passed_data=0, show_plots=1, days_to_predict=10):
    """Fit SIR model and plot data vs model result for 90 days for comparison"""
    if passed_data:
        ydata   = country
        country = 'Worldwide (excluding China)' 
    else:
        confirmed          = np.array(confirmed_df.loc[country, confirmed_df.columns[2:]])
        recovered          = np.array(recoveries_df.loc[country, recoveries_df.columns[2:]])
        deaths             = np.array(deaths_df.loc[country, deaths_df.columns[2:]])
        ydata              = confirmed - recovered - deaths
        
    xdata = np.arange(len(ydata))+1
    days_to_predict = len(xdata) + days_to_predict
    ind   = np.where(ydata>0)[0][0]
    model_output = ydata[ind:]
    model_input = np.arange(len(model_output))

    inf0 = model_output[0]
    sus0 = cluster_population - inf0
    rec0 = 0

    def sir_model(y, x, beta, gamma):
        sus = -beta * y[0] * y[1]/cluster_population
        rec = gamma * y[1]
        inf = -(sus + rec)
        return sus, inf, rec

    def fit_odeint(x, beta, gamma):
        return odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]

    popt, pcov = curve_fit(fit_odeint, model_input, model_output)
    fitted = fit_odeint(np.arange(days_to_predict-ind), *popt)
    fitted = np.append(np.zeros((ind,1)), fitted)

    if show_plots:
        fig = plt.figure()
        plt.plot(xdata, ydata, 'o')
        plt.plot(np.arange(len(fitted))+1, fitted)
        plt.plot([len(xdata), len(xdata)],[0, np.max(fitted)], ':k')
        plt.legend(['data', 'model prediction', "today we're here"])
        plt.title("SIR model fit to 'active cases' of " + country)
        plt.ylabel("Population infected")
        plt.xlabel("Days since 22 Jan 2020")
        plt.grid()

        print("Optimal parameters: beta =", round(popt[0],3), " gamma = ", round(popt[1],3))
    #     print('Goodness of fit', round(r2_score(ydata, fit_odeint(xdata, *popt)),4)*100, ' %')
        print('Optimal parameters Standard Dev: std_beta =', np.round(np.sqrt(pcov[0][0]),3), ' std_gamma =', np.round(np.sqrt(pcov[1][1]),3))
    else:
        return fitted


# In[22]:


def data_plot_country(country, show_plots=1): 
   confirmed          = np.array(confirmed_df.loc[country, confirmed_df.columns[2:]])
   recovered          = np.array(recoveries_df.loc[country, recoveries_df.columns[2:]])
   deaths             = np.array(deaths_df.loc[country, deaths_df.columns[2:]])
       
   if show_plots:
       fig = plt.figure()
       plt.stackplot(dates, confirmed - recovered - deaths, recovered, deaths, labels=['active cases', 'recovered','deaths'])
       plt.grid()
       plt.title('Cases in ' + country)
       plt.ylabel("# of cases")
       plt.xticks(date_ticks, date_labels)
       plt.legend(loc='upper left')
       print('Mortality rate:', round(deaths[-1]/confirmed[-1]*100,2), '%')
   
   return confirmed, recovered, deaths


# In[23]:


country= "China"
china_confirmed, china_recovered, china_fatalities  = data_plot_country(country)
plt.plot([1, 1], [0, np.max(china_confirmed)], ':r', label='Hubei Lockdown')
plt.legend(loc='upper left')
plt.show()

sir_model_fitting(country)


# In[24]:


country= "Korea, South"
korea_confirmed, _,_  = data_plot_country(country)
plt.show()

sir_model_fitting(country)


# In[25]:


number_of_top_countries_of_interest = 10

worldwide_confirmed                 = confirmed_df.loc[:, confirmed_df.columns[2:]].sum(axis=0)
worldwide_recoveries                = recoveries_df.loc[:, confirmed_df.columns[2:]].sum(axis=0)
worldwide_fatalities                = deaths_df.loc[:, confirmed_df.columns[2:]].sum(axis=0)
worldwide_active_cases              = np.array(worldwide_confirmed - worldwide_recoveries - worldwide_fatalities)

top_confirmed_worldwide             = confirmed_df.loc[:, confirmed_df.columns[-1]] - recoveries_df.loc[:, recoveries_df.columns[-1]] - deaths_df.loc[:, deaths_df.columns[-1]]
top_confirmed_worldwide             = top_confirmed_worldwide.sort_values(ascending=False)[:number_of_top_countries_of_interest]

total_of_top_countries      = 0
confirmed_of_top_countries  = np.zeros((1, len(confirmed_df.columns[2:])))
recoveries_of_top_countries = np.zeros((1, len(recoveries_df.columns[2:])))
fatalities_of_top_countries = np.zeros((1, len(deaths_df.columns[2:])))

fig, ax = plt.subplots(1, 3, figsize=(20, 4))
df = []
for country in top_confirmed_worldwide.index:
    data    = top_confirmed_worldwide[country]
    total_of_top_countries = total_of_top_countries + data
    confirmed_country, recoveries_country, fatalities_country = data_plot_country(country, show_plots=0)
    confirmed_of_top_countries    = confirmed_of_top_countries  + np.array(confirmed_country)
    recoveries_of_top_countries   = recoveries_of_top_countries + np.array(recoveries_country)
    fatalities_of_top_countries   = fatalities_of_top_countries + np.array(fatalities_country)
    ax[0].plot(dates, confirmed_country - recoveries_country - fatalities_country,label=country)
    country_fit = sir_model_fitting(country, show_plots=0, days_to_predict=10)
    df.append(country_fit)
    ax[1].plot(dates, country_fit[:-10], label=country)
    

ax[0].grid()
ax[0].set_title('Highest number of active cases Worldwide')
ax[0].set_ylabel("# confirmed cases")
ax[0].set_xticks(date_ticks)
ax[0].set_xticklabels(date_labels)
ax[0].legend()

ax[1].grid()
ax[1].set_title('Model prediction, active cases worldwide')
ax[1].set_ylabel("# predicted confirmed cases")
ax[1].set_xticks(date_ticks)
ax[1].set_xticklabels(date_labels)
ax[1].legend()


top_fatalities_worldwide    = deaths_df.loc[:, deaths_df.columns[-1]].sort_values(ascending=False)[:number_of_top_countries_of_interest]

for country in top_fatalities_worldwide.index:
#     data    = top_fatalities_except_china[country]
    _, _, fatalities_country = data_plot_country(country, show_plots=0)
    ax[2].plot(dates, fatalities_country,label=country)

ax[2].grid()
ax[2].set_title('Highest number of fatalities worldwide')
ax[2].set_ylabel("# fatalities")
ax[2].set_xticks(date_ticks)
ax[2].set_xticklabels(date_labels)
ax[2].legend()
plt.show()

total_proportion_of_top_countries = total_of_top_countries/worldwide_active_cases[-1]
print('Total proportion of top countries in worldwide active cases  = ', round(total_proportion_of_top_countries,2)*100, '% \n')


confirmed_non_top_countries      = np.array(worldwide_confirmed) - np.array(confirmed_of_top_countries[0])
recovered_non_top_countries      = np.array(worldwide_recoveries) - np.array(recoveries_of_top_countries[0])
fatalities_non_top_countries     = np.array(worldwide_fatalities) - np.array(fatalities_of_top_countries[0])
projection_non_top_countries     = sir_model_fitting(confirmed_non_top_countries -  recovered_non_top_countries - fatalities_non_top_countries, passed_data=1, cluster_population=50000000, show_plots=0)

# projection = np.sum(df, axis=0)/total_proportion_of_top_countries # Rough estimate using top countries data only
projection = np.sum(df, axis=0) + projection_non_top_countries

sir_model_fitting(worldwide_active_cases, passed_data=1, cluster_population=50000000)
plt.plot(projection)
plt.legend(['Worldwide data', 'Prediction by using worldwide data', "today we're here", 'Prediction by modelling top clusters individually'])
plt.show()


# In[26]:


sir_model_fitting('India')


# In[27]:


sir_model_fitting('United Kingdom')

