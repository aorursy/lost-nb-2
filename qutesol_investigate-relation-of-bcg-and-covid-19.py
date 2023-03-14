#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly
import plotly.figure_factory as ff
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from datetime import date, datetime
import scipy as sc
import plotly.express as px
import requests

init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

cum_death_threshold = 100
pop_cum_death_threshold = 2
max_days_study = 35
min_days_study = 15
smooth_halflife = 4
debug=False

today = date.today()
#print("Today's date:", today)
today_ymd = today.strftime('%Y-%m-%d')
timestamp_str = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
download_url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
download_file = f'/kaggle/working/download-{today_ymd}.csv'
myfile = requests.get(download_url, allow_redirects=True)
open(download_file, 'wb').write(myfile.content)
# download_file = '/kaggle/input/covid19qutesol/download-2020-04-08.csv'
daily_df = pd.read_csv(download_file, encoding='latin-1')
daily_df = daily_df[daily_df.popData2018 >= 1000000]
daily_df.rename(columns={'countriesAndTerritories': 'Country'}, inplace=True)
daily_df.sort_values(['Country', 'year', 'month', 'day'], inplace=True)
daily_df['cum_cases'] = daily_df[['Country', 'cases']].groupby('Country').cumsum()
daily_df['cum_deaths'] = daily_df[['Country', 'deaths']].groupby('Country').cumsum()
daily_df['ewm_deaths'] = daily_df.groupby(['Country'])['deaths'].transform(lambda x: x.ewm(halflife=smooth_halflife).mean())

daily_df['pop_cum_cases'] = daily_df['cum_cases'] / daily_df['popData2018'] * 1000000
daily_df['pop_cum_deaths'] = daily_df['cum_deaths'] / daily_df['popData2018'] * 1000000
daily_df['pop_ewm_deaths'] = daily_df['ewm_deaths'] / daily_df['popData2018'] * 1000000

rebase_series = daily_df[(daily_df['pop_cum_deaths'] >= pop_cum_death_threshold)]
rebase_series = rebase_series.copy()
rebase_series['day_count'] = rebase_series.copy().groupby('Country')['cum_deaths'].cumcount()
rebase_series = rebase_series.groupby('Country').filter(lambda x: x['day_count'].max() >= min_days_study)

base = rebase_series[rebase_series['day_count'] == 0].sort_values(['year', 'month', 'day']).drop(columns=['day', 'month', 'year', 'geoId', 'cases', 'deaths', 'cum_deaths', 'cum_cases', 'day_count'])
# display(base, show_index=False)
# base = base.style.hide_index()
base = base[['Country', 'dateRep']]
base.rename(columns={'dateRep': 'Date of Country Onset'}, inplace=True)
rebase_pivot = pd.pivot_table(rebase_series, values=['pop_cum_deaths'], index='day_count', columns='Country')
rebase_pivot.columns = rebase_pivot.columns.droplevel(0)
rebase_pivot = rebase_pivot.reset_index()
rebase_pivot = rebase_pivot[rebase_pivot['day_count'] <= max_days_study]
rebase_pivot.drop(columns=['day_count'], inplace =True)

rebase_pivot_ewm = pd.pivot_table(rebase_series, values=['pop_ewm_deaths'], index='day_count', columns='Country')
rebase_pivot_ewm.columns = rebase_pivot_ewm.columns.droplevel(0)
rebase_pivot_ewm = rebase_pivot_ewm.reset_index()
rebase_pivot_ewm = rebase_pivot_ewm[rebase_pivot_ewm['day_count'] <= max_days_study]
rebase_pivot_ewm.drop(columns=['day_count'], inplace =True)

bcg_headline = pd.read_csv('/kaggle/input/covid19qutesol/bcg-atlas-v3.csv').set_index('Country')
death_scores = rebase_pivot[rebase_pivot.index == 12].melt(value_name='Day 12 Total Deaths Per Million').set_index('Country').join(bcg_headline).reset_index()


# In[2]:


slope, intercept, r_value, p_value, std_err =         sc.stats.linregress(death_scores['BCG Coverage'],death_scores['Day 12 Total Deaths Per Million'])
title = 'R^2 = %s, y = %sx+ %s, p-value = %s' % (round(r_value*r_value,2), round(slope,2), round(intercept,2), round(p_value,4))

scat_fig_bcg = px.scatter(death_scores, title=title, text='Country', x='BCG Coverage', y='Day 12 Total Deaths Per Million', trendline="ols", hover_data=['Country'], trendline_color_override=False)
scat_fig_bcg.update_traces(marker=dict(size=12)) 
scat_fig_bcg.update_traces(textposition='top center')
results = px.get_trendline_results(scat_fig_bcg)


scat_fig_bcg.show()


# In[3]:


import plotly.graph_objects as go
death_scores.sort_values(by='Day 12 Total Deaths Per Million', inplace=True)
countries=death_scores['Country'].values

fig = go.Figure(data=[
    go.Bar(name='BCG Coverage', x=countries, y=death_scores['BCG Coverage']),
    go.Bar(name='Day 12 Total Deaths Per Million', x=countries, y=death_scores['Day 12 Total Deaths Per Million'])
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# In[4]:


tot_fig = rebase_pivot.iplot(asFigure=True, legend=True, kind='scatter',xTitle='Days Since Country Onset',yTitle='Total Deaths per million people',title='Total Deaths')
tot_fig.update_traces(line_width=4)
tot_fig['layout'].update(annotations=[dict(
        showarrow = True,
        x = 12,
        y = 0,
        text = "Day 12",
        xanchor = "center",
        ax=0,
        ay=-150,
        yshift = 0,
        arrowsize = 2,
        arrowhead = 1,
        opacity = 0.9)])
iplot(tot_fig)


# In[5]:


ewm_fig = rebase_pivot_ewm.iplot(asFigure=True, kind='scatter',xTitle='Days Since Country Onset',yTitle='Death Rate per Million People',title=f'Daily Death Rate')
ewm_fig.update_traces(line_width=4)
iplot(ewm_fig)


# In[6]:


table = ff.create_table(base)
base.to_csv('onset.csv')
iplot(table, filename='covid19_start_date.html')


# In[7]:


results.iloc[0][0].summary()

