#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import geopandas as gpd

import datetime
import calendar
from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns

import folium
from folium.plugins import MarkerCluster,FastMarkerCluster, HeatMap

#matplotlib display in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#seaborn style
sns.set(style='whitegrid', palette='muted', color_codes=True)


# In[2]:


df_air_reserve = pd.read_csv('../input/air_reserve.csv')
df_air_store = pd.read_csv('../input/air_store_info.csv')
df_air_visit = pd.read_csv('../input/air_visit_data.csv')
df_hpg_reserve = pd.read_csv('../input/hpg_reserve.csv')
df_hpg_store = pd.read_csv('../input/hpg_store_info.csv')
df_date_info = pd.read_csv('../input/date_info.csv')
df_store_id_rel = pd.read_csv('../input/store_id_relation.csv')


# In[3]:


df_air_reserve.head()


# In[4]:


df_air_store.head()


# In[5]:


df_air_visit.head()


# In[6]:


df_hpg_reserve.head()


# In[7]:


df_hpg_store.head()


# In[8]:


df_store_id_rel.head()


# In[9]:


df_date_info.head()


# In[10]:


df_date_info.groupby('day_of_week')            .agg({'holiday_flg':'sum'})             .sort_values(by='holiday_flg', ascending=False)            .reset_index() 


# In[11]:


# merge 'air' tables and bring over any 'hpg' store data
df_air_merged = df_air_reserve.merge(df_air_store,on='air_store_id', how='left').merge(
    df_store_id_rel, on='air_store_id', how='left').merge(df_hpg_store,on='hpg_store_id', how='left',suffixes=('_air','_hpg'))


# In[12]:


# merge 'hpg' tables and bring over any 'air' store data
df_hpg_merged = df_hpg_reserve.merge(df_hpg_store,on='hpg_store_id', how='left').merge(
    df_store_id_rel,on='hpg_store_id', how='left').merge(df_air_store,on='air_store_id', how='left',suffixes=('_hpg','_air'))


# In[13]:


# add source column
df_air_merged['source'] = 'air'
df_hpg_merged['source'] = 'hpg'


# In[14]:


# append tables together
df_res_merged = df_air_merged.append(df_hpg_merged)
df_res_merged.reset_index(inplace=True)


# In[15]:


# format date fields
df_res_merged['visit_datetime'] = pd.to_datetime(df_res_merged.visit_datetime)
df_res_merged['reserve_datetime'] = pd.to_datetime(df_res_merged.reserve_datetime)

df_res_merged['calendar_date'] = df_res_merged.visit_datetime.dt.date
df_res_merged['visit_time'] = df_res_merged.visit_datetime.dt.time
df_res_merged['reserve_date'] = df_res_merged.reserve_datetime.dt.date
df_res_merged['reserve_time'] = df_res_merged.reserve_datetime.dt.time


# In[16]:


# add month, year, and season
df_res_merged['visit_month'] = df_res_merged.visit_datetime.apply(lambda x: x.strftime("%b"))
df_res_merged['visit_year'] = df_res_merged.visit_datetime.apply(lambda x: x.strftime("%Y"))
df_res_merged['reserve_month'] = df_res_merged.reserve_datetime.apply(lambda x: x.strftime("%b"))
df_res_merged['reserve_year'] = df_res_merged.reserve_datetime.apply(lambda x: x.strftime("%Y"))

seasons = {'Jan': 'Winter','Feb': 'Winter','Mar': 'Spring','Apr': 'Spring','May': 'Spring','Jun': 'Summer',
           'Jul': 'Summer','Aug': 'Summer','Sep': 'Autumn','Oct': 'Autumn','Nov': 'Autumn','Dec': 'Winter'}

df_res_merged['reserve_season'] = df_res_merged['reserve_month'].map(seasons)
df_res_merged['visit_season'] = df_res_merged['visit_month'].map(seasons)


# In[17]:


# format df_date_info date to merge
df_date_info['calendar_date'] = pd.to_datetime(df_date_info.calendar_date)
df_date_info['calendar_date'] = df_date_info.calendar_date.dt.date
df_res_merged = df_res_merged.merge(df_date_info, on='calendar_date', how='left')
df_res_merged.rename(columns={"day_of_week": "day_of_week_visit", "holiday_flg": "holiday_flag_visit"}, inplace=True)
df_date_info.rename(columns={"calendar_date": "reserve_date"}, inplace=True)
df_res_merged = df_res_merged.merge(df_date_info, on='reserve_date', how='left')
df_res_merged.rename(columns={"day_of_week": "day_of_week_res", "holiday_flg": "holiday_flag_res"}, inplace=True)


# In[18]:


# time between reservation and visit
df_res_merged['res_vs_visit'] = df_res_merged['visit_datetime'] - df_res_merged['reserve_datetime']
df_res_merged['res_vs_visit_days'] = df_res_merged['res_vs_visit'].astype('timedelta64[D]')
df_res_merged['res_vs_visit_hours'] = df_res_merged['res_vs_visit'].astype('timedelta64[h]')


# In[19]:


# holiday the day before and after visit
df_res_merged['holiday_before_visit'] = df_res_merged.holiday_flag_visit.shift(1)
df_res_merged.holiday_before_visit.fillna(0,inplace=True)
df_res_merged['holiday_after_visit'] = df_res_merged.holiday_flag_visit.shift(-1)
df_res_merged.holiday_after_visit.fillna(0,inplace=True)


# In[20]:


df_res_merged.describe()


# In[21]:


df_genre = df_res_merged[df_res_merged.hpg_genre_name != 'No Data'].groupby(['hpg_genre_name'])                         .agg({'index':'size', 'reserve_visitors':'mean', 'res_vs_visit_hours':'mean'})                        .reset_index()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(13, 15))
ax1.set_title('Reservations made total')
ax2.set_title('Mean time between\n reservation and visit')
ax3.set_title('Mean number of\n visitors per reservation')
sns.barplot(x='index', y='hpg_genre_name', data=df_genre,
            ax=ax1, color='r')
sns.barplot(x='res_vs_visit_hours', y='hpg_genre_name', data=df_genre, 
            ax=ax2, color='g')
sns.barplot(x='reserve_visitors', y='hpg_genre_name', data=df_genre, 
            ax=ax3, color='b')


# In[22]:


df_genre = df_res_merged[df_res_merged.air_genre_name != 'No Data'].groupby(['air_genre_name'])                         .agg({'index':'size', 'reserve_visitors':'mean', 'res_vs_visit_hours':'mean'})                        .reset_index()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(13, 10))
ax1.set_title('Reservations made total')
ax2.set_title('Mean time between\n reservation and visit')
ax3.set_title('Mean number of\n visitors per reservation')
sns.barplot(x='index', y='air_genre_name', data=df_genre,
            ax=ax1, color='r')
sns.barplot(x='res_vs_visit_hours', y='air_genre_name', data=df_genre, 
            ax=ax2, color='g')
sns.barplot(x='reserve_visitors', y='air_genre_name', data=df_genre, 
            ax=ax3, color='b')


# In[23]:


# plot holiday vs non holiday
df_genre_by_holiday = df_res_merged.groupby(['hpg_genre_name', 'holiday_flag_visit'])                                    .agg({'index':'size', 'reserve_visitors':'mean', 'res_vs_visit_hours':'mean'}) 

test = pd.DataFrame(df_genre_by_holiday.groupby(level=0)['index'].apply(lambda x:100 * x / float(x.sum())))
test.rename(columns={"index": "index_pct"}, inplace=True)

df_genre_by_holiday = df_genre_by_holiday.merge(test, left_index=True, right_index=True).reset_index()
            
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(13, 15))
ax1.set_title('Reservations made\n for holiday vs non')
ax2.set_title('Mean time between\n reservation and visit')
ax3.set_title('Mean number of\n visitors per reservation')
sns.barplot(x='index_pct', y='hpg_genre_name', data=df_genre_by_holiday,
            ax=ax1, hue='holiday_flag_visit', hue_order=[0,1], color='r')
sns.barplot(x='res_vs_visit_hours', y='hpg_genre_name', data=df_genre_by_holiday, 
            ax=ax2, hue='holiday_flag_visit', hue_order=[0,1], color='g')
sns.barplot(x='reserve_visitors', y='hpg_genre_name', data=df_genre_by_holiday, 
            ax=ax3, hue='holiday_flag_visit', hue_order=[0,1], color='b')


# In[24]:


# plot holiday vs non holiday
df_genre_by_holiday = df_res_merged.groupby(['air_genre_name', 'holiday_flag_visit'])                                    .agg({'index':'size', 'reserve_visitors':'mean', 'res_vs_visit_hours':'mean'}) 

test = pd.DataFrame(df_genre_by_holiday.groupby(level=0)['index'].apply(lambda x:100 * x / float(x.sum())))
test.rename(columns={"index": "index_pct"}, inplace=True)

df_genre_by_holiday = df_genre_by_holiday.merge(test, left_index=True, right_index=True).reset_index()
            
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(13, 10))
ax1.set_title('Reservations made\n for holiday vs non')
ax2.set_title('Mean time between\n reservation and visit')
ax3.set_title('Mean number of\n visitors per reservation')
sns.barplot(x='index_pct', y='air_genre_name', data=df_genre_by_holiday, 
            ax=ax1, hue='holiday_flag_visit', hue_order=[0,1], color='r')
sns.barplot(x='res_vs_visit_hours', y='air_genre_name', data=df_genre_by_holiday, 
            ax=ax2, hue='holiday_flag_visit', hue_order=[0,1], color='g')
sns.barplot(x='reserve_visitors', y='air_genre_name', data=df_genre_by_holiday, 
            ax=ax3, hue='holiday_flag_visit', hue_order=[0,1], color='b')


# In[25]:


# too many genres - amalgamate
genres = {
    'Japanese style':'Japanese',
    'International cuisine':'Other',
    'Grilled meat':'Other Asian',
    'Creation':'Japanese',
    'Italian':'European',
    'Seafood':'Other',
    'Spain Bar/Italian Bar':'European',
    'Japanese food in general':'Japanese',
    'Shabu-shabu/Sukiyaki':'Japanese',
    'Chinese general':'Other Asian',
    'Creative Japanese food':'Japanese',
    'Japanese cuisine/Kaiseki':'Japanese',
    'Korean cuisine':'Other Asian',
    'Okonomiyaki/Monja/Teppanyaki':'Japanese',
    'Karaoke':'Bar or Club',
    'Steak/Hamburger/Curry':'Other',
    'French':'European',
    'Cafe':'European',
    'Bistro':'Other',
    'Sushi':'Japanese',
    'Party':'Bar or Club',
    'Western food':'Other',
    'Pasta/Pizza':'Other',
    'Thai/Vietnamese food':'Other Asian',
    'Bar/Cocktail':'Bar or Club',
    'Amusement bar':'Bar or Club',
    'Cantonese food':'Other Asian',
    'Dim Sum/Dumplings':'Other Asian',
    'Sichuan food':'Other Asian',
    'Sweets':'Other',
    'Spain/Mediterranean cuisine':'European',
    'Udon/Soba':'Japanese',
    'Shanghai food':'Other Asian',
    'Taiwanese/Hong Kong cuisine':'Other Asian',
    'Japanese food':'Japanese', 
    'Dining bar':'Bar or Club', 
    'Izakaya':'Japanese',
    'Okonomiyaki/Monja/Teppanyaki':'Japanese', 
    'Italian/French':'European', 
    'Cafe/Sweets':'Other',
    'Yakiniku/Korean food':'Other Asian', 
    'Western food':'Other', 
    'Bar/Cocktail':'Bar or Club', 
    'Other':'Other',
    'Creative cuisine':'Japanese', 
    'Karaoke/Party':'Bar or Club', 
    'International cuisine':'Other',
    'Asian':'Other Asian',
    'None':'None',
    'No Data':'No Data'}
df_res_merged.hpg_genre_name.fillna('No Data', inplace=True)
df_res_merged.air_genre_name.fillna('No Data', inplace=True)
df_res_merged.hpg_store_id.fillna('No Data', inplace=True)
df_res_merged.air_store_id.fillna('No Data', inplace=True)
df_res_merged['air_genre_2'] = df_res_merged['air_genre_name'].map(genres)
df_res_merged['hpg_genre_2'] = df_res_merged['hpg_genre_name'].map(genres)

# take hpg genre first then air
df_res_merged['genre_2']=df_res_merged['hpg_genre_2']
df_res_merged.loc[df_res_merged['hpg_genre_2']=='No Data',['genre_2']] = df_res_merged['air_genre_2']


# In[26]:


# plot holiday vs non holiday
df_genre_by_holiday = df_res_merged.groupby(['genre_2', 'holiday_flag_visit'])                                    .agg({'index':'size', 'reserve_visitors':'mean', 'res_vs_visit_hours':'mean'}) 

test = pd.DataFrame(df_genre_by_holiday.groupby(level=0)['index'].apply(lambda x:100 * x / float(x.sum())))
test.rename(columns={"index": "index_pct"}, inplace=True)

df_genre_by_holiday = df_genre_by_holiday.merge(test, left_index=True, right_index=True).reset_index()
            
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(13, 8))
ax1.set_title('Reservations made\n for holiday vs non')
ax2.set_title('Mean time between\n reservation and visit')
ax3.set_title('Mean number of\n visitors per reservation')
sns.barplot(x='index_pct', y='genre_2', data=df_genre_by_holiday, 
            ax=ax1, hue='holiday_flag_visit', hue_order=[0,1], color='r')
sns.barplot(x='res_vs_visit_hours', y='genre_2', data=df_genre_by_holiday, 
            ax=ax2, hue='holiday_flag_visit', hue_order=[0,1], color='g')
sns.barplot(x='reserve_visitors', y='genre_2', data=df_genre_by_holiday, 
            ax=ax3, hue='holiday_flag_visit', hue_order=[0,1], color='b')


# In[27]:


# Reservation visits per season
df_visit_by_season = df_res_merged.groupby(['genre_2','visit_season','visit_month'])                              .agg({'index':'size'})                              .reset_index() 
df_visit_by_season['total_size'] = df_visit_by_season.groupby('genre_2')['index'].transform('sum')
df_visit_by_season.sort_values(by='total_size', inplace=True)
df_visit_by_season['visit_pct'] = (df_visit_by_season['index'] / df_visit_by_season['total_size'])*100
        
pal=['blue','blue','green','green','green', 'red','red','red', 'orange','orange','orange', 'blue']
g = sns.FacetGrid(df_visit_by_season, col="genre_2", col_wrap=3, size=5)
g.map(sns.barplot, "visit_month", "visit_pct", palette=pal, order=['Jan','Feb','Mar','Apr','May','Jun',
           'Jul','Aug','Sep','Oct','Nov','Dec'])


# In[28]:


# Reservation visits per season
df_res_visitors_by_season = df_res_merged.groupby(['visit_month'])                              .agg({'reserve_visitors':'mean'})                              .reset_index() 
pal=['blue','blue','green','green','green', 'red','red','red', 'orange','orange','orange', 'blue']
sns.barplot("visit_month", "reserve_visitors", data=df_res_visitors_by_season, palette=pal, 
            order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])


# In[29]:


# reservations and visits by day of week
dow = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']

df_res_by_dow = df_res_merged.groupby(['day_of_week_res'])                              .agg({'index':'size'})                              .reset_index() 
df_res_by_dow['day_of_week_res'] = pd.Categorical(df_res_by_dow['day_of_week_res'],categories=dow, ordered=True)
df_res_by_dow.sort_values(by='day_of_week_res', inplace=True) 
df_res_by_dow.set_index('day_of_week_res', inplace=True)
df_res_by_dow.rename(columns={"index": "reservation_count"}, inplace=True)
df_res_by_dow.index.names = ['day_of_week']

df_visit_by_dow = df_res_merged.groupby(['day_of_week_visit'])                              .agg({'index':'size'})                              .reset_index() 

df_visit_by_dow['day_of_week_visit'] = pd.Categorical(df_visit_by_dow['day_of_week_visit'],categories=dow, ordered=True)

df_visit_by_dow.sort_values(by='day_of_week_visit', inplace=True)
df_visit_by_dow.set_index('day_of_week_visit', inplace=True)
df_visit_by_dow.rename(columns={"index": "visit_count"}, inplace=True)
df_visit_by_dow.index.names = ['day_of_week']

df_by_dow = df_res_by_dow.merge(df_visit_by_dow, left_index=True, right_index=True).reset_index()
df_by_dow = df_by_dow.melt(id_vars=['day_of_week'], value_vars=['reservation_count', 'visit_count'])

sns.factorplot(x='day_of_week', y='value', data=df_by_dow, hue='variable', aspect=3)


# In[30]:



f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(16, 10))
visit_vs_res = df_res_merged.groupby(['res_vs_visit_hours', 'source'])       .agg({'index':'size'})        .reset_index()
visit_vs_res = visit_vs_res[visit_vs_res.res_vs_visit_hours<150]    

ax1.bar(x='res_vs_visit_hours', height='index', data=visit_vs_res[visit_vs_res.source == 'air'], color='b')
ax1.set_title('Air')
ax1.set_ylabel('Visits')
ax1.grid(b=None, axis='x')

ax2.bar(x='res_vs_visit_hours', height='index', data=visit_vs_res[visit_vs_res.source == 'hpg'], color='g')
ax2.set_title('HPG')
ax2.set_ylabel('Visits')
ax2.set_xlabel('Hours')
ax2.set_xlim(0)
ax2.grid(b=None, axis='x')

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=24))


# In[31]:


df_res_by_time = df_res_merged.groupby(['reserve_time'])                              .agg({'index':'size'})                              .reset_index() 
df_res_by_time.sort_values(by='reserve_time', inplace=True) 
df_res_by_time.set_index('reserve_time', inplace=True)
df_res_by_time.rename(columns={"index": "reservation_count"}, inplace=True)
df_res_by_time.index.names = ['time']

df_visit_by_time = df_res_merged.groupby(['visit_time'])                              .agg({'index':'size'})                              .reset_index() 
df_visit_by_time.sort_values(by='visit_time', inplace=True)
df_visit_by_time.set_index('visit_time', inplace=True)
df_visit_by_time.rename(columns={"index": "visit_count"}, inplace=True)
df_visit_by_time.index.names = ['time']

df_by_time = df_res_by_time.merge(df_visit_by_time, left_index=True, right_index=True).reset_index()
df_by_time = df_by_time.melt(id_vars=['time'], value_vars=['reservation_count', 'visit_count'])

sns.factorplot(x='time', y='value', data=df_by_time, hue='variable', aspect=3).set_xticklabels(rotation=30)


# In[32]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(16, 10))
plt.xticks(rotation=30)
visit_time = df_res_merged.groupby(['visit_time', 'source'])             .agg({'index':'size'})              .reset_index() 
visit_time['visit_time'] = visit_time.visit_time.apply(lambda x: str(x))
        
ax1.bar(x='visit_time', height='index', data=visit_time[visit_time.source == 'air'], color='b')
ax1.set_title('Air')
ax1.set_ylabel('Visits')
ax1.grid(b=None, axis='x')

ax2.bar(x='visit_time', height='index', data=visit_time[visit_time.source == 'hpg'], color='g')
ax2.set_title('HPG')
ax2.set_ylabel('Visits')
ax2.set_xlabel('Hour of the day')
ax2.set_xlim(0)
ax2.grid(b=None, axis='x')


# In[33]:


# Reservations by visit date and source
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(16, 10))
visits_per = df_res_merged.groupby(['calendar_date', 'source'])             .agg({'index':'size'})              .reset_index() 

ax1.plot_date(x='calendar_date', y='index', data=visits_per[visits_per.source == 'air'], ms=3, c='b', ls='solid', lw=1)
ax1.set_title('Air visits')
ax1.set_ylabel('Visits')
ax1.grid(b=None, axis='x')

ax2.plot_date(x='calendar_date', y='index', data=visits_per[visits_per.source == 'hpg'], ms=3, c='g', ls='solid', lw=1)
ax2.set_title('HPG visits')
ax2.set_ylabel('Visits')
ax2.grid(b=None, axis='x')

ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))


# In[34]:


visit_vs_res_top = df_res_merged.groupby(['air_store_id', 'hpg_store_id','air_genre_name', 'hpg_genre_name'])                                .agg({'res_vs_visit_hours':'mean', 'index':'size', 'reserve_visitors':'mean'})                                 .reset_index()                                 .sort_values(by='res_vs_visit_hours', ascending=False)
visit_vs_res_top['res_vs_visit_hours'] = visit_vs_res_top.res_vs_visit_hours / 24
visit_vs_res_top.rename(columns={'res_vs_visit_hours':'res_vs_visit_days','index':'No. of reservations'}, inplace=True)
visit_vs_res_top = visit_vs_res_top[visit_vs_res_top['No. of reservations']>100]
visit_vs_res_top.head(10)


# In[35]:


# format df_date_info and add month, year, and season

df_air_visit['visit_date'] = pd.to_datetime(df_air_visit.visit_date)
df_air_visit['visit_date'] = df_air_visit.visit_date.dt.date
df_air_visit = df_air_visit.merge(df_date_info, left_on='visit_date', right_on='reserve_date', how='left')
df_air_visit.drop('reserve_date', axis=1, inplace=True)
df_air_visit = df_air_visit.merge(df_air_store,on='air_store_id', how='left')
df_air_visit['visit_month'] = df_air_visit.visit_date.apply(lambda x: x.strftime("%b"))
df_air_visit['visit_year'] = df_air_visit.visit_date.apply(lambda x: x.strftime("%Y"))
df_air_visit['visit_season'] = df_air_visit['visit_month'].map(seasons)


# In[36]:


# amalgamate genres like in hpg
df_air_visit.air_genre_name.fillna('No Data', inplace=True)
df_air_visit['air_genre_2'] = df_air_visit['air_genre_name'].map(genres)


# In[37]:


# holiday the day before and after visit
df_air_visit['holiday_before_visit'] = df_air_visit.holiday_flg.shift(1)
df_air_visit.holiday_before_visit.fillna(0,inplace=True)
df_air_visit['holiday_after_visit'] = df_air_visit.holiday_flg.shift(-1)
df_air_visit.holiday_after_visit.fillna(0,inplace=True)


# In[38]:


df_air_visit.describe()


# In[39]:


# visitors by day of week and holiday
df_visitors_by_dow = df_air_visit.groupby(['day_of_week','holiday_flg'])                              .agg({'visitors':'mean'})                              .reset_index() 

df_visitors_by_dow['day_of_week'] = pd.Categorical(df_visitors_by_dow['day_of_week'],categories=dow, ordered=True)
df_visitors_by_dow.sort_values(by='day_of_week', inplace=True)
sns.factorplot(x='day_of_week', y='visitors', data=df_visitors_by_dow, hue='holiday_flg', aspect=3)


# In[40]:


# visits by day of week and holiday after visit
df_visitors_by_dow = df_air_visit.groupby(['day_of_week','holiday_after_visit'])                              .agg({'visitors':'mean'})                              .reset_index() 

df_visitors_by_dow['day_of_week'] = pd.Categorical(df_visitors_by_dow['day_of_week'],categories=dow, ordered=True)
df_visitors_by_dow.sort_values(by='day_of_week', inplace=True)
sns.factorplot(x='day_of_week', y='visitors', data=df_visitors_by_dow, hue='holiday_after_visit', aspect=3)


# In[41]:


# Visitors from reservations vs Total visitors
air_res_store = df_res_merged.groupby(['air_store_id', 'calendar_date'])                             .agg({'reserve_visitors':'sum'})                              .reset_index()                              .rename(columns={'calendar_date':'visit_date'})
air_res_store['visit_date'] = air_res_store.visit_date.apply(lambda x: str(x))
df_air_visit['visit_date'] = df_air_visit.visit_date.apply(lambda x: str(x))
air_res_store['air_store_id'] = air_res_store.air_store_id.apply(lambda x: str(x))
df_air_visit['air_store_id'] = df_air_visit.air_store_id.apply(lambda x: str(x))
air_res = pd.merge(df_air_visit, air_res_store,  how='left', 
                         left_on=['air_store_id','visit_date'], right_on = ['air_store_id','visit_date'])
air_res.reserve_visitors.fillna(0.0, inplace=True)
air_res_date = air_res.groupby('visit_date')                 .agg({'reserve_visitors':'sum', 'visitors':'sum'})                  .reset_index()
air_res_date['visit_date'] = air_res_date.visit_date.apply(lambda x: pd.to_datetime(x).date())
f, ax1 = plt.subplots(figsize=(16, 10))
ax1.plot_date(x='visit_date', y=air_res_date.loc[:,['reserve_visitors', 'visitors']], data=air_res_date, ms=3, ls='solid', lw=1)
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.grid(b=None, axis='x')
ax1.set_ylabel('Visitors')
ax1.legend(['Visitors from reservations', 'Total visitors'])


# In[42]:


# Plot total daily visitors distribution before and after jump in visitors(Jul 1st 2016)

date_jump = pd.to_datetime('2016-07-01').date()

df_total_vis = df_air_visit.groupby(['visit_date'])                           .agg({'visitors':'sum'})                            .reset_index() 
df_total_vis['before_20160701'] = df_total_vis.visit_date.apply(lambda x: pd.to_datetime(x).date()<date_jump)

sns.kdeplot(df_total_vis[df_total_vis['before_20160701'] == True].visitors, color= "b", lw= 3, label= "Before 2016-07-01")
sns.kdeplot(df_total_vis[df_total_vis['before_20160701'] == False].visitors, color= "g", lw= 3, label= "On/After 2016-07-01")


# In[43]:


# Reservation visits per season
df_air_visit_by_season = df_air_visit.groupby(['visit_month'])                              .agg({'visitors':'sum'})                              .reset_index() 
df_air_visit_by_season['total_size'] = df_air_visit_by_season.visitors.sum()
df_air_visit_by_season.sort_values(by='total_size', inplace=True)
df_air_visit_by_season['visit_pct'] = (df_air_visit_by_season['visitors'] / df_air_visit_by_season['total_size'])*100
pal=['blue','blue','green','green','green', 'red','red','red', 'orange','orange','orange', 'blue']
sns.barplot("visit_month", "visit_pct", data=df_air_visit_by_season, palette=pal, 
            order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])


# In[44]:


# Visitors per genre
df_air_visit['visit_date']= df_air_visit.visit_date.apply(lambda x: pd.to_datetime(x).date())
df_visitors_by_genre = df_air_visit.groupby(['air_genre_2','air_genre_name','visit_date'])                              .agg({'visitors':'sum'})                              .reset_index() 
g = sns.FacetGrid(df_visitors_by_genre, col="air_genre_name", col_wrap=4,
                  hue='air_genre_2',aspect=1.5,palette='Set1')
for q in g.axes:
    q.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    q.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
g.map(plt.plot, "visit_date", "visitors")
g.set(yscale='log')
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Daily visitors per genre')


# In[45]:


df_air_visit['visit_date']= df_air_visit.visit_date.apply(lambda x: pd.to_datetime(x).date())
df_visitors_by_genre = df_air_visit.groupby(['air_genre_2','air_genre_name','day_of_week'])                              .agg({'visitors':'mean'})                              .reset_index() 

g = sns.FacetGrid(df_visitors_by_genre, col="air_genre_name", col_wrap=4, sharey=False,
                  hue='air_genre_2',aspect=1.5,palette='Set1')
g.map(sns.barplot, "day_of_week", "visitors",order=dow)
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Mean visitors per day by genre')


# In[46]:


# Visitors from reservations vs Total visitors by season
g=sns.lmplot('reserve_visitors', 'visitors', air_res[(air_res.reserve_visitors>0)&(air_res.reserve_visitors<250)],
          hue='visit_season',legend_out=False,scatter_kws={'alpha':0.5})
g.set(xlim=(0,200), ylim=(0,250))
g.fig.set_figwidth(16)
g.fig.set_figheight(7)


# In[47]:


f, (ax1,ax2) = plt.subplots(1,2,figsize=(12,7))

store_counts_air=pd.DataFrame(df_air_store.air_area_name.value_counts()).reset_index()
store_counts_air.rename(columns={'air_area_name':'store_count','index':'air_area_name'},inplace=True)
store_counts_hpg=pd.DataFrame(df_hpg_store.hpg_area_name.value_counts()).reset_index()
store_counts_hpg.rename(columns={'hpg_area_name':'store_count','index':'hpg_area_name'},inplace=True)

sns.barplot(y='air_area_name',x='store_count',data=store_counts_air.head(25),color='b', ax=ax1)
sns.barplot(y='hpg_area_name',x='store_count',data=store_counts_hpg.head(25),color='g', ax=ax2)

ax1.set_title('Air')
ax2.set_title('HPG')
plt.tight_layout()


# In[48]:


# Plot individual restaurants on map based on lat and long

# Group and count stores as map function can't handle plotting individual stores
df_hpg_store_merged = df_hpg_store.merge(df_store_id_rel,on='hpg_store_id', how='left').merge(
                                         df_air_store,on='air_store_id', how='left',suffixes=('_hpg','_air'))

df_air_store_merged = df_air_store.merge(df_store_id_rel,on='air_store_id', how='left').merge(
                                         df_hpg_store,on='hpg_store_id', how='left',suffixes=('_air','_hpg'))

df_hpg_store_merged = df_hpg_store_merged[~df_hpg_store_merged.hpg_store_id.isin(df_air_store_merged.hpg_store_id)]

df_hpg_store_merged = df_hpg_store_merged.groupby(['hpg_area_name','latitude_hpg','longitude_hpg','hpg_genre_name'])                                         .agg({'hpg_store_id':'size'})                                          .reset_index() 
        
df_air_store_merged = df_air_store_merged.groupby(['air_area_name','latitude_air','longitude_air','air_genre_name'])                                         .agg({'air_store_id':'size'})                                          .reset_index() 

maxlat = df_air_store_merged.latitude_air.mean()
maxlong = df_air_store_merged.longitude_air.mean()
        
# Use folium to map
m = folium.Map(
    location=[maxlat, maxlong],
    tiles='CartoDB positron',
    zoom_start=5
)

marker_cluster = MarkerCluster(
    name='Restaurants',
).add_to(m)


for point in df_air_store_merged.iterrows():
    lat=point[1]['latitude_air']
    lon=point[1]['longitude_air']
    store=point[1]['air_store_id']
    genre=point[1]['air_genre_name']
    desc=point[1]['air_area_name']
    folium.Marker((lat,lon), popup='lon:{}<br>lat:{}<br>stores:{}<br>area:{}<br>genre:{}'.format(lon, lat, store, desc,genre),  
                  icon=folium.Icon(color='darkblue', icon_color='white', 
                                   icon='male', angle=0, prefix='fa')).add_to(marker_cluster)

for point in df_hpg_store_merged.iterrows():
    lat=point[1]['latitude_hpg']
    lon=point[1]['longitude_hpg']
    genre=point[1]['hpg_genre_name']
    desc=point[1]['hpg_area_name']
    store=point[1]['hpg_store_id']
    folium.Marker((lat,lon), popup='lon:{}<br>lat:{}<br>stores:{}<br>area:{}<br>genre:{}'.format(lon, lat, store, desc,genre),  
                  icon=folium.Icon(color='red', icon_color='white', 
                                   icon='male', angle=0, prefix='fa')).add_to(marker_cluster)    
    
folium.LayerControl().add_to(m)
    
m


# In[49]:


# plot restaurants as a heatmap

heat = df_air_store_merged[['latitude_air','longitude_air']].apply(pd.to_numeric)
heat=heat.values.tolist()
m = folium.Map(
    location=[maxlat, maxlong],
    tiles='CartoDB positron',
    zoom_start=5
)
m.add_child(HeatMap(heat, radius=15, min_opacity=.5))
m


# In[50]:




