#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('html', '', '<span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at \'In [49]\'.</span>')


# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


## Read in data

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')


# In[3]:


## Check for null values

train.apply(lambda x: x.isnull().sum()/len(x))


# In[4]:


## Change to datetime

train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])


# In[5]:


## Split into 2 datsets

cases_train = train[train['Target'] == 'ConfirmedCases'].drop(['Target'], axis = 1).rename(
    columns = {'TargetValue': 'ConfirmedCases'})
fatal_train = train[train['Target'] == 'Fatalities'].drop(['Target'], axis = 1).rename(
    columns = {'TargetValue': 'Fatalities'})


# In[6]:


## Imports

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import folium
from folium import plugins


# In[7]:


## For bar graphs

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width()
            _y = p.get_y() + p.get_height()/2 
            value = '{:.2f}'.format(-p.get_width())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# In[8]:


def Find_pattern (df, list_of_countries, pattern):
    
    list_matches = []
    
    for country in list_of_countries:
    
        cntry = df[df['Country_Region'] == country]
        cntry['changes_bool'] = cntry['changes'] < 0
    
        matches = [cntry.index[i - len(pattern)] 
         for i in range(len(pattern), len(cntry))
         if all(cntry['changes_bool'][i-len(pattern):i] == pattern)]
        
        if len(matches) > 0:
            list_matches += matches
        
    return list_matches


# In[9]:


def Graph_covid (df, country, date = None):
    plt.figure(figsize = (15, 10))
   
    ax = sns.lineplot(x = 'Date', y = 'ConfirmedCases', data = df, label = 'Confirmed Cases', color = 'grey')
    ax.fill_between(df['Date'], df['ConfirmedCases'], color = 'silver')
    
    ax1 = ax.twinx()
    sns.lineplot(x = 'Date', y = 'mortality', data = df, label = 'Mortality rate', ax = ax1, color = 'k')
    
    if date != None:
        fm_date = pd.to_datetime(date)
        ax.axvline(fm_date, color = 'red')
    
        # Annotate

        ymin, ymax = plt.ylim()
        text = 'National lockdown \n' + date
        ax.annotate(text,
                xy  = (fm_date, ymax),
                xycoords='data',
                xytext=(10, 10), textcoords='offset points',
                annotation_clip=False,
                arrowprops = {'width': 1, 'headwidth': 1, 'headlength': 1, 'shrink':0.05},
                fontsize=12)
    
    plt.title(country)
    plt.show()


# In[10]:


def Graph_changes (df, country, num_day, ax = None, **kwargs):
    
    ax = ax or plt.gca()
    
    cntry = df[df['Country_Region'] == country]

    ax.plot(cntry['Date'], cntry['changes'], linewidth = 3, **kwargs)
    ax.axhline(0, color = 'k')
    return ax.set_title('Changes in confirmed cases in ' + country)


# In[11]:


def Classify_den (column):
 
    den_level = ''
    if column < edges[1]:
        den_level = 'low'
    elif (column > edges[1]) and (column < edges[2]):
        den_level = 'medium'
    else:
        den_level = 'high'
        
    return den_level


# In[12]:


def Graph_testing_data (df, country):
    
    cntry = df[df['Country_Region'] == country]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = False, figsize = (20, 10))
    
    ax1.fill_between(cntry['Date'], cntry['total_tested'], color = 'silver', 
                    label = 'Total tested')
    ax1.fill_between(cntry['Date'], cntry['positive'], color = 'tomato',
                    label = 'Total positive')
    ax1.set_title('Number of tests and positive cases')
    ax1.legend()
   
    ax2.fill_between(cntry['Date'], cntry['positive'], color = 'tomato',
                    label = 'Total positive')
    ax2.fill_between(cntry['Date'], cntry['recovered'], color = 'lightgreen',
                    label = 'Total recoveries')
    ax2.set_title('Number of positve cases and recoveries')
    ax2.legend()

    plt.suptitle(country)
    
    return ax1, ax2


# In[13]:


## Confimed cases in US states

plt.figure(figsize = (20, 10))
gb_us = cases_train[cases_train['Country_Region'] == 'US'].groupby(['Province_State']).sum().sort_values(['ConfirmedCases'], 
                                                                                                        ascending = False)
sns.barplot(x = gb_us.index, y = 'ConfirmedCases', data = gb_us, palette = 'BuGn_r')

plt.xticks(rotation = 70)
plt.xlabel('US State')
plt.ylabel('Confirmed Cases')
plt.title('Confirmed cases in the US by state')


# In[14]:


## grab the states' populations 

gb_us.drop(['Population'], axis = 1, inplace = True)
state_pop = cases_train[cases_train['Country_Region'] == 'US'][['Province_State', 'Population']].drop_duplicates().groupby(['Province_State']).sum()
gb_us = gb_us.merge(state_pop, how = 'outer', right_index = True, left_index = True)


# In[15]:


## Number of cases per 10,000 people

gb_us['cases per 10000'] = gb_us['ConfirmedCases']/gb_us['Population']*10000

plt.figure(figsize = (20, 10))
ax = sns.barplot(x = gb_us.index, y = 'cases per 10000', data = gb_us.sort_values(['cases per 10000'], ascending = False), color = 'b')

plt.xticks(rotation = 70)
plt.xlabel('US State')
plt.ylabel('Confirmed Cases')
plt.title('Number of cases per 10000 people in the US by state')
#show_values_on_bars(ax)


# In[16]:


## US states coordinates
coords = pd.read_csv('/kaggle/input/covid19-forecasting-metadata/region_metadata.csv')

us_coords = coords[coords['Country_Region'] == 'US'].set_index(['Province_State']).drop(['population', 'area',
                                                                                     'continent', 'Country_Region'], axis = 1)
gb_us = gb_us.merge(us_coords, how = 'outer', left_index = True, right_index = True)
gb_us.dropna(axis = 0, inplace = True)
gb_us.head()                                                


# In[17]:


## Create an empty map

folium_map = folium.Map(location=[37.0902,-95.7129],# USA coordinates
                        zoom_start=4,tiles='openstreetmap')

for index, state in gb_us.iterrows():
    
    #add pop up
    
    popup_text = """{}, {}, {}"""
    popup_text = popup_text.format(index, state['ConfirmedCases'], state['cases per 10000'])
    
    color = 'red'
    ## Size of bubbles are number of cases
    size = state['ConfirmedCases']/10000
    
    folium.CircleMarker(location = (state['lat'], state['lon']),
                        weight=2,radius = size, color = color, opacity = 10,
                        fill = True, fill_color = color, popup = popup_text).add_to(folium_map)
    
folium_map


# In[18]:


## US fatalities by states
gb_us_fatal = fatal_train[fatal_train['Country_Region'] == 'US'].groupby(['Province_State']).sum().sort_values(['Fatalities'], 
                                                                                                        ascending = False)
gb_us_tot = gb_us.merge(gb_us_fatal[['Fatalities']], how = 'left', left_index = True, right_index = True)

## to create a diverging graph
gb_us_tot['fatal_viz'] = -gb_us_tot['Fatalities']
gb_us_tot.sort_values(['ConfirmedCases'], ascending = False, inplace = True)
gb_us_tot.head()


# In[19]:


fig, ax = plt.subplots(figsize = (20,20))

sns.set_color_codes("muted")
sns.barplot(x = "fatal_viz", y = gb_us_tot.index, data = gb_us_tot,
            label = 'Fatalities', color = 'b')

show_values_on_bars(ax)

sns.set_color_codes("pastel")
sns.barplot(x = "ConfirmedCases", y = gb_us_tot.index, data = gb_us_tot,
            label = "ConfirmedCases", color = 'b')


ax.legend(ncol=2, loc="center", frameon=True)
ax.set(ylabel="",
       xlabel="Deaths/Cases", title = 'Number of confirmed cases and deaths in US states')

sns.despine(left=True, bottom=True)


# In[20]:


cases_by_date = cases_train.groupby(['Country_Region', 'Date'], as_index = False).sum()
deaths_by_date = fatal_train.groupby(['Country_Region', 'Date'], as_index = False).sum()

## Extract countries with the most number of confirmed cases as of May 10, 2020
top6 = cases_by_date[cases_by_date['Date'] == cases_by_date['Date'].max()].sort_values(['ConfirmedCases'],
                                                                                      ascending = False)['Country_Region'][:6]
top6


# In[21]:


## Calculate daily mortality rates

cases_by_date['mortality'] = deaths_by_date['Fatalities'] / cases_by_date['ConfirmedCases']
cases_by_date['mortality'].fillna(0, inplace = True)
cases_by_date['mortality'].replace(np.inf, 0, inplace = True)


# In[22]:


## Changes in confirmed cases
changes = []
for country in cases_by_date['Country_Region'].unique():
    
    cntry = cases_by_date[cases_by_date['Country_Region'] == country]
    ## Subtract current date from previous date
    cntry_changes = cntry['ConfirmedCases'].diff().to_list()
    
    changes += cntry_changes
    
cases_by_date['changes'] = changes
cases_by_date.fillna(0, inplace = True)
cases_by_date


# In[23]:


fig, ax = plt.subplots(3,2, figsize = (30, 20))

Graph_changes(cases_by_date, top6.iloc[0],num_day = 3, ax = ax[0,0], color = 'purple')
Graph_changes(cases_by_date, top6.iloc[1],num_day = 3, ax = ax[0,1], color = 'grey')
Graph_changes(cases_by_date, top6.iloc[2],num_day = 3, ax = ax[1,0], color = 'r')
Graph_changes(cases_by_date, top6.iloc[3],num_day = 3, ax = ax[1,1], color = 'b')
Graph_changes(cases_by_date, top6.iloc[4],num_day = 3, ax = ax[2,0], color = 'green')
Graph_changes(cases_by_date, top6.iloc[5],num_day = 3, ax = ax[2,1], color = 'cyan')

plt.suptitle('Changes in Confirmed Cases overtime')


# In[24]:


#plt.figure(figsize = (20, 10))

cntry1 = cases_by_date[cases_by_date['Country_Region'] == top6.iloc[0]]
cntry2 = cases_by_date[cases_by_date['Country_Region'] == top6.iloc[1]]
cntry3 = cases_by_date[cases_by_date['Country_Region'] == top6.iloc[2]]
cntry4 = cases_by_date[cases_by_date['Country_Region'] == top6.iloc[3]]
cntry5 = cases_by_date[cases_by_date['Country_Region'] == top6.iloc[4]]
cntry6 = cases_by_date[cases_by_date['Country_Region'] == top6.iloc[5]]

#plt.figure(figsize = (20, 10))

Graph_covid(cntry1, top6.iloc[0], '2020-03-22')
Graph_covid(cntry2, top6.iloc[1], '2020-05-11')
Graph_covid(cntry3, top6.iloc[2], '2020-05-12')
Graph_covid(cntry4, top6.iloc[3], '2020-03-24')
Graph_covid(cntry5, top6.iloc[4], '2020-03-15')
Graph_covid(cntry6, top6.iloc[5], '2020-03-18')

plt.show()


# In[25]:


## Replace the 15 with the mean

cntry6['mortality'].replace(15, cntry6['mortality'].mean(), inplace = True)
plt.plot; Graph_covid(cntry6, 'UK')


# In[26]:


## Fatalities from deaths_by_date

by_date = cases_by_date.merge(deaths_by_date[['Country_Region', 
                                             'Date', 'Fatalities']],
                             how = 'outer', on = ['Country_Region', 'Date'])
## Change type to datetime
by_date['Date'] = pd.to_datetime(by_date['Date'])


# In[27]:


## Countries with decreasing number of cases in 6 consecutive days

## Only use data from May
may_covid = by_date[by_date['Date'] >= '2020-05-05']
## Get the names of the countries in descending order
sorted_cntries = may_covid.groupby(['Country_Region'], as_index = False).sum().sort_values(['ConfirmedCases'],
                                                                                              ascending = False)['Country_Region']
pattern = [True] * 6

cntr_ind = Find_pattern(may_covid, sorted_cntries, pattern)

recov_cntries = []

## Get the names 
for index in cntr_ind:
    country = by_date.iloc[index]['Country_Region']
    recov_cntries.append(country)
    
## Remove duplicates
recov_cntries = list(set(recov_cntries))
recov_cntries


# In[28]:


## Countries that are recovering

revcntry1 = by_date[by_date['Country_Region'] == recov_cntries[0]]
revcntry2 = by_date[by_date['Country_Region'] == recov_cntries[1]]
revcntry3 = by_date[by_date['Country_Region'] == recov_cntries[2]]
revcntry4 = by_date[by_date['Country_Region'] == recov_cntries[3]]

Graph_covid(revcntry3, 'Norway', '2020-03-12')
Graph_covid(revcntry1, 'Turkey', '2020-05-01')
Graph_covid(revcntry4, 'Belgium', '2020-03-18')
Graph_covid(revcntry2, 'Ghana', '2020-03-30')


# In[29]:


## Get the coordinates

by_date_tot = by_date.merge(coords.drop(['population', 'Province_State'], axis = 1), how = 'outer', on = ['Country_Region'])
by_date_tot


# In[30]:


## Histograms of confirmed cases by continent

facet = sns.FacetGrid(by_date_tot, row = 'continent', hue = 'continent', size = 7,sharex = False, sharey = False)
facet.map(plt.hist, 'ConfirmedCases')
facet.axes[1,0].set_xlim(0,27000)
facet.axes[2,0].set_xlim(0,1800)


# In[31]:


plt.figure(figsize = (8, 10))
ax = sns.barplot(y = 'ConfirmedCases', x = 'continent', 
                 data = by_date_tot.groupby(['continent'], as_index = False).sum().sort_values(['ConfirmedCases']),
                palette = 'GnBu_d')
ax.set_title('Confirmed cases by continent')


# In[32]:


## Number of cases per 10000 km2
by_date_tot['Cases per 100 km2'] = np.round(by_date_tot['ConfirmedCases'] / by_date_tot['area'] * 100)
by_date_tot['Cases per 100 km2'].fillna(0, inplace = True)
by_date_tot['Cases per 100 km2'].replace(np.inf, 0, inplace = True)

## Split densities into 3 categories 

counts, edges = np.histogram(by_date_tot['density'], 3)
print(counts)
print(edges)

## Classify density into low, medium, and high

by_date_tot['density level'] = by_date_tot['density'].apply(Classify_den)
by_date_tot['density level'].value_counts()


# In[33]:


## Histogram of confirmed cases by density level
facet = sns.FacetGrid(by_date_tot, 
                      col = 'density level', hue = 'density level', 
                      col_order = ['low', 'medium', 'high'],
                       size = 10,sharex = False, sharey = False,
                        palette = 'GnBu_d', margin_titles = True)
facet.map(plt.hist, 'Cases per 100 km2')
facet.axes[0, 0].ticklabel_format(style='plain')
facet.axes[0, 2].ticklabel_format(style='plain')


# In[34]:


testing = pd.read_csv('/kaggle/input/covid19testing/tested_worldwide.csv')
testing.head()


# In[35]:


## Check for any mismatches in country names
set(testing['Country_Region']) - set(by_date_tot['Country_Region'])


# In[36]:


## Resolve naming issues
testing['Country_Region'].replace(['United States', 'South Korea', 'Democratic Republic of the Congo', 'Taiwan'],
                                 ['US', 'Korea, South', 'Congo (Brazzaville)', 'Taiwan*'], inplace = True)
testing['Date'] = pd.to_datetime(testing['Date'])

new_testing = testing[['Date', 'Country_Region', 'positive',
                      'hospitalizedCurr', 'recovered', 'total_tested']]


# In[37]:


## Testing info
testing_by_date = by_date_tot.merge(new_testing, how = 'left', on = ['Country_Region', 'Date'])
## Fill in NAs with 0
testing_by_date.fillna(0, inplace = True)
testing_by_date.head()


# In[38]:


Graph_testing_data(testing_by_date, 'US')
Graph_testing_data(testing_by_date, 'Korea, South')
Graph_testing_data(testing_by_date, 'Russia')
Graph_testing_data(testing_by_date, 'India')
Graph_testing_data(testing_by_date, 'Italy')


# In[39]:


## Create a new dataframe to store aggregated confirmed cases and number of people tested
gb = testing_by_date[['Country_Region', 'Population', 'total_tested', 'ConfirmedCases']]
gb = gb.groupby(['Country_Region']).agg(                                       Population = ('Population', 'mean'),
                                       total_tested = ('total_tested', 'sum'),
                                       ConfirmedCases = ('ConfirmedCases', 'sum'))
gb['% population tested'] = gb['total_tested'] / gb['Population'] * 100
gb['% population infected'] = gb['ConfirmedCases'] / gb['Population'] * 100

## Sort by the tested population and confirmed cases
gb.sort_values(['% population tested', '% population infected'], ascending = False, inplace = True)

## Remove the US
gb.drop(['US'], axis = 0, inplace = True)


# In[40]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15,15))


## Countries with high testing populations
sns.set_color_codes("muted")
sns.barplot(y = "% population tested", x = gb[:20].index, data = gb[:20], color = 'khaki', 
            label = '% population tested', ax = ax1)

ax1_dual = ax1.twinx()

sns.set_color_codes("pastel")
sns.barplot(y = '% population infected', x = gb[:20].index, data = gb[:20], color = 'k',
                  label = '% population infected', ax = ax1_dual, alpha = 0.5)

ax1.legend(ncol=2, loc="upper right", frameon=True)
ax1.set(ylabel='% population tested',
       xlabel="% Population", title = 'Percent of population infected vs. testing population in countries with highest Covid19 testing populations')


## Countries with lower testing populations 

low_20 = gb[(gb['ConfirmedCases'] != 0) & (gb['total_tested'] != 0)][-20:]

sns.set_color_codes("muted")
sns.barplot(y = "% population tested", x = low_20.index, 
            data = low_20, color = 'khaki', 
            label = '% population tested', ax = ax2)

ax2_dual = ax2.twinx()

sns.set_color_codes("pastel")
sns.barplot(y = '% population infected', x = low_20.index,
                data = low_20, color = 'k', ax = ax2_dual, alpha = 0.5)

ax2.legend(ncol=2, loc="upper right", frameon=True)
ax2.set(ylabel="% population tested",
       xlabel='Country', title = 'Percent of population infected vs. testing population in countries with lowest Covid19 testing populations')


# In[41]:


## Resolve comma issue later on
testing_by_date['Country_Region'].replace('Korea, South', 'South Korea', inplace = True)

spread = {}

## Date on which the first case of covid was detected in each country
for i in testing_by_date['Country_Region'].unique():
    
    df = testing_by_date[(testing_by_date['Country_Region'] == i) & testing_by_date['ConfirmedCases'] != 0]
    start_date = df['Date'].min()
    
    if start_date in spread:
        spread[start_date] = spread.get(start_date) + ',' + i
    else:
        spread[start_date] = i


# In[42]:


## Sort by date

sorted_spread = sorted(spread.items(), key = lambda kv:(kv[0],kv[1]))
sorted_spread


# In[43]:


dates = []
num_cntries = []
cntries = []

for n in sorted_spread:
        
        ## Collect the dates
        dates.append(n[0])
        ## Names of Countries infected
        cntries.append(n[1])
        ## Number of countries infected
        num_cntries.append(len(n[1].split(',')))


# In[44]:


plt.figure(figsize = (15, 10))

ax = sns.barplot(x = dates, y = num_cntries, color = 'lightblue')
ax.set(xlabel = 'Date', ylabel = 'Number of countries infected', title = 'Covid19 spread by date')
plt.xticks(rotation = 45)


# In[45]:


## New dataframe to store the timeline of covid19 spread

covid_spread_df = pd.DataFrame(columns = ['Date', 'Country_Region'], data = sorted_spread)
covid_spread_df = covid_spread_df.set_index(['Date']).apply(lambda x: x.str.split(',').explode()).reset_index()
covid_spread_df


# In[46]:


## Prepare continents dataframe, drop duplicates to remove all provinces
continents = coords[['Country_Region', 'continent', 'lat', 'lon']].drop_duplicates(subset = ['Country_Region']).replace('Korea, South', 'South Korea')

covid_spread_df = covid_spread_df.merge(continents, how = 'outer', on = 'Country_Region')
covid_spread_df


# In[47]:


from datetime import datetime


# In[48]:


def color_by_month(row):
    if row['Date'] < datetime.strptime('2020-02-01', '%Y-%m-%d'):
        return ['background-color: yellow'] * 5
    if (row.Date >= datetime.strptime('2020-02-01', '%Y-%m-%d')) & (row.Date < datetime.strptime('2020-03-01', '%Y-%m-%d')):
        return ['background-color: white']*5
    if (row.Date >= datetime.strptime('2020-03-01', '%Y-%m-%d')) & (row.Date <= datetime.strptime('2020-04-01', '%Y-%m-%d')):
        return ['background-color: lightgreen']*5
    if (row.Date >= datetime.strptime('2020-04-01', '%Y-%m-%d')) & (row.Date <= datetime.strptime('2020-05-01', '%Y-%m-%d')):
        return ['background-color: lightblue']*5
    elif row['Date'] >= datetime.strptime('2020-05-01', '%Y-%m-%d'):
        return ['background-color: red']*5


# In[49]:


covid_spread_df.style.apply(color_by_month, axis=1)


# In[ ]:




