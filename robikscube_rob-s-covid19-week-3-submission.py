#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os, gc, pickle, copy, datetime, warnings
import pycountry

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
pd.options.display.float_format = '{:.2f}'.format


# In[2]:


# Read in data
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

tt = pd.concat([train, test], sort=False)
tt = train.merge(test, on=['Province_State','Country_Region','Date'], how='outer')

# concat Country/Region and Province/State
def name_place(x):
    try:
        x_new = x['Country_Region'] + "_" + x['Province_State']
    except:
        x_new = x['Country_Region']
    return x_new
tt['Place'] = tt.apply(lambda x: name_place(x), axis=1)
# tt = tt.drop(['Province_State','Country_Region'], axis=1)
tt['Date'] = pd.to_datetime(tt['Date'])
tt['doy'] = tt['Date'].dt.dayofyear
tt['dow'] = tt['Date'].dt.dayofweek
tt['hasProvidence'] = ~tt['Province_State'].isna()


country_meta = pd.read_csv('../input/covid19-forecasting-metadata/region_metadata.csv')
tt = tt.merge(country_meta, how='left')

country_date_meta = pd.read_csv('../input/covid19-forecasting-metadata/region_date_metadata.csv')
#tt = tt.merge(country_meta, how='left')

tt['HasFatality'] = tt.groupby('Place')['Fatalities'].transform(lambda x: x.max() > 0)
tt['HasCases'] = tt.groupby('Place')['ConfirmedCases'].transform(lambda x: x.max() > 0)

first_case_date = tt.query('ConfirmedCases >= 1').groupby('Place')['Date'].min().to_dict()
ten_case_date = tt.query('ConfirmedCases >= 10').groupby('Place')['Date'].min().to_dict()
hundred_case_date = tt.query('ConfirmedCases >= 100').groupby('Place')['Date'].min().to_dict()
first_fatal_date = tt.query('Fatalities >= 1').groupby('Place')['Date'].min().to_dict()
ten_fatal_date = tt.query('Fatalities >= 10').groupby('Place')['Date'].min().to_dict()
hundred_fatal_date = tt.query('Fatalities >= 100').groupby('Place')['Date'].min().to_dict()

tt['First_Case_Date'] = tt['Place'].map(first_case_date)
tt['Ten_Case_Date'] = tt['Place'].map(ten_case_date)
tt['Hundred_Case_Date'] = tt['Place'].map(hundred_case_date)
tt['First_Fatal_Date'] = tt['Place'].map(first_fatal_date)
tt['Ten_Fatal_Date'] = tt['Place'].map(ten_fatal_date)
tt['Hundred_Fatal_Date'] = tt['Place'].map(hundred_fatal_date)

tt['Days_Since_First_Case'] = (tt['Date'] - tt['First_Case_Date']).dt.days
tt['Days_Since_Ten_Cases'] = (tt['Date'] - tt['Ten_Case_Date']).dt.days
tt['Days_Since_Hundred_Cases'] = (tt['Date'] - tt['Hundred_Case_Date']).dt.days
tt['Days_Since_First_Fatal'] = (tt['Date'] - tt['First_Fatal_Date']).dt.days
tt['Days_Since_Ten_Fatal'] = (tt['Date'] - tt['Ten_Fatal_Date']).dt.days
tt['Days_Since_Hundred_Fatal'] = (tt['Date'] - tt['Hundred_Fatal_Date']).dt.days

# Merge smoking data
smoking = pd.read_csv("../input/smokingstats/share-of-adults-who-smoke.csv")
smoking = smoking.rename(columns={'Smoking prevalence, total (ages 15+) (% of adults)': 'Smoking_Rate'})
smoking_dict = smoking.groupby('Entity')['Year'].max().to_dict()
smoking['LastYear'] = smoking['Entity'].map(smoking_dict)
smoking = smoking.query('Year == LastYear').reset_index()
smoking['Entity'] = smoking['Entity'].str.replace('United States', 'US')

tt = tt.merge(smoking[['Entity','Smoking_Rate']],
         left_on='Country_Region',
         right_on='Entity',
         how='left',
         validate='m:1') \
    .drop('Entity', axis=1)

# Country data
country_info = pd.read_csv('../input/countryinfo/covid19countryinfo.csv')


tt = tt.merge(country_info, left_on=['Country_Region','Province_State'],
              right_on=['country','region'],
              how='left',
              validate='m:1')

# State info from wikipedia
us_state_info = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states_by_population')[0]     [['State','Population estimate, July 1, 2019[2]']]     .rename(columns={'Population estimate, July 1, 2019[2]' : 'Population'})
#us_state_info['2019 population'] = pd.to_numeric(us_state_info['2019 population'].str.replace('[note 1]','').replace('[]',''))

tt = tt.merge(us_state_info[['State','Population']],
         left_on='Province_State',
         right_on='State',
         how='left')

tt['pop'] = pd.to_numeric(tt['pop'].str.replace(',',''))
tt['pop'] = tt['pop'].fillna(tt['Population'])
tt['pop'] = pd.to_numeric(tt['pop'])

tt['pop_diff'] = tt['pop'] - tt['Population']
tt['Population_final'] = tt['Population']
tt.loc[~tt['hasProvidence'], 'Population_final'] = tt.loc[~tt['hasProvidence']]['pop']

tt['Confirmed_Cases_Diff'] = tt.groupby('Place')['ConfirmedCases'].diff()
tt['Fatailities_Diff'] = tt.groupby('Place')['Fatalities'].diff()
max_date = tt.dropna(subset=['ConfirmedCases'])['Date'].max()
tt['gdp2019'] = pd.to_numeric(tt['gdp2019'].str.replace(',',''))


# In[3]:


# Correcting population for missing countries
# Googled their names and copied the numbers here
pop_dict = {'Angola': int(29.78 * 10**6),
            'Australia_Australian Capital Territory': 423_800,
            'Australia_New South Wales': int(7.544 * 10**6),
            'Australia_Northern Territory': 244_300,
            'Australia_Queensland' : int(5.071 * 10**6),
            'Australia_South Australia' : int(1.677 * 10**6),
            'Australia_Tasmania': 515_000,
            'Australia_Victoria': int(6.359 * 10**6),
            'Australia_Western Australia': int(2.589 * 10**6),
            'Brazil': int(209.3 * 10**6),
            'Canada_Alberta' : int(4.371 * 10**6),
            'Canada_British Columbia' : int(5.071 * 10**6),
            'Canada_Manitoba' : int(1.369 * 10**6),
            'Canada_New Brunswick' : 776_827,
            'Canada_Newfoundland and Labrador' : 521_542,
            'Canada_Nova Scotia' : 971_395,
            'Canada_Ontario' : int(14.57 * 10**6),
            'Canada_Prince Edward Island' : 156_947,
            'Canada_Quebec' : int(8.485 * 10**6),
            'Canada_Saskatchewan': int(1.174 * 10**6),
            'China_Anhui': int(62 * 10**6),
            'China_Beijing': int(21.54 * 10**6),
            'China_Chongqing': int(30.48 * 10**6),
            'China_Fujian' :  int(38.56 * 10**6),
            'China_Gansu' : int(25.58 * 10**6),
            'China_Guangdong' : int(113.46 * 10**6),
            'China_Guangxi' : int(48.38 * 10**6),
            'China_Guizhou' : int(34.75 * 10**6),
            'China_Hainan' : int(9.258 * 10**6),
            'China_Hebei' : int(74.7 * 10**6),
            'China_Heilongjiang' : int(38.31 * 10**6),
            'China_Henan' : int(94 * 10**6),
            'China_Hong Kong' : int(7.392 * 10**6),
            'China_Hubei' : int(58.5 * 10**6),
            'China_Hunan' : int(67.37 * 10**6),
            'China_Inner Mongolia' :  int(24.71 * 10**6),
            'China_Jiangsu' : int(80.4 * 10**6),
            'China_Jiangxi' : int(45.2 * 10**6),
            'China_Jilin' : int(27.3 * 10**6),
            'China_Liaoning' : int(43.9 * 10**6),
            'China_Macau' : 622_567,
            'China_Ningxia' : int(6.301 * 10**6),
            'China_Qinghai' : int(5.627 * 10**6),
            'China_Shaanxi' : int(37.33 * 10**6),
            'China_Shandong' : int(92.48 * 10**6),
            'China_Shanghai' : int(24.28 * 10**6),
            'China_Shanxi' : int(36.5 * 10**6),
            'China_Sichuan' : int(81.1 * 10**6),
            'China_Tianjin' : int(15 * 10**6),
            'China_Tibet' : int(3.18 * 10**6),
            'China_Xinjiang' : int(21.81 * 10**6),
            'China_Yunnan' : int(45.97 * 10**6),
            'China_Zhejiang' : int(57.37 * 10**6),
            'Denmark_Faroe Islands' : 51_783,
            'Denmark_Greenland' : 56_171,
            'France_French Guiana' : 290_691,
            'France_French Polynesia' : 283_007,
            'France_Guadeloupe' : 395_700,
            'France_Martinique' : 376_480,
            'France_Mayotte' : 270_372,
            'France_New Caledonia' : 99_926,
            'France_Reunion' : 859_959,
            'France_Saint Barthelemy' : 9_131,
            'France_St Martin' : 32_125,
            'Netherlands_Aruba' : 105_264,
            'Netherlands_Curacao' : 161_014,
            'Netherlands_Sint Maarten' : 41_109,
            'Papua New Guinea' : int(8.251 * 10**6),
            'US_Guam' : 164_229,
            'US_Virgin Islands' : 107_268,
            'United Kingdom_Bermuda' : 65_441,
            'United Kingdom_Cayman Islands' : 61_559,
            'United Kingdom_Channel Islands' : 170_499,
            'United Kingdom_Gibraltar' : 34_571,
            'United Kingdom_Isle of Man' : 84_287,
            'United Kingdom_Montserrat' : 4_922,
            'Botswana' : int(2.292 * 10**6),
            'Burma' : int(53.37 * 10**6),
            'Burundi': int(10.86 * 10**6),
            'Canada' : int(37.59 * 10**6),
            'MS Zaandam' : 1_829,
            'Sierra Leone': int(7.557 * 10**6),
            'United Kingdom' : int(66.65 * 10**6),
            'West Bank and Gaza' : int(4.685 * 10**6),
            'Canada_Northwest Territories': 44_826,
            'Canada_Yukon' : 35_874,
            'United Kingdom_Anguilla' : 15_094,
            'United Kingdom_British Virgin Islands' : 35_802,
            'United Kingdom_Turks and Caicos Islands' : 31_458
           }

tt['Population_final'] = tt['Population_final'].fillna(tt['Place'].map(pop_dict))

tt.loc[tt['Place'] == 'Diamond Princess', 'Population final'] = 2_670

tt['ConfirmedCases_Log'] = tt['ConfirmedCases'].apply(np.log1p)
tt['Fatalities_Log'] = tt['Fatalities'].apply(np.log1p)

tt['Population_final'] = tt['Population_final'].astype('int')
tt['Cases_Per_100kPop'] = (tt['ConfirmedCases'] / tt['Population_final']) * 100000
tt['Fatalities_Per_100kPop'] = (tt['Fatalities'] / tt['Population_final']) * 100000

tt['Cases_Percent_Pop'] = ((tt['ConfirmedCases'] / tt['Population_final']) * 100)
tt['Fatalities_Percent_Pop'] = ((tt['Fatalities'] / tt['Population_final']) * 100)

tt['Cases_Log_Percent_Pop'] = ((tt['ConfirmedCases'] / tt['Population_final']) * 100).apply(np.log1p)
tt['Fatalities_Log_Percent_Pop'] = ((tt['Fatalities'] / tt['Population_final']) * 100).apply(np.log1p)


tt['Max_Confirmed_Cases'] = tt.groupby('Place')['ConfirmedCases'].transform(max)
tt['Max_Fatalities'] = tt.groupby('Place')['Fatalities'].transform(max)

tt['Max_Cases_Per_100kPop'] = tt.groupby('Place')['Cases_Per_100kPop'].transform(max)
tt['Max_Fatalities_Per_100kPop'] = tt.groupby('Place')['Fatalities_Per_100kPop'].transform(max)


# In[4]:


tt.query('Date == @max_date')     .query('Place != "Diamond Princess"')     .query('Cases_Log_Percent_Pop > -10000')     ['Cases_Log_Percent_Pop'].plot(kind='hist', bins=500)
plt.show()


# In[5]:


fig, ax1 = plt.subplots(figsize=(15, 5))

tt.query('Days_Since_Ten_Cases > 0')     .query('Place != "Diamond Princess"')     .dropna(subset=['Cases_Percent_Pop'])     .query('Days_Since_Ten_Cases < 40')     .groupby('Place')     .plot(x='Days_Since_Ten_Cases',
          y='Cases_Log_Percent_Pop',
          style='.-',
          figsize=(15, 5),
          alpha=0.2,
          ax=ax1,
         title='Days since 10 Cases by Percent of Population with Cases')
ax1.get_legend().remove()
plt.show()

fig, ax2 = plt.subplots(figsize=(15, 5))
tt.query('Days_Since_Ten_Fatal > 0')     .query('Place != "Diamond Princess"')     .dropna(subset=['Cases_Percent_Pop'])     .query('Days_Since_Ten_Fatal < 100')     .groupby('Place')     .plot(x='Days_Since_Ten_Fatal',
          y='Cases_Log_Percent_Pop',
          style='.-',
          figsize=(15, 5),
          alpha=0.2,
         title='Days since 10 Fatailites by Percent of Population with Cases',
         ax=ax2)
ax2.get_legend().remove()
plt.show()


# In[6]:


PLOT = False
if PLOT:
    for x in tt['Place'].unique():
        try:
            fig, ax = plt.subplots(1, 4, figsize=(15, 2))
            tt.query('Place == @x')                 .query('ConfirmedCases > 0')                 .set_index('Date')['Cases_Log_Percent_Pop']                 .plot(title=f'{x} confirmed log pct pop', ax=ax[0])
            tt.query('Place == @x')                 .query('ConfirmedCases > 0')                 .set_index('Date')['Cases_Percent_Pop']                 .plot(title=f'{x} confirmed cases', ax=ax[1])
            tt.query('Place == @x')                 .query('Fatalities > 0')                 .set_index('Date')['Fatalities_Log_Percent_Pop']                 .plot(title=f'{x} confirmed log pct pop', ax=ax[2])
            tt.query('Place == @x')                 .query('Fatalities > 0')                 .set_index('Date')['Fatalities_Percent_Pop']                 .plot(title=f'{x} confirmed cases', ax=ax[3])
        except:
            pass
        plt.show()


# In[7]:


tt.query('Date == @max_date')[['Place','Max_Cases_Per_100kPop',
                               'Max_Fatalities_Per_100kPop','Max_Confirmed_Cases',
                               'Population_final',
                              'Days_Since_First_Case',
                              'Confirmed_Cases_Diff']] \
    .drop_duplicates() \
    .sort_values('Max_Cases_Per_100kPop', ascending=False)


# In[8]:


tt['Past_7Days_ConfirmedCases_Std'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200324').groupby('Place')['ConfirmedCases'].std().to_dict())
tt['Past_7Days_Fatalities_Std'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200324').groupby('Place')['Fatalities'].std().to_dict())

tt['Past_7Days_ConfirmedCases_Min'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200324').groupby('Place')['ConfirmedCases'].min().to_dict())
tt['Past_7Days_Fatalities_Min'] = tt['Place'].map(tt.dropna(subset=['Fatalities']).query('Date >= 20200324').groupby('Place')['Fatalities'].min().to_dict())

tt['Past_7Days_ConfirmedCases_Max'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200324').groupby('Place')['ConfirmedCases'].max().to_dict())
tt['Past_7Days_Fatalities_Max'] = tt['Place'].map(tt.dropna(subset=['Fatalities']).query('Date >= 20200324').groupby('Place')['Fatalities'].max().to_dict())

tt['Past_7Days_Confirmed_Change_of_Total'] = (tt['Past_7Days_ConfirmedCases_Max'] - tt['Past_7Days_ConfirmedCases_Min']) / (tt['Past_7Days_ConfirmedCases_Max'])
tt['Past_7Days_Fatalities_Change_of_Total'] = (tt['Past_7Days_Fatalities_Max'] - tt['Past_7Days_Fatalities_Min']) / (tt['Past_7Days_Fatalities_Max'])


# In[9]:


tt['Past_21Days_ConfirmedCases_Std'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200310').groupby('Place')['ConfirmedCases'].std().to_dict())
tt['Past_21Days_Fatalities_Std'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200310').groupby('Place')['Fatalities'].std().to_dict())

tt['Past_21Days_ConfirmedCases_Min'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200310').groupby('Place')['ConfirmedCases'].min().to_dict())
tt['Past_21Days_Fatalities_Min'] = tt['Place'].map(tt.dropna(subset=['Fatalities']).query('Date >= 20200324').groupby('Place')['Fatalities'].min().to_dict())

tt['Past_21Days_ConfirmedCases_Max'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200310').groupby('Place')['ConfirmedCases'].max().to_dict())
tt['Past_21Days_Fatalities_Max'] = tt['Place'].map(tt.dropna(subset=['Fatalities']).query('Date >= 20200310').groupby('Place')['Fatalities'].max().to_dict())

tt['Past_21Days_Confirmed_Change_of_Total'] = (tt['Past_21Days_ConfirmedCases_Max'] - tt['Past_21Days_ConfirmedCases_Min']) / (tt['Past_21Days_ConfirmedCases_Max'])
tt['Past_21Days_Fatalities_Change_of_Total'] = (tt['Past_21Days_Fatalities_Max'] - tt['Past_21Days_Fatalities_Min']) / (tt['Past_21Days_Fatalities_Max'])

tt['Past_7Days_Fatalities_Change_of_Total'] = tt['Past_7Days_Fatalities_Change_of_Total'].fillna(0)
tt['Past_21Days_Fatalities_Change_of_Total'] = tt['Past_21Days_Fatalities_Change_of_Total'].fillna(0)


# In[10]:


tt['Date_7Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 7]     .set_index('Place')['Date']     .to_dict())
tt['Date_14Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 14]     .set_index('Place')['Date']     .to_dict())
tt['Date_21Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 21]     .set_index('Place')['Date']     .to_dict())
tt['Date_28Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 28]     .set_index('Place')['Date']     .to_dict())
tt['Date_35Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 35]     .set_index('Place')['Date']     .to_dict())
tt['Date_60Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 60]     .set_index('Place')['Date']     .to_dict())

tt['Date_7Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 7]     .set_index('Place')['Date']     .to_dict())
tt['Date_14Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 14]     .set_index('Place')['Date']     .to_dict())
tt['Date_21Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 21]     .set_index('Place')['Date']     .to_dict())
tt['Date_28Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 28]     .set_index('Place')['Date']     .to_dict())
tt['Date_35Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 35]     .set_index('Place')['Date']     .to_dict())
tt['Date_60Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 60]     .set_index('Place')['Date']     .to_dict())


tt['Date_7Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 7]     .set_index('Place')['Date']     .to_dict())
tt['Date_14Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 14]     .set_index('Place')['Date']     .to_dict())
tt['Date_21Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 21]     .set_index('Place')['Date']     .to_dict())
tt['Date_28Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 28]     .set_index('Place')['Date']     .to_dict())
tt['Date_35Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 35]     .set_index('Place')['Date']     .to_dict())
tt['Date_60Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 60]     .set_index('Place')['Date']     .to_dict())


# In[11]:


tt['CC_7D_1stCase'] = tt.loc[tt['Date_7Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_14D_1stCase'] = tt.loc[tt['Date_14Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_21D_1stCase'] = tt.loc[tt['Date_21Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_28D_1stCase'] = tt.loc[tt['Date_28Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_35D_1stCase'] = tt.loc[tt['Date_35Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_60D_1stCase'] = tt.loc[tt['Date_60Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']

tt['F_7D_1stCase'] = tt.loc[tt['Date_7Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_14D_1stCase'] = tt.loc[tt['Date_14Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_21D_1stCase'] = tt.loc[tt['Date_21Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_28D_1stCase'] = tt.loc[tt['Date_28Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_35D_1stCase'] = tt.loc[tt['Date_35Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_60D_1stCase'] = tt.loc[tt['Date_60Days_Since_First_Case'] == tt['Date']]['Fatalities']

tt['CC_7D_10Case'] = tt.loc[tt['Date_7Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_14D_10Case'] = tt.loc[tt['Date_14Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_21D_10Case'] = tt.loc[tt['Date_21Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_28D_10Case'] = tt.loc[tt['Date_28Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_35D_10Case'] = tt.loc[tt['Date_35Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_60D_10Case'] = tt.loc[tt['Date_60Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']

tt['F_7D_10Case'] = tt.loc[tt['Date_7Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_14D_10Case'] = tt.loc[tt['Date_14Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_21D_10Case'] = tt.loc[tt['Date_21Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_28D_10Case'] = tt.loc[tt['Date_28Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_35D_10Case'] = tt.loc[tt['Date_35Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_60D_10Case'] = tt.loc[tt['Date_60Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']

tt['CC_7D_1Fatal'] = tt.loc[tt['Date_7Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_14D_1Fatal'] = tt.loc[tt['Date_14Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_21D_1Fatal'] = tt.loc[tt['Date_21Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_28D_1Fatal'] = tt.loc[tt['Date_28Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_35D_1Fatal'] = tt.loc[tt['Date_35Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_60D_1Fatal'] = tt.loc[tt['Date_60Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']

tt['F_7D_1Fatal'] = tt.loc[tt['Date_7Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_14D_1Fatal'] = tt.loc[tt['Date_14Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_21D_1Fatal'] = tt.loc[tt['Date_21Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_28D_1Fatal'] = tt.loc[tt['Date_28Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_35D_1Fatal'] = tt.loc[tt['Date_35Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_60D_1Fatal'] = tt.loc[tt['Date_60Days_Since_First_Fatal'] == tt['Date']]['Fatalities']


# In[12]:


fig, axs = plt.subplots(1, 2, figsize=(15, 5))
tt[['Place','Past_7Days_Confirmed_Change_of_Total','Past_7Days_Fatalities_Change_of_Total',
    'Past_7Days_ConfirmedCases_Max','Past_7Days_ConfirmedCases_Min',
   'Past_7Days_Fatalities_Max','Past_7Days_Fatalities_Min']] \
    .drop_duplicates() \
    .sort_values('Past_7Days_Confirmed_Change_of_Total')['Past_7Days_Confirmed_Change_of_Total'] \
    .plot(kind='hist', bins=50, title='Distribution of Pct change confirmed past 7 days', ax=axs[0])
tt[['Place','Past_21Days_Confirmed_Change_of_Total','Past_21Days_Fatalities_Change_of_Total',
    'Past_21Days_ConfirmedCases_Max','Past_21Days_ConfirmedCases_Min',
   'Past_21Days_Fatalities_Max','Past_21Days_Fatalities_Min']] \
    .drop_duplicates() \
    .sort_values('Past_21Days_Confirmed_Change_of_Total')['Past_21Days_Confirmed_Change_of_Total'] \
    .plot(kind='hist', bins=50, title='Distribution of Pct change confirmed past 21 days', ax=axs[1])
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
tt[['Place','Past_7Days_Fatalities_Change_of_Total']]     .drop_duplicates()['Past_7Days_Fatalities_Change_of_Total']     .plot(kind='hist', bins=50, title='Distribution of Pct change confirmed past 7 days', ax=axs[0])
tt[['Place', 'Past_21Days_Fatalities_Change_of_Total']]     .drop_duplicates()['Past_21Days_Fatalities_Change_of_Total']     .plot(kind='hist', bins=50, title='Distribution of Pct change confirmed past 21 days', ax=axs[1])
plt.show()


# In[13]:


tt.head()


# In[14]:


# Example of flat prop
tt.query("Place == 'China_Chongqing'").set_index('Date')['ConfirmedCases'].dropna().plot(figsize=(15, 5))
plt.show()


# In[15]:


# Assume the places with small rate of change will continue slow down of virus spread
constant_case_places = tt.loc[(tt['Past_21Days_Confirmed_Change_of_Total'] < 0.01) & (tt['ConfirmedCases'] > 10)]['Place'].unique()
constant_case_places


# In[16]:


# Assume the places with small rate of change will continue slow down of virus spread
constant_fatal_places = tt.loc[(tt['Past_21Days_Fatalities_Change_of_Total'] < 0.01) & (tt['Fatalities'] > 1)]['Place'].unique()
constant_fatal_places


# In[17]:


tt.query("Place == 'Italy'").set_index('Date')[['ConfirmedCases']]     .dropna().plot(figsize=(15, 5), title='Italy Confirmed Cases')
plt.show()
tt.query("Place == 'Italy'").set_index('Date')[['ConfirmedCases_Log']]     .dropna().plot(figsize=(15, 5), title='Italy Fatalities')
plt.show()


# In[18]:


latest_summary_stats = tt.query('Date == @max_date')     [['Country_Region',
      'Place',
      'Max_Cases_Per_100kPop',
      'Max_Fatalities_Per_100kPop',
      'Max_Confirmed_Cases',
      'Population_final',
      'Days_Since_First_Case',
      'Days_Since_Ten_Cases']] \
    .drop_duplicates()


# In[19]:


tt.query('Province_State == "Maryland"').set_index('Date')     [['ConfirmedCases','Confirmed_Cases_Diff']].plot(figsize=(15,5 ))


# In[20]:


tt['ConfirmedCasesRolling2'] = tt.groupby('Place')['ConfirmedCases'].rolling(2, center=True).mean().values
tt['FatalitiesRolling2'] = tt.groupby('Place')['Fatalities'].rolling(2, center=True).mean().values
train = tt.loc[~tt['ConfirmedCases'].isna()].query('Days_Since_First_Case > 0')

TARGET = 'ConfirmedCasesRolling2'


# In[21]:


# LightGBM is no bueno

# import lightgbm as lgb

# SEED = 529
# params = {'num_leaves': 8,
#           'min_data_in_leaf': 5,  # 42,
#           'objective': 'regression',
#           'max_depth': 2,
#           'learning_rate': 0.02,
# #           'boosting': 'gbdt',
#           'bagging_freq': 5,  # 5
#           'bagging_fraction': 0.8,  # 0.5,
#           'feature_fraction': 0.82,
#           'bagging_seed': SEED,
#           'reg_alpha': 1,  # 1.728910519108444,
#           'reg_lambda': 4.98,
#           'random_state': SEED,
#           'metric': 'mse',
#           'verbosity': 100,
#           'min_gain_to_split': 0.02,  # 0.01077313523861969,
#           'min_child_weight': 5,  # 19.428902804238373,
#           'num_threads': 6,
#           }

# model = lgb.LGBMRegressor(**params, n_estimators=5000)
# model.fit(train[FEATURES],
#           train[TARGET])


# In[22]:


# model.feature_importances_


# In[23]:


tt['Date'].min()


# In[24]:


tt['doy'] = tt['Date'].dt.dayofyear


# In[ ]:





# In[25]:


# test = tt.loc[~tt['ForecastId'].isna()]
# preds = model.predict(test[FEATURES])
# tt.loc[~tt['ForecastId'].isna(),
#        'Confirmed_Cases_Diff_Pred'] = preds
# # tt['ConfirmedCases_Pred'] = tt['ConfirmedCases_Log_Pred'].apply(np.expm1)


# In[26]:


import scipy.optimize as opt

def sigmoid(t, M, beta, alpha):
    return M / (1 + np.exp(-beta * (t - alpha)))


for myplace in tt['Place'].unique():


    pop = tt.loc[tt['Place'] == myplace]['Population_final'].values[0]

    BOUNDS=(0, [pop, 2.0, 100])

    xin = tt.query('Place == @myplace').dropna(subset=['ConfirmedCases'])['doy'].values
    yin = tt.query('Place == @myplace').dropna(subset=['ConfirmedCases'])['ConfirmedCases'].values
    popt, pcov = opt.curve_fit(sigmoid,
                               xin,
                               yin,
                               bounds=BOUNDS)

    M, beta, alpha = popt
    print(M, beta, alpha)
    x = tt.loc[tt['Place'] == myplace]['doy'].values
    tt.loc[tt['Place'] == myplace, 'ConfirmedCases_forecast'] = sigmoid(x, M, beta, alpha)

    xin = tt.query('Place == @myplace').dropna(subset=['Fatalities'])['doy'].values
    yin = tt.query('Place == @myplace').dropna(subset=['Fatalities'])['Fatalities'].values
    popt, pcov = opt.curve_fit(sigmoid,
                               xin,
                               yin,
                               bounds=BOUNDS)

    M, beta, alpha = popt
    print(M, beta, alpha)
    x = tt.loc[tt['Place'] == myplace]['doy'].values
    tt.loc[tt['Place'] == myplace, 'Fatalities_forecast'] = sigmoid(x, M, beta, alpha)

    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_forecast']].plot(title=myplace, ax=axs[0])
    ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_forecast']].plot(title=myplace, ax=axs[1])
    plt.show()


# In[27]:


from sklearn.linear_model import LinearRegression, ElasticNet

for myplace in tt['Place'].unique():
    try:
        # Confirmed Cases
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        dat = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','ConfirmedCases_Log']].dropna()
        X = dat['Days_Since_Ten_Cases']
        y = dat['ConfirmedCases_Log']
        y = y.cummax()
        dat_all = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','ConfirmedCases_Log']]
        X_pred = dat_all['Days_Since_Ten_Cases']
        en = ElasticNet()
        en.fit(X.values.reshape(-1, 1), y.values)
        preds = en.predict(X_pred.values.reshape(-1, 1))
        tt.loc[(tt['Place'] == myplace) & (tt['Days_Since_Ten_Cases'] >= 0), 'ConfirmedCases_Log_Pred1'] = preds
        tt.loc[(tt['Place'] == myplace), 'ConfirmedCases_Pred1'] = tt['ConfirmedCases_Log_Pred1'].apply(np.expm1)
        # Cap at 10 % Population
        pop_myplace = tt.query('Place == @myplace')['Population_final'].values[0]
        tt.loc[(tt['Place'] == myplace) & (tt['ConfirmedCases_Pred1'] > (0.05 * pop_myplace)), 'ConfirmedCases_Pred1'] = (0.05 * pop_myplace)
        ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[0])
        # Fatalities
        # If low count then do percent of confirmed:
        dat = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','Fatalities_Log']].dropna()
        if len(dat) < 5:
            tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt.loc[(tt['Place'] == myplace)]['ConfirmedCases_Pred1'] * 0.0001
        elif tt.query('Place == @myplace')['Fatalities'].max() < 5:
            tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt.loc[(tt['Place'] == myplace)]['ConfirmedCases_Pred1'] * 0.0001
        else:
            X = dat['Days_Since_Ten_Cases']
            y = dat['Fatalities_Log']
            y = y.cummax()
            dat_all = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','Fatalities_Log']]
            X_pred = dat_all['Days_Since_Ten_Cases']
            en = ElasticNet()
            en.fit(X.values.reshape(-1, 1), y.values)
            preds = en.predict(X_pred.values.reshape(-1, 1))
            tt.loc[(tt['Place'] == myplace) & (tt['Days_Since_Ten_Cases'] >= 0), 'Fatalities_Log_Pred1'] = preds
            tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt['Fatalities_Log_Pred1'].apply(np.expm1)

            # Cap at 0.0001 Population
            pop_myplace = tt.query('Place == @myplace')['Population_final'].values[0]
            tt.loc[(tt['Place'] == myplace) & (tt['Fatalities_Pred1'] > (0.0001 * pop_myplace)), 'Fatalities_Pred1'] = (0.0001 * pop_myplace)

        ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[1])
        plt.show()
    except:
        print(f'============= FAILED FOR {myplace} =============')


# In[28]:


constant_fatal_places


# In[29]:


tt.loc[tt['Place'].isin(constant_fatal_places), 'ConfirmedCases_Pred1'] = tt.loc[tt['Place'].isin(constant_fatal_places)]['Place'].map(tt.loc[tt['Place'].isin(constant_fatal_places)].groupby('Place')['ConfirmedCases'].max())
tt.loc[tt['Place'].isin(constant_fatal_places), 'Fatalities_Pred1'] = tt.loc[tt['Place'].isin(constant_fatal_places)]['Place'].map(tt.loc[tt['Place'].isin(constant_fatal_places)].groupby('Place')['Fatalities'].max())


# In[30]:


for myplace in constant_fatal_places:
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[0])
    ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[1])
    plt.show()


# In[31]:


tt['ConfirmedCases_Pred'] = tt[['ConfirmedCases_Pred1','ConfirmedCases_forecast']].mean(axis=1)
tt['Fatalities_Pred'] = tt[['Fatalities_Pred1','Fatalities_forecast']].mean(axis=1)


# In[32]:


# Estimated total
tt.groupby('Place')['Fatalities_Pred'].max().sum()


# In[33]:


# Clean Up any time the actual is less than the real
tt['ConfirmedCases_Pred'] = tt[['ConfirmedCases','ConfirmedCases_Pred']].max(axis=1)
tt['Fatalities_Pred'] = tt[['Fatalities','Fatalities_Pred']].max(axis=1)

tt['ConfirmedCases_Pred'] = tt['ConfirmedCases_Pred'].fillna(0)
tt['Fatalities_Pred'] = tt['Fatalities_Pred'].fillna(0)

# Fill pred with
tt.loc[~tt['ConfirmedCases'].isna(), 'ConfirmedCases_Pred1'] = tt.loc[~tt['ConfirmedCases'].isna()]['ConfirmedCases']
tt.loc[~tt['Fatalities'].isna(), 'Fatalities_Pred1'] = tt.loc[~tt['Fatalities'].isna()]['Fatalities']

tt['ConfirmedCases_Pred'] = tt.groupby('Place')['ConfirmedCases_Pred'].transform('cummax')
tt['Fatalities_Pred'] = tt.groupby('Place')['Fatalities_Pred'].transform('cummax')


# In[34]:


for myplace in tt['Place'].unique():
    try:
        # Confirmed Cases
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_Pred']].plot(title=myplace, ax=axs[0])
        ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_Pred']].plot(title=myplace, ax=axs[1])
        plt.show()
    except:
        print(f'============= FAILED FOR {myplace} =============')


# In[35]:


tt.groupby('Place')['Fatalities_Pred'].max().sort_values()


# In[36]:


ss = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')


# In[37]:


print(ss.shape)
ss.head()


# In[38]:


mysub = tt.dropna(subset=['ForecastId'])[['ForecastId','ConfirmedCases_Pred1','Fatalities_Pred1']]
mysub['ForecastId'] = mysub['ForecastId'].astype('int')
mysub = mysub.rename(columns={'ConfirmedCases_Pred1':'ConfirmedCases',
                      'Fatalities_Pred1': 'Fatalities'})
mysub.to_csv('submission.csv', index=False)


# In[39]:


tt.groupby('Date').sum()[['ConfirmedCases',
                          'ConfirmedCases_Pred',]].plot(figsize=(15, 5))
tt.groupby('Date').sum()[['Fatalities',
                          'Fatalities_Pred']].plot(figsize=(15, 5))


# In[ ]:




