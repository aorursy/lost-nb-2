#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from scipy import integrate, optimize
import warnings
warnings.filterwarnings('ignore')


# In[2]:


class InputParam(object):
    """param for config."""

    def __init__(self, date, confirmed, fatality, country, province, dead_rate):
        self.date = date
        self.confirmed = confirmed
        self.fatality = fatality
        self.country = country
        self.province = province
        self.dead_rate = dead_rate
        self.data_df = None


# In[3]:


def load_virus_report_source():
    # const
    input_param = InputParam(
        date = 'Date',
        confirmed = 'Confirmed',
        fatality = 'Deaths',
        country = 'Country/Region',
        province = 'Province/State',
        dead_rate = 'Deaths_per_Confirmed_%')

    train = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
    train = train.fillna('_')

    train[input_param.date] = train[input_param.date].apply(lambda x: datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d'))
    display(train.head(10))
    display(train.describe())
    print("""
    Number of Country_Region: {country_region}
    from {start_day} to {end_day}
    """.format(
        country_region=train[input_param.country].nunique(),
        start_day=min(train[input_param.date]),
        end_day=max(train[input_param.date])))

    input_param.data_df = train
    return input_param


# In[4]:


def _convert_dt_col_to_row(input_df, col, country_key_name, province_key_name, date_key_name):
    province_list = input_df[province_key_name].tolist()
    country_list = input_df[country_key_name].tolist()

    arr = input_df.columns.tolist()[4:]
    dt_list, pl, cl, key_list = [], [], [], []
    for dt in arr:
        key_list += input_df[dt].tolist()
        dt_list += [datetime.strptime(dt, '%m/%d/%y').strftime('%Y-%m-%d')] * len(province_list)
        pl += province_list
        cl += country_list

    m = {
        province_key_name: pl,
        country_key_name: cl,
        date_key_name: dt_list,
        col: key_list,
    }

    df1 = pd.DataFrame(m, columns = [province_key_name, country_key_name, date_key_name, col])    .set_index([province_key_name, country_key_name, date_key_name])

    return df1


def load_cssegi_source():
    # const
    input_param = InputParam(
        date = 'Date',
        confirmed = 'Confirmed',
        fatality = 'Deaths',
        country = 'Country/Region',
        province = 'Province/State',
        dead_rate = 'Deaths_per_Confirmed_%')

    df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

    df_confirmed = df_confirmed.fillna('_')
    df_deaths = df_deaths.fillna('_')
    
    df_c1 = _convert_dt_col_to_row(df_confirmed, input_param.confirmed, input_param.country, input_param.province, input_param.date)
    df_d1 = _convert_dt_col_to_row(df_deaths, input_param.fatality, input_param.country, input_param.province, input_param.date)

    display(df_c1.head(10))
    display(df_d1.head(10))
    
    train = df_c1.join(df_d1).reset_index()

    display(train.head(10))
    display(train.describe())
    print("""
    Number of Country_Region: {country_region}
    from {start_day} to {end_day}
    """.format(
        country_region=train[input_param.country].nunique(),
        start_day=min(train[input_param.date]),
        end_day=max(train[input_param.date])))
    
    input_param.data_df = train
    return input_param


# In[5]:


def load_forecasting_week_source():
    input_param = InputParam(
        date = 'Date',
        confirmed = 'ConfirmedCases',
        fatality = 'Fatalities',
        country = 'Country_Region',
        province = 'Province_State',
        dead_rate = 'Fatality_per_Confirmed_%')

    train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

    train.fillna('_')
    display(train.head(5))
    display(train.describe())
    print("""
    Number of Country_Region: {country_region}
    from {start_day} to {end_day}
    """.format(
        country_region=train[input_param.country].nunique(),
        start_day=min(train[input_param.date]),
        end_day=max(train[input_param.date]))
    )

    input_param.data_df = train

    return input_param


# In[6]:


# pick up one of below
#param = load_virus_report_source()
#param = load_forecasting_week_source()
param = load_cssegi_source()

# const
train = param.data_df
DATE = param.date
CONFIRMED = param.confirmed
FATALITY = param.fatality
COUNTRY = param.country
PROVINCE = param.province
DEAD_RATE = param.dead_rate


# In[7]:


def region_summary(input_df, key_list):
    """key_list is like ['Country_Region']"""
    latest_date = max(input_df[DATE].tolist())
    df1 = input_df[input_df[DATE] == latest_date]
    df1 = df1.groupby(key_list).agg({CONFIRMED: ['sum'], FATALITY: ['sum']})

    # merge to single level index
    df1.columns = df1.columns.get_level_values(0)

    df1[DEAD_RATE] = df1[FATALITY] / df1[CONFIRMED] * 100
    return df1.reset_index().sort_values(by=[CONFIRMED], ascending=False)


# In[8]:


df_country = region_summary(train, [COUNTRY])
df_province = region_summary(train, [COUNTRY, PROVINCE])

top_countries = df_country[COUNTRY].unique()
top_province_pairs = df_province[[COUNTRY, PROVINCE]].to_records().tolist()

# print(top_countries)
# print(top_province_pairs)

display(df_country.head(20))
display(df_province.head(50))


# In[9]:


def plot_one_region(input_df, input_country, input_province=None):
    if input_province is not None:
        df1 = input_df[(input_df[COUNTRY]==input_country) & (input_df[PROVINCE]==input_province)]
        title = '{}, {}'.format(input_country, input_province)
    else:
        df1 = input_df[input_df[COUNTRY]==input_country]
        title = input_country

    df_confirmed = df1.groupby(DATE).agg({CONFIRMED: ['sum']})
    df_fatalities = df1.groupby(DATE).agg({FATALITY: ['sum']})
    df_join = df_confirmed.join(df_fatalities)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,7))
    plot_size = 13
    df_join.plot(ax=ax1, color=['blue', 'orange'])
    ax1.set_title('{} -- confirmed cases'.format(title), size=plot_size)
    ax1.set_ylabel('Number of cases', size=plot_size)
    ax1.set_xlabel('Date', size=plot_size)
    df_fatalities.plot(ax=ax2, color='orange')
    ax2.set_title('{} -- death cases'.format(title), size=plot_size)
    ax2.set_ylabel('Number of cases', size=plot_size)
    ax2.set_xlabel('Date', size=plot_size)


# In[10]:


for country in top_countries[:20]:
    plot_one_region(train, country)


# In[11]:


for (idx_unused, country, province) in top_province_pairs[:30]:
    plot_one_region(train, country, input_province=province)

