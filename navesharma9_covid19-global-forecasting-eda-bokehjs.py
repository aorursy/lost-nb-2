import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import reverse_geocoder as rg
import matplotlib.animation as animation
import datetime
import pycountry
import calendar
import geopandas as gp

#install lib
    #1. pip install reverse_geocoder

pip install reverse_geocoder

import warnings
warnings.filterwarnings('ignore')

master_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

master_df.head()

sns.heatmap(master_df.isnull())

master_df.info()

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'bold',
        'size': 12,
        }

def fillNullProvince(x):
  coordinates = (x['Lat'],x['Long'])
  result =  rg.search(coordinates)
  return result[0].get('name')

master_df['Province/State'] = master_df.apply(lambda x:fillNullProvince(x) if pd.isnull(x['Province/State']) else x['Province/State'] ,axis=1)
master_df['Province/State'].value_counts()

master_df.head()

#master_df = train_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/covid19/covid19master.csv')

master_df['month'] = pd.DatetimeIndex(master_df['Date']).month 
master_df['month'] = master_df['month'].apply(lambda x: calendar.month_abbr[x])

master_df.head()

confirmed_cases_by_country = master_df.groupby('Country/Region').max()[['ConfirmedCases','Fatalities']]
confirmed_cases_by_country.sort_values(by=['ConfirmedCases','Fatalities'],ascending=False,inplace=True)

confirmed_cases_by_country.head(10)

confirmed_cases_by_country['Country'] = confirmed_cases_by_country.index

plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
confirmed_cases_by_country['ConfirmedCases'].head(10).plot(kind='barh',color=(0,0.9,.25,1.0))
plt.xticks(rotation=90)
xlocs, xlabs = plt.xticks()
xlocs=[i+1 for i in range(0,10)]
xlabs=[i/2 for i in range(0,10)]
for i, v in enumerate(confirmed_cases_by_country['ConfirmedCases'].head(10)):
  plt.text(v, xlocs[i]-0.9 , str(v))
plt.xlabel('total number of cases (Normalized)')
plt.title('Top 10 most infected countries')

plt.subplot(1,2,2)
confirmed_cases_by_country['Fatalities'].head(10).plot(kind='barh',color = (0.9,0.2,0.2,1.0))
for i, v in enumerate(confirmed_cases_by_country['Fatalities'].head(10)):
  plt.text(v, xlocs[i]-0.9 , str(v))
plt.xlabel('total number of cases')
plt.title('Top 10 most fatalities countries')
plt.xticks(rotation=90)
plt.show


from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, ColumnDataSource
from bokeh.models import HoverTool
import json
#Input GeoJSON source that contains features for plotting.
merged_json = json.loads(master_df.to_json())
json_data = json.dumps(merged_json)

geosource = GeoJSONDataSource(geojson = json_data)
#Create figure object.
p = figure(title = 'Worldwide spread of Coronavirus', plot_height = 600 , plot_width = 1050)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
#Add patch renderer to figure. 
patch=p.patches(xs='xs',ys='ys', source = geosource,fill_color = '#fff7bc',
          line_color = 'black', line_width = 0.35, fill_alpha = 1, 
                hover_fill_color="#fec44f")
p.add_tools(HoverTool(tooltips=[('Country','@country'),('ConfirmedCases','@confirmedcases'), ('Fatalities','@fatalities')], renderers=[patch]))

#Display figure inline in Jupyter Notebook.
output_notebook()
#Display figure.
show(p)

def getAlph(input):
  countries={}
  for country in pycountry.countries:
    countries[country.name] = country.alpha_3
    codes = countries.get(input, 'Unknown code')
  return codes

confirmed_cases_by_country['iso_alpha'] = confirmed_cases_by_country['Country'].apply(lambda x:getAlph(x))

confirmed_cases_by_country['TotalConfirmedCases'] = confirmed_cases_by_country['ConfirmedCases'].pow(0.3) * 3.5

confirmed_cases_by_country.head()

import plotly.express as px
#df = px.data.
#month = confirmed_cases_by_country['']
fig = px.scatter_geo(confirmed_cases_by_country, locations="iso_alpha",color="ConfirmedCases",
                     text='Fatalities', size="TotalConfirmedCases",
                     projection="natural earth")
fig.update_layout(
    title={
        'text': "Virus spred all over the world",
        'y':1,
        'x':0.4,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

master_df.head()

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=master_df.Date, y=master_df['ConfirmedCases'], name="ConfirmedCases",
                         line_color='orange'))

fig.add_trace(go.Scatter(x=master_df.Date, y=master_df['Fatalities'], name="Deaths",
                        line_color='red'))

fig.update_layout(title_text='Covid-19 life lose over time',xaxis_rangeslider_visible=True)
fig.show()

model_df = master_df[['Country/Region','ConfirmedCases','Fatalities','Date']]

model_df['month'] = pd.DatetimeIndex(master_df['Date']).month 
model_df['year'] = pd.DatetimeIndex(master_df['Date']).year

model_df.head()

model_df['PositiveCases'] = model_df['ConfirmedCases'].pow(0.3) * 3.5 

model_df['Death'] = model_df['Fatalities'].pow(0.3) * 3.5 

model_df[['PositiveCases','Fatalities']].plot(figsize=(15,6))
plt.title('Comparision of Confirmed and Fatalities Cases',fontdict=font)

temp_df = model_df[['PositiveCases','Fatalities','Country/Region','ConfirmedCases']]

temp_df.index = model_df['Date']


temp_df[['ConfirmedCases']].plot(figsize=(15,3),color='yellow')
temp_df[['Fatalities']].plot(figsize=(15,3),color='red')

china_df = temp_df[temp_df['Country/Region']=='China']['ConfirmedCases'].sort_values()
italy_df = temp_df[temp_df['Country/Region']=='Italy']['ConfirmedCases']
spain_df = temp_df[temp_df['Country/Region']=='Spain']['ConfirmedCases']
india_df    = temp_df[temp_df['Country/Region']=='India']['ConfirmedCases']

china_df.index = temp_df[temp_df['Country/Region']=='China'].index
italy_df.index = temp_df[temp_df['Country/Region']=='Italy'].index
spain_df.index = temp_df[temp_df['Country/Region']=='Spain'].index
india_df.index = temp_df[temp_df['Country/Region']=='India'].index

#I could created function

plt.figure(figsize=(13,5))
plt.subplot(1,2,1)

italy_df.plot(label='Italy')
spain_df.plot(label='Spain')
india_df.plot(label='India')
plt.xticks(rotation=20)
plt.title('Week wise positive cases on contries')
plt.legend()
plt.subplot(1,2,2)
china_df.plot(label='China')
plt.xticks(rotation=20)
plt.show()

master_df[master_df['Country/Region']=='Italy']['ConfirmedCases'].max()


