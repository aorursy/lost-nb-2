#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q calmap')


# In[2]:


get_ipython().system('pip install -q pycountry_convert')


# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
from pathlib import Path
from datetime import datetime
import calmap

import os
import glob
import copy

import folium 

import geopandas
import pycountry_convert as pc
from google.cloud import bigquery
from folium import plugins
from folium import Marker,GeoJson,Choropleth, Circle
from folium.plugins import HeatMap
from folium.plugins import HeatMap, MarkerCluster
from scipy.spatial.distance import cdist


# In[5]:


#pd.set_option('max_columns', 100)
#pd.set_option('max_rows', 500)
import warnings
warnings.filterwarnings('ignore')


# In[6]:


# Code for displaying plotly express plot
def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))


# In[7]:


from plotly import tools
import plotly.offline as py

py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

from plotly.subplots import make_subplots
configure_plotly_browser_state()
from IPython.display import IFrame
from IPython.display import Javascript
from IPython.core.display import display
from IPython.core.display import HTML

from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"


# In[8]:


IFrame('https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6', width='90%', height=600)


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[10]:


# Building and fitting Random Forest
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


# In[11]:


## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout


# In[12]:


# Retriving Dataset
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')


casescountry_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
casestime_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv",parse_dates=['Last_Update'])


# In[13]:


#Country
country_df=pd.read_csv("../input/countryinfo/covid19countryinfo.csv")
covidtests_df = pd.read_csv("../input/countryinfo/covid19tests.csv")


# In[14]:


confirmed_df.rename(columns={'Country/Region':'Country'}, inplace=True)
deaths_df.rename(columns={'Country/Region':'Country'}, inplace=True)
recovered_df.rename(columns={'Country/Region':'Country'}, inplace=True)
casescountry_df.rename(columns={'Country_Region':'Country'}, inplace=True)
casestime_df.rename(columns={'Country_Region':'Country'}, inplace=True)


# In[15]:


# Changing the conuntry names as required by pycountry_convert Lib


confirmed_df.loc[confirmed_df["Country"] == "US", "Country"] = "USA"

confirmed_df.loc[confirmed_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
confirmed_df.loc[confirmed_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
confirmed_df.loc[confirmed_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
confirmed_df.loc[confirmed_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
confirmed_df.loc[confirmed_df['Country'] == "Reunion", "Country"] = "Réunion"
confirmed_df.loc[confirmed_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'
confirmed_df.loc[confirmed_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
confirmed_df.loc[confirmed_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'



deaths_df.loc[deaths_df['Country'] == "US", "Country"] = "USA"
deaths_df.loc[deaths_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
deaths_df.loc[deaths_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
deaths_df.loc[deaths_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
deaths_df.loc[deaths_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
deaths_df.loc[deaths_df['Country'] == "Reunion", "Country"] = "Réunion"
deaths_df.loc[deaths_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'
deaths_df.loc[deaths_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
deaths_df.loc[deaths_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'



recovered_df.loc[recovered_df['Country'] == "US", "Country"] = "USA"
recovered_df.loc[recovered_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
recovered_df.loc[recovered_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
recovered_df.loc[recovered_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
recovered_df.loc[recovered_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
recovered_df.loc[recovered_df['Country'] == "Reunion", "Country"] = "Réunion"
recovered_df.loc[recovered_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'
recovered_df.loc[recovered_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
recovered_df.loc[recovered_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'


casescountry_df.loc[casescountry_df['Country'] == "US", "Country"] = "USA"
casescountry_df.loc[casescountry_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
casescountry_df.loc[casescountry_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
casescountry_df.loc[casescountry_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
casescountry_df.loc[casescountry_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
casescountry_df.loc[casescountry_df['Country'] == "Reunion", "Country"] = "Réunion"
casescountry_df.loc[casescountry_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'
casescountry_df.loc[casescountry_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
casescountry_df.loc[casescountry_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'



casestime_df.loc[casestime_df['Country'] == "US", "Country"] = "USA"
casestime_df.loc[casestime_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
casestime_df.loc[casestime_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
casestime_df.loc[casestime_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
casestime_df.loc[casestime_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
casestime_df.loc[casestime_df['Country'] == "Reunion", "Country"] = "Réunion"
casestime_df.loc[casestime_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'
casestime_df.loc[casestime_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
casestime_df.loc[casestime_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'


# In[16]:


# getting all countries
countries = np.asarray(confirmed_df["Country"])
countries1 = np.asarray(casescountry_df["Country"])

# Continent_code to Continent_names
continents = {
    'NA': 'North America',
    'SA': 'South America', 
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU' : 'Europe',
    'na' : 'Others'
}

# Defininng Function for getting continent code for country.
def country_to_continent_code(Country):
    try:
        return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(Country))
    except :
        return 'na'

#Collecting Continent Information
confirmed_df.insert(2,"continent", [continents[country_to_continent_code(Country)] for Country in countries[:]])
deaths_df.insert(2,"continent",  [continents[country_to_continent_code(Country)] for Country in countries[:]])
recovered_df.insert(2,"continent",  [continents[country_to_continent_code(Country)] for Country in recovered_df["Country"].values] )   
casescountry_df.insert(1,"continent",  [continents[country_to_continent_code(Country)] for Country in countries1[:]])
casestime_df.insert(1,"continent",  [continents[country_to_continent_code(Country)] for Country in casestime_df["Country"].values])


casescountry_df['Active'] = casescountry_df['Confirmed']-casescountry_df['Deaths']-casescountry_df['Recovered']
casescountry_df["Mortality Rate (per 100)"] = np.round(100*casescountry_df["Deaths"]/casescountry_df["Confirmed"],2)


# In[17]:


casescountry_df.style.background_gradient(cmap='Purples',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='Blues',subset=["Active"])                        .background_gradient(cmap='Oranges',subset=["Mortality Rate (per 100)"])


# In[18]:


countrywise_df = casescountry_df.copy().drop(['Lat','Long_','continent','Last_Update'],axis =1)
countrywise_df.index = countrywise_df["Country"]
countrywise_df = countrywise_df.drop(['Country'],axis=1)

continentwise_df = casescountry_df.copy().drop(['Lat','Long_','Country','Last_Update'],axis =1)
continentwise_df = continentwise_df.groupby(["continent"]).sum()


# In[19]:


continentwise_df["Mortality Rate (per 100)"] = np.round(100*continentwise_df["Deaths"]/continentwise_df["Confirmed"],2)
continentwise_df.sort_values('Mortality Rate (per 100)', ascending= False).style.background_gradient(cmap='Blues',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='Purples',subset=["Active"])                        .background_gradient(cmap='Oranges',subset=["Mortality Rate (per 100)"])


# In[20]:


fig = px.bar(continentwise_df,
            x=continentwise_df.index, y="Confirmed",
            text = continentwise_df.index,
            hover_name=continentwise_df.index,
            hover_data=["Confirmed","Deaths","Recovered","Active","Mortality Rate (per 100)"],
            color_continuous_scale=px.colors.cyclical.IceFire,
            title='COVID-19: Continentwise Details'
)
fig.update_xaxes(title_text="Continent")


# In[21]:


countrywise_df["Mortality Rate (per 100)"] = np.round(100*countrywise_df["Deaths"]/countrywise_df["Confirmed"],0)
countrywise_df.sort_values('Confirmed', ascending= False).style.background_gradient(cmap='Oranges',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='Purples',subset=["Active"])                        .background_gradient(cmap='RdPu',subset=["Mortality Rate (per 100)"])


# In[22]:


f = plt.figure(figsize=(20,10))
f.add_subplot(2,1,1)
calmap.yearplot(casestime_df.groupby('Last_Update')['Confirmed'].sum().diff(), fillcolor='white', cmap='RdPu', linewidth=0.5,linecolor="#fafafa",year=2020,)
plt.title("Daily Confirmed Cases",fontsize=20)
plt.tick_params(labelsize=15)

f.add_subplot(2,1,2)
calmap.yearplot(casestime_df.groupby('Last_Update')['Deaths'].sum().diff(), fillcolor='white', cmap='Oranges', linewidth=1,linecolor="#fafafa",year=2020,)
plt.title("Daily Deaths Cases",fontsize=20)
plt.tick_params(labelsize=15)
plt.show()


# In[23]:


world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2,max_zoom=8,min_zoom=2)
for i in range(0,len(confirmed_df)):
    folium.Circle(
        location=[confirmed_df.iloc[i]['Lat'], confirmed_df.iloc[i]['Long']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+confirmed_df.iloc[i]['Country']+"</h5>"+
                    "<div style='text-align:center;'>"+str(np.nan_to_num(confirmed_df.iloc[i]['Province/State']))+"</div>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
                    "<li>Confirmed: "+str(confirmed_df.iloc[i,-1])+"</li>"+
                    "<li>Deaths:   "+str(deaths_df.iloc[i,-1])+"</li>"+
                    "<li>Mortality Rate:   "+str(np.round(deaths_df.iloc[i,-1]/(confirmed_df.iloc[i,-1]+1.00001)*100,2))+"</li>"+
                    "</ul>",
        radius=(int((np.log(confirmed_df.iloc[i,-1]+1.00001)))+0.2)*50000,
        color='#bb66ff',
        fill_color='#ff8533',
        fill=True).add_to(world_map)

world_map


# In[24]:


# Dictionary to get the state codes from state names for US
us_states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

us_data = casestime_df[casestime_df["Country"]=="USA"]
us_data['Last_Update'] = us_data['Last_Update'].astype(str)
us_data['state_code'] = us_data.apply(lambda x: us_states.get(x.Province_State,float('nan')), axis=1)
#us_data.tail()


# In[25]:


fig = px.choropleth(us_data, 
                    locations="state_code", 
                    locationmode='USA-states',
                    scope='usa',
                    color=np.power(us_data["Confirmed"],0.3) ,  
                    hover_name="Province_State",
                    hover_data=['Confirmed'],
                    #range_color=[1,2000],
                    color_continuous_scale=px.colors.sequential.Emrld,
                    animation_frame = 'Last_Update',
                    title='US with cases',  height=600)
#fig.update(layout_coloraxis_showscale=False)
fig.update_coloraxes(colorbar_title="confirmed",colorscale="tropic")
fig.show()


# In[26]:


europe_data = casescountry_df[casescountry_df["continent"]=="Europe"]
fig = px.choropleth(europe_data, locations="Country", 
                    locationmode='country names', color="Country", 
                    hover_name="Confirmed", range_color=[1,2000], 
                    color_continuous_scale=px.colors.sequential.Inferno, 
                    title='European Countries with Confirmed Cases', scope='europe', height=600)
# fig.update(layout_coloraxis_showscale=False)
fig.update_coloraxes(colorbar_title="Country",colorscale="RdPu")
fig.show()


# In[27]:


df = pd.DataFrame(countrywise_df['Confirmed'])
df = df.reset_index()
fig = px.choropleth(df, locations="Country",
                    color=np.log10(df["Confirmed"]), # lifeExp is a column of gapminder
                    hover_name="Country", # column to add to hover information
                    hover_data=["Confirmed"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Worldwide Confirmed Cases")
fig.update_coloraxes(colorbar_title="Confirmed Cases(Log Scale)",colorscale="RdPu")
# fig.to_image("Global Heat Map confirmed.png")
fig.show()


# In[28]:


df = pd.DataFrame(countrywise_df['Deaths'])
df = df.reset_index()
fig = px.choropleth(df, locations="Country",
                    color=np.log10(df["Deaths"]), 
                    hover_name="Country", # column to add to hover information
                    hover_data=["Deaths"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Worldwide Fatalities")
fig.update_coloraxes(colorbar_title="FATALITIES(Log Scale)",colorscale="viridis")
# fig.to_image("Global Heat Map confirmed.png")
fig.show()


# In[29]:


df = pd.DataFrame(countrywise_df['Recovered'])
df = df.reset_index()
fig = px.choropleth(df, locations="Country",
                    color=np.log10(df["Recovered"]), 
                    hover_name="Country", # column to add to hover information
                    hover_data=["Recovered"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Worldwide Recovered")
fig.update_coloraxes(colorbar_title="Recovered(Log Scale)",colorscale="curl")
fig.show()


# In[30]:


asia_data = casescountry_df[casescountry_df["continent"]=="Asia"]
fig = px.choropleth(asia_data, locations="Country", 
                    locationmode='country names', color="Country", 
                    hover_name="Confirmed", range_color=[1,2000], 
                    color_continuous_scale='Reds', 
                    title='Asian nations with Confirmed Cases', scope='asia', height=600)
fig.update_coloraxes(colorbar_title="Countries Affected",colorscale="sunsetdark")
fig.show()


# In[31]:


temp = casescountry_df.groupby('Last_Update')['Recovered', 'Deaths', 'Confirmed'].sum().reset_index()
temp = temp.melt(id_vars="Last_Update", value_vars=['Recovered', 'Deaths', 'Confirmed'],
                 var_name='case', value_name='count')


#fig = px.line(temp, x="Last_Update", y="count", color='case',
#             title='Cases over time: Line Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])
#fig.show()


fig = px.area(temp, x="Last_Update", y="count", color='case',
             title='Cases over time: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])
fig.show()


# In[32]:


#px.set_mapbox_access_token(open(".mapbox_token").read())

#px.scatter_mapbox(casescountry_df, lat="Lat", lon="Long_",     color=np.power(casescountry_df["Confirmed"],0.3)-2 , 
#                        size= np.power(casescountry_df["Confirmed"]+1,0.3)-1,
#                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
#fig.show()

africa_data = casescountry_df[casescountry_df["continent"]=="Africa"]
fig = px.choropleth(africa_data, locations="Country", 
                    locationmode='country names', color="Country", 
                    hover_name="Confirmed", range_color=[1,12], 
                    color_continuous_scale='Oranges', 
                    title='African Countries with Confirmed Cases', scope='africa', height=600)
fig.show()


# In[33]:


df = casescountry_df
df["world"] = "world" # in order to have a single root node
fig = px.treemap(df, path=['world', 'continent', 'Country'], values='Confirmed',
                  color='Country', hover_data=['Deaths'],
                  title=' COVID-19 Affected Countries ',
                  color_continuous_scale=px.colors.diverging.Tealrose,
                  color_continuous_midpoint=np.average(df['Active'], weights=df['Confirmed']))
fig.show()


# In[34]:


import math
map_ = folium.Map(location=[54,15], tiles='cartodbpositron', zoom_start=2)

# Add points to the map
mc = MarkerCluster()
for idx, row in casescountry_df.iterrows():
    if not math.isnan(row['Long_']) and not math.isnan(row['Lat']):
        mc.add_child(Marker([row['Lat'], row['Long_']]))
map_.add_child(mc)

# Display the map
#map_


# In[35]:


continentwise_df_tmp = continentwise_df
continentwise_df_tmp["continent"] = continentwise_df_tmp.index
px.area(continentwise_df_tmp, x="Confirmed", y="Deaths", color="continent", line_group="continent")


# In[36]:


df_data = casestime_df.groupby(['Last_Update', 'Country'])['Confirmed', 'Deaths'].max().reset_index()
df_data["Last_Update"] = pd.to_datetime( df_data["Last_Update"]).dt.strftime('%m/%d/%Y')

px.scatter_geo(df_data, locations="Country", locationmode='country names', 
                     color=np.power(df_data["Confirmed"],0.3)-2 , size= np.power(df_data["Confirmed"]+1,0.3)-1, hover_name="Country",
                     hover_data=["Confirmed"],
                     range_color= [0, max(np.power(df_data["Confirmed"],0.3))], 
                     projection="natural earth", animation_frame="Last_Update", 
                     color_continuous_scale=px.colors.cyclical.IceFire,
                     title='COVID-19: Progression of spread'
                    )


# In[37]:


import math
map_ = folium.Map(location=[54,15], tiles='cartodbpositron', zoom_start=2)

# Add points to the map
mc = MarkerCluster()
for idx, row in confirmed_df.iterrows():
    if not math.isnan(row['Long']) and not math.isnan(row['Lat']):
        mc.add_child(Marker([row['Lat'], row['Long']]))
map_.add_child(mc)

# Display the map
map_


# In[38]:


data_path = Path('/kaggle/input/covid19-global-forecasting-week-4/')
wk4train_df = pd.read_csv(data_path / 'train.csv')
wk4test_df = pd.read_csv(data_path / 'test.csv')


# In[39]:


print ('Training Data provided from', wk4train_df['Date'].min(),'to ', wk4train_df['Date'].max() )

print ('Test Data provided from', wk4test_df['Date'].min(),'to ', wk4test_df['Date'].max() )


# In[40]:


traintest_df = pd.concat([wk4train_df, wk4test_df])
print(wk4train_df.shape, wk4test_df.shape, traintest_df.shape)


# In[41]:


wk4train_df.rename(columns={'Province_State':'Province','Country_Region':'Country'}, inplace=True)
wk4test_df.rename(columns={'Province_State':'Province','Country_Region':'Country'}, inplace=True)
#clean_df.rename(columns={'Province/State':'Province','Country/Region':'Country'}, inplace=True)
traintest_df.rename(columns={'Province_State':'Province','Country_Region':'Country'}, inplace=True)


# In[42]:


wk4train_df['Date'] = pd.to_datetime(wk4train_df['Date'])
wk4test_df['Date'] = pd.to_datetime(wk4test_df['Date'])
traintest_df['Date'] = pd.to_datetime(traintest_df['Date'])


# In[43]:


def create_time_features(df):
    """
    Creates time series features from datetime index
    """
    #df['date'] = df.index
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


# In[44]:


create_time_features(wk4train_df).head()
create_time_features(wk4test_df).head()
create_time_features(traintest_df).head()


# In[45]:


print(wk4train_df["Date"].min(), "-", wk4train_df["Date"].max())
print(wk4test_df["Date"].min(), "-", wk4test_df["Date"].max())


# In[46]:


traintest_df.loc[traintest_df['Country'] == 'US', "Country"] = 'USA'

traintest_df.loc[traintest_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
traintest_df.loc[traintest_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
traintest_df.loc[traintest_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
traintest_df.loc[traintest_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
traintest_df.loc[traintest_df['Country'] == "Reunion", "Country"] = "Réunion"
traintest_df.loc[traintest_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'
traintest_df.loc[traintest_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
traintest_df.loc[traintest_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'
traintest_df.loc[traintest_df["Country"] == "Burma", "Country"] = "Myanmar"

traintest_df['Province'] = traintest_df['Province'].replace(np.nan,'')
traintest_df['Country_Province'] = traintest_df['Country'] + "." + traintest_df['Province']


# In[47]:


train_copy = wk4train_df.copy()
test_copy = wk4test_df.copy()
traintest_copy = traintest_df.copy()


# In[48]:


#Mobility
mobility_df=pd.read_csv("../input/google-cummunity-mobility-cv-19/2020-03-29-reports.csv")

mobility_df["mob_date"] = ""
mobility_df["country_alpha_2_code"] =""
mobility_df["country_alpha_3_code"] =""
mobility_df["Province"] =""
mobility_df["Country"] = ""
#mobility_df["continent"] = ""

for i in range(0, mobility_df.shape[0]):
    mobility_df["mob_date"][i] = mobility_df["file_name"][i][0:10]
    mobility_df["country_alpha_2_code"][i] = mobility_df["file_name"][i][11:13]   #l = len(mobility_df["file_name"][i])
    x = mobility_df['file_name'][i].split('.')    #x[0]
    pos = x[0].find('_Mobility')
    mobility_df["Province"][i] = mobility_df["file_name"][i][14:pos]
    mobility_df["Country"][i] = pc.country_alpha2_to_country_name(mobility_df['country_alpha_2_code'][i])
    mobility_df["country_alpha_3_code"][i] = pc.country_name_to_country_alpha3(mobility_df['Country'][i])

    
#mobility_df.tail(2)    

#mobility_df[mobility_df["Country"]=="United States"]
mobility_df.rename(columns={'Country_Region':'Country', 'Province_State':'Province'}, inplace=True)
mobility_df.loc[mobility_df["Country"] == "United States", "Country"] = "USA"
#mobility_df[mobility_df["Country"]=="USA"].head(3)

mobility_df.loc[mobility_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
mobility_df.loc[mobility_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
mobility_df.loc[mobility_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
mobility_df.loc[mobility_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
mobility_df.loc[mobility_df['Country'] == "Reunion", "Country"] = "Réunion"
mobility_df.loc[mobility_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'
mobility_df.loc[mobility_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
mobility_df.loc[mobility_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'

mobility_df['Province'] = mobility_df['Province'].replace(np.nan,'')
mobility_df['Country_Province'] = mobility_df['Country'] + "." + mobility_df['Province']


# In[49]:


# getting all countries
countries = np.asarray(mobility_df["Country"])

# Continent_code to Continent_names
continents = {
    'NA': 'North America',
    'SA': 'South America', 
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU' : 'Europe',
    'na' : 'Others'
}

# Defininng Function for getting continent code for country.
def country_to_continent_code(Country):
    try:
        return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(Country))
    except :
        return 'na'

#Collecting Continent Information
mobility_df.insert(1,"continent", [continents[country_to_continent_code(Country)] for Country in countries[:]])


# In[50]:


mobility_df_tmp = mobility_df.copy()
mobility_df_tmp = mobility_df_tmp.drop(['Country','Province'],  axis=1)

traintest_df = pd.merge(traintest_df, mobility_df_tmp, on='Country_Province', how='left')

traintest_df["mob_date"] = pd.to_datetime(traintest_df["mob_date"])
traintest_df.head(2)


# In[51]:


politics_governance_df = pd.read_csv('/kaggle/input/global-politcs-and-governance-data-apr-2020/politics_apr2020.csv')

politics_governance_df.rename(columns={'country':'Country'}, inplace=True)
politics_governance_df.head(2)

politics_governance_df.loc[politics_governance_df['Country'] == 'US', "Country"] = 'USA'

politics_governance_df.loc[politics_governance_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
politics_governance_df.loc[politics_governance_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
politics_governance_df.loc[politics_governance_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
politics_governance_df.loc[politics_governance_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
politics_governance_df.loc[politics_governance_df['Country'] == "Reunion", "Country"] = "Réunion"
politics_governance_df.loc[politics_governance_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'
politics_governance_df.loc[politics_governance_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
politics_governance_df.loc[politics_governance_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'
politics_governance_df.loc[politics_governance_df["Country"] == "Burma", "Country"] = "Myanmar"


# In[52]:


traintest_df = pd.merge(traintest_df, politics_governance_df, on='Country', how='left')
traintest_df.head()


# In[53]:


data_release_df = pd.read_csv ('/kaggle/input/covid19-country-data-wk3-release/Data Join - RELEASE.csv')
#data_release_df.isnull().sum()

data_release_df['temperature']=data_release_df['temperature'].replace(np.nan,'')
data_release_df['humidity']=data_release_df['humidity'].replace(np.nan,'')
data_release_df['Personality_pdi']=data_release_df['Personality_pdi'].replace(np.nan,'')
data_release_df['Personality_idv']=data_release_df['Personality_idv'].replace(np.nan,'')
data_release_df['Personality_mas']=data_release_df['Personality_mas'].replace(np.nan,'')
data_release_df['Personality_uai']=data_release_df['Personality_uai'].replace(np.nan,'')
data_release_df['Personality_ltowvs']=data_release_df['Personality_ltowvs'].replace(np.nan,'')
data_release_df['personality_perform']=data_release_df['personality_perform'].replace(np.nan,'')

data_release_df.rename(columns={'Country_Region':'Country', 'Province_State':'Province'}, inplace=True)

data_release_df.loc[data_release_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
data_release_df.loc[data_release_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
data_release_df.loc[data_release_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
data_release_df.loc[data_release_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
#data_release_df.loc[data_release_df['Country'] == "Reunion", "Country"] = "Réunion"
data_release_df.loc[data_release_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'

data_release_df.loc[data_release_df["Country"] == "Burma", "Country"] = "Myanmar"

data_release_df['Province'] = data_release_df['Province'].replace(np.nan,'')
data_release_df['Country_Province'] = data_release_df['Country'] + "." + data_release_df['Province']


# In[54]:


data_release_df_tmp = data_release_df.copy()
data_release_df_tmp = data_release_df_tmp.drop(['Country','Province'],  axis=1)

traintest_df = pd.merge(traintest_df, data_release_df_tmp, on='Country_Province', how='left')
traintest_df.shape


# In[55]:


#WDI reports
wdi_df = pd.read_csv('../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv')

# Display NaN records
wdi_nan_df = (wdi_df[wdi_df['Country_Region'].isna()])
wdi_nan_df.head(3)
#Drop those NaN records
wdi_df = wdi_df.drop(wdi_df[wdi_df['Country_Region'].isna()].index, axis = 0)
#wdi_df.shape
#wdi_df.head(3)

wdi_df['Health_exp_pct_GDP_2016']=wdi_df['Health_exp_pct_GDP_2016'].replace(np.nan,'')
wdi_df['Health_exp_public_pct_2016']=wdi_df['Health_exp_public_pct_2016'].replace(np.nan,'')
wdi_df['Health_exp_out_of_pocket_pct_2016']=wdi_df['Health_exp_out_of_pocket_pct_2016'].replace(np.nan,'')
wdi_df['Health_exp_per_capita_USD_2016']=wdi_df['Health_exp_per_capita_USD_2016'].replace(np.nan,'')
wdi_df['per_capita_exp_PPP_2016']=wdi_df['per_capita_exp_PPP_2016'].replace(np.nan,'')
wdi_df['External_health_exp_pct_2016']=wdi_df['External_health_exp_pct_2016'].replace(np.nan,'')
wdi_df['Physicians_per_1000_2009-18']=wdi_df['Physicians_per_1000_2009-18'].replace(np.nan,'')

wdi_df['Nurse_midwife_per_1000_2009-18']=wdi_df['Nurse_midwife_per_1000_2009-18'].replace(np.nan,'')
wdi_df['Specialist_surgical_per_1000_2008-18']=wdi_df['Specialist_surgical_per_1000_2008-18'].replace(np.nan,'')
wdi_df['Completeness_of_birth_reg_2009-18']=wdi_df['Completeness_of_birth_reg_2009-18'].replace(np.nan,'')
wdi_df['Completeness_of_death_reg_2008-16']=wdi_df['Completeness_of_death_reg_2008-16'].replace(np.nan,'')


wdi_df.rename(columns={'Country_Region':'Country', 'Province_State':'Province'}, inplace=True)
#wdi_df[wdi_df["Country"]=="US"]
wdi_df.loc[wdi_df["Country"] == "US", "Country"] = "USA"
#wdi_df[wdi_df["Country"]=="USA"]

wdi_df.loc[wdi_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
wdi_df.loc[wdi_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
wdi_df.loc[wdi_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
wdi_df.loc[wdi_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
wdi_df.loc[wdi_df['Country'] == "Reunion", "Country"] = "Réunion"
wdi_df.loc[wdi_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'
wdi_df.loc[wdi_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
wdi_df.loc[wdi_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'
wdi_df.loc[wdi_df["Country"] == "Burma", "Country"] = "Myanmar"

wdi_df['Province'] = wdi_df['Province'].replace(np.nan,'')
wdi_df['Country_Province'] = wdi_df['Country'] + "." + wdi_df['Province']


# In[56]:


wdi_df_tmp = wdi_df.copy()
wdi_df_tmp = wdi_df_tmp.drop(['Country','Province'],  axis=1)

traintest_df = pd.merge(traintest_df, wdi_df_tmp, on='Country_Province', how='left')
traintest_df.head()


# In[57]:


lockdown_df = pd.read_csv('/kaggle/input/covid19-lockdown-dates-by-country/countryLockdowndates.csv')
lockdownJHU_df = pd.read_csv('/kaggle/input/covid19-lockdown-dates-by-country/countryLockdowndatesJHUMatch.csv')
lockdownJHU_df.rename(columns={'Country/Region':'Country'}, inplace=True)

lockdownJHU_df.loc[lockdownJHU_df["Country"] == "US", "Country"] = "USA"
#lockdownJHU_df[lockdownJHU_df["Country"]=="USA"]

lockdownJHU_df.loc[lockdownJHU_df['Country'] == 'Korea, South', "Country"] = 'South Korea'
lockdownJHU_df.loc[lockdownJHU_df['Country'] == 'Taiwan*', "Country"] = 'Taiwan'
lockdownJHU_df.loc[lockdownJHU_df['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'
lockdownJHU_df.loc[lockdownJHU_df['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"
lockdownJHU_df.loc[lockdownJHU_df['Country'] == "Reunion", "Country"] = "Réunion"
#lockdownJHU_df.loc[lockdownJHU_df['Country'] == 'Congo (Brazzaville)', "Country"] = 'Democratic Republic of the Congo'
#lockdownJHU_df.loc[lockdownJHU_df['Country'] == 'DR Congo', "Country"] = 'Republic of the Congo'
lockdownJHU_df.loc[lockdownJHU_df['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'
lockdownJHU_df.loc[lockdownJHU_df['Country'] == 'Gambia, The', "Country"] = 'Gambia'
lockdownJHU_df.loc[lockdownJHU_df["Country"] == "Burma", "Country"] = "Myanmar"

lockdownJHU_df.rename(columns={'Date':'lockdown_date'}, inplace=True)

traintest_df = pd.merge(traintest_df, lockdownJHU_df, on='Country', how='left')
traintest_df['lockdown_date'] = pd.to_datetime(traintest_df['lockdown_date'])
traintest_df.head()


# In[58]:


#week4
day_before_valid = 78 + 7 # 3-11 day  before of validation
day_before_public = 85 +7 # 3-18 last day of train
day_before_launch = 92 + 7# 4-1 last day before launch
day_before_private = traintest_df['dayofyear'][pd.isna(traintest_df['ForecastId'])].max() # last day of train


# In[59]:


print ('Training Data provided from', wk4train_df['Date'].min(),'to ', wk4train_df['Date'].max() )

print ('Test Data provided from', wk4test_df['Date'].min(),'to ', wk4test_df['Date'].max() )


# In[60]:


def calc_score(y_true, y_pred):
    y_true[y_true<0] = 0
    score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5
    return score


# In[61]:


# covert object type to float
def func(x):
    x_new = 0
    try:
        x_new = float(x.replace(",", ""))
    except:
#         print(x)
        x_new = np.nan
    return x_new
cols = [
   'TRUE POPULATION', 
    ' TFR ', 'pct_in_largest_city', 
    ' Avg_age ', 'humidity',
    'temperature'  ,                                 
    'Personality_pdi', 'Personality_idv', 'Personality_mas',
       'Personality_uai', 'Personality_ltowvs', 
      'personality_perform', 'personality_agreeableness',
      'AIR_AVG',
       'Health_exp_pct_GDP_2016', 'Health_exp_public_pct_2016',
       'Health_exp_out_of_pocket_pct_2016', 'Health_exp_per_capita_USD_2016',
       'per_capita_exp_PPP_2016', 'External_health_exp_pct_2016',
       'Physicians_per_1000_2009-18', 'Nurse_midwife_per_1000_2009-18',
       'Specialist_surgical_per_1000_2008-18',
       'Completeness_of_birth_reg_2009-18',
       'Completeness_of_death_reg_2008-16'                                                                      
]
for col in cols:
    traintest_df[col] = traintest_df[col].apply(lambda x: func(x))  
print(traintest_df['AIR_AVG'].dtype)


# In[62]:


traintest_df['country_alpha_2_code'] = traintest_df['country_alpha_2_code'].astype(str)
traintest_df['country_alpha_3_code'] = traintest_df['country_alpha_3_code'].astype(str)
traintest_df['leader'] = traintest_df['leader'].astype(str)
traintest_df['government'] = traintest_df['government'].astype(str)
traintest_df['World_Bank_Name'] = traintest_df['World_Bank_Name'].astype(str)
traintest_df['Type'] = traintest_df['Type'].astype(str)


# In[63]:


traintest_df.dtypes


# In[64]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

traintest_df['country_encoded'] = labelencoder.fit_transform(traintest_df['Country'])
traintest_df['province_encoded'] = labelencoder.fit_transform(traintest_df['Province'])
traintest_df['cnpn_encoded'] = labelencoder.fit_transform(traintest_df['Country_Province'])
traintest_df['cn_alp2_encoded'] = labelencoder.fit_transform(traintest_df['country_alpha_2_code'])
traintest_df['cn_alp3_encoded'] = labelencoder.fit_transform(traintest_df['country_alpha_3_code'])
traintest_df['leader_encoded'] = labelencoder.fit_transform(traintest_df['leader'])
traintest_df['gn_encoded'] = labelencoder.fit_transform(traintest_df['government'])
traintest_df['wbn_encoded'] = labelencoder.fit_transform(traintest_df['World_Bank_Name'])
traintest_df['Type_encoded'] = labelencoder.fit_transform(traintest_df['Type'])


# In[65]:


#traintest_df.loc[traintest_df["Date"]<"2020-03-20", "split"] = "train"
#traintest_df.loc[traintest_df["Date"]>="2020-03-20", "split"] = "test"


# In[66]:


traintest_df.shape


# In[67]:


'''
plt.figure(figsize=(35,20))
a = sns.heatmap(traintest_df.dropna().corr(), annot = True, cmap = 'cubehelix')
a.Title = ' Week3 - COVID19 - Data Correlation';
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=45)
plt.show()
'''


# In[68]:


# get place list
places = np.sort(traintest_df['Country_Province'].unique())
print(len(places))


# In[69]:


# calc cases, fatalities per day
traintest_df1 = copy.deepcopy(traintest_df)
traintest_df1['cases/day'] = 0
traintest_df1['fatal/day'] = 0
tmp_list = np.zeros(len(traintest_df1))
for place in places:
    tmp = traintest_df1['ConfirmedCases'][traintest_df1['Country_Province']==place].values
    tmp[1:] -= tmp[:-1]
    traintest_df1['cases/day'][traintest_df1['Country_Province']==place] = tmp
    tmp = traintest_df1['Fatalities'][traintest_df1['Country_Province']==place].values
    tmp[1:] -= tmp[:-1]
    traintest_df1['fatal/day'][traintest_df1['Country_Province']==place] = tmp
print(traintest_df1.shape)
traintest_df1[traintest_df1['Country_Province']=='Italy.'].head()


# In[70]:


# aggregate cases and fatalities
def do_aggregation(df, col, mean_range):
    df_new = copy.deepcopy(df)
    col_new = '{}_({}-{})'.format(col, mean_range[0], mean_range[1])
    df_new[col_new] = 0
    tmp = df_new[col].rolling(mean_range[1]-mean_range[0]+1).mean()
    df_new[col_new][mean_range[0]:] = tmp[:-(mean_range[0])]
    df_new[col_new][pd.isna(df_new[col_new])] = 0
    return df_new[[col_new]].reset_index(drop=True)

def do_aggregations(df):
    df = pd.concat([df, do_aggregation(df, 'cases/day', [1,1]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'cases/day', [1,7]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'cases/day', [8,14]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'cases/day', [15,21]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal/day', [1,1]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal/day', [1,7]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal/day', [8,14]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal/day', [15,21]).reset_index(drop=True)], axis=1)
    for threshold in [1, 10, 100]:
        days_under_threshold = (df['ConfirmedCases']<threshold).sum()
        tmp = df['dayofyear'].values - 22 - days_under_threshold
        tmp[tmp<=0] = 0
        df['days_since_{}cases'.format(threshold)] = tmp
            
    for threshold in [1, 10, 100]:
        days_under_threshold = (df['Fatalities']<threshold).sum()
        tmp = df['dayofyear'].values - 22 - days_under_threshold
        tmp[tmp<=0] = 0
        df['days_since_{}fatal'.format(threshold)] = tmp
    
    # process China/Hubei
    if df['Country_Province'][0]=='China/Hubei':
        df['days_since_1cases'] += 35 # 2019/12/8
        df['days_since_10cases'] += 35-13 # 2019/12/8-2020/1/2 assume 2019/12/8+13
        df['days_since_100cases'] += 4 # 2020/1/18
        df['days_since_1fatal'] += 13 # 2020/1/9
    return df


# In[71]:


traintest_df_cc = copy.deepcopy(traintest_df)


# In[72]:


traintest_df1[traintest_df1['Country_Province']=='Italy.'].head(2)


# In[73]:


traintest_df2 = []
for place in places[:]:
    df_tmp = traintest_df1[traintest_df1['Country_Province']==place].reset_index(drop=True)
    df_tmp = do_aggregations(df_tmp)
    traintest_df2.append(df_tmp)
traintest_df2 = pd.concat(traintest_df2).reset_index(drop=True)
traintest_df2[traintest_df2['Country_Province']=='Italy.'].head()


# In[74]:


traintest_df2['cases/day'] = traintest_df2['cases/day'].astype(np.float)
traintest_df2['fatal/day'] = traintest_df2['fatal/day'].astype(np.float)


# In[75]:


traintest_df2.shape


# In[76]:


import xgboost as xgb
params = {"objective":"reg:squaredlogerror",'colsample_bytree': 0.3,'learning_rate': 0.015,
                'max_depth': 5, 'alpha': 0.15}

#cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
#                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[77]:


# train model to predict fatalities/day
# features are selected manually based on valid score
col_target = 'fatal/day'
col_var = [
 'ForecastId' ,                                
'Id',                                            
'hour',                                            
'dayofweek',                                       
'quarter',                                         
'month',                                        
'year',                                           
'dayofyear',                                       
'dayofmonth',                                      
'weekofyear',                                      
'retail_recreation',                             
'grocery_pharmacy',                              
'parks',                                         
'transit_station',                               
'workplaces',                                    
'residential',                                   
'ccode',                                         
'elected',                                      
'age',                                          
'male',                                          
'militarycareer',                                
'tenure_months',                                 
'anticipation',                                  
'ref_ant',                                       
'leg_ant',                                       
'exec_ant',                                      
'irreg_lead_ant',                                
'election_now',                                
'election_recent',                               
'leg_recent',                                   
'exec_recent',                                   
'lead_recent',                                  
'ref_recent',                                    
'direct_recent',                                 
'indirect_recent',                             
'victory_recent',                                
'defeat_recent',                                 
'change_recent',                             
'nochange_recent',                              
'delayed',                                      
'prev_conflict',                                 
'GDP_region',                                    
'latitude',                                      
'longitude',                                     
'abs_latitude',                                  
'murder',                                        
'High_rises',                                    
'max_high_rises',                                
'AIR_CITIES',                                                                           
'continent_gdp_pc',                              
'continent_happiness',                           
'continent_generosity',                          
'continent_corruption',                         
'continent_Life_expectancy',                      
     'days_since_1cases', 
     'days_since_10cases', 
     'days_since_100cases',
     'days_since_1fatal', 
     'days_since_10fatal', 'days_since_100fatal',
    'cases/day_(1-1)', 
    'cases/day_(1-7)', 
     'cases/day_(8-14)',  
     'cases/day_(15-21)',     
     'fatal/day_(1-1)', 
    'fatal/day_(1-7)', 
    'fatal/day_(8-14)', 
    'fatal/day_(15-21)', 
  'TRUE POPULATION', 
    ' TFR ', 'pct_in_largest_city', 
    ' Avg_age ', 'humidity',
    'temperature'  ,                                 
    'Personality_pdi', 'Personality_idv', 'Personality_mas',
       'Personality_uai', 'Personality_ltowvs', 'Personality_assertive',
      'personality_perform', 'personality_agreeableness',
      'AIR_AVG',
       'Health_exp_pct_GDP_2016', 'Health_exp_public_pct_2016',
       'Health_exp_out_of_pocket_pct_2016', 'Health_exp_per_capita_USD_2016',
       'per_capita_exp_PPP_2016', 'External_health_exp_pct_2016',
       'Physicians_per_1000_2009-18', 'Nurse_midwife_per_1000_2009-18',
        'Type_encoded', 'leader_encoded',
    'country_encoded','province_encoded',
    'cnpn_encoded','cn_alp2_encoded',
    'cn_alp3_encoded','gn_encoded','wbn_encoded',
       'Specialist_surgical_per_1000_2008-18',
       'Completeness_of_birth_reg_2009-18',
      'Completeness_of_death_reg_2008-16'   
]
col_cat = []
df_train = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (traintest_df2['dayofyear']<=day_before_valid)]
df_valid = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (day_before_valid<traintest_df2['dayofyear']) & (traintest_df2['dayofyear']<=day_before_public)]
df_test = traintest_df2[pd.isna(traintest_df2['ForecastId'])==False]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = xgb.DMatrix(np.asarray(X_train), y_train)
valid_data = xgb.DMatrix(np.asarray(X_valid), y_valid)
xgb0 = xgb.train( params, train_data, evals=[(train_data, 'train')], num_boost_round=10000,early_stopping_rounds=10,verbose_eval=False ) 


# In[78]:


y_true = df_valid['fatal/day'].values
y_pred = np.exp(xgb0.predict(valid_data))-1
score = calc_score(y_true, y_pred)
print("{:.6f}".format(score))


# In[79]:


# train with all data before public
df_train = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (traintest_df2['dayofyear']<=day_before_public)]
df_valid = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (traintest_df2['dayofyear']<=day_before_public)]
df_test = traintest_df2[pd.isna(traintest_df2['ForecastId'])==False]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = xgb.DMatrix(np.asarray(X_train), y_train)
valid_data = xgb.DMatrix(np.asarray(X_valid), y_valid)

xgb1 = xgb.train( params, train_data, evals=[(train_data, 'train')], num_boost_round=10000,early_stopping_rounds=10,verbose_eval=False ) 


# In[80]:


# train model to predict cases/day
col_target2 = 'cases/day'
col_var2 = [
 'ForecastId' ,                                
'Id',                                            
'hour',                                            
'dayofweek',                                       
'quarter',                                         
'month',                                        
'year',                                           
'dayofyear',                                       
'dayofmonth',                                      
'weekofyear',                                      
'retail_recreation',                             
'grocery_pharmacy',                              
'parks',                                         
'transit_station',                               
'workplaces',                                    
'residential',                                   
'ccode',                                         
'elected',                                      
'age',                                          
'male',                                          
'militarycareer',                                
'tenure_months',                                 
'anticipation',                                  
'ref_ant',                                       
'leg_ant',                                       
'exec_ant',                                      
'irreg_lead_ant',                                
'election_now',                                
'election_recent',                               
'leg_recent',                                   
'exec_recent',                                   
'lead_recent',                                  
'ref_recent',                                    
'direct_recent',                                 
'indirect_recent',                             
'victory_recent',                                
'defeat_recent',                                 
'change_recent',                             
'nochange_recent',                              
'delayed',                                      
'prev_conflict',                                 
'GDP_region',                                    
'latitude',                                      
'longitude',                                     
'abs_latitude',                                  
'murder',                                        
'High_rises',                                    
'max_high_rises',                                
'AIR_CITIES',                                                                           
'continent_gdp_pc',                              
'continent_happiness',                           
'continent_generosity',                          
'continent_corruption',                         
'continent_Life_expectancy',                      
     'days_since_1cases', 
     'days_since_10cases', 
     'days_since_100cases',
     'days_since_1fatal', 
     'days_since_10fatal', 'days_since_100fatal',
    'cases/day_(1-1)', 
    'cases/day_(1-7)', 
     'cases/day_(8-14)',  
     'cases/day_(15-21)',     
     'fatal/day_(1-1)', 
    'fatal/day_(1-7)', 
    'fatal/day_(8-14)', 
    'fatal/day_(15-21)', 
  'TRUE POPULATION', 
    ' TFR ', 'pct_in_largest_city', 
    ' Avg_age ', 'humidity',
    'temperature'  ,                                 
    'Personality_pdi', 'Personality_idv', 'Personality_mas',
       'Personality_uai', 'Personality_ltowvs', 'Personality_assertive',
      'personality_perform', 'personality_agreeableness',
      'AIR_AVG',
       'Health_exp_pct_GDP_2016', 'Health_exp_public_pct_2016',
       'Health_exp_out_of_pocket_pct_2016', 'Health_exp_per_capita_USD_2016',
       'per_capita_exp_PPP_2016', 'External_health_exp_pct_2016',
       'Physicians_per_1000_2009-18', 'Nurse_midwife_per_1000_2009-18',
        'Type_encoded', 'leader_encoded',
    'country_encoded','province_encoded',
    'cnpn_encoded','cn_alp2_encoded',
    'cn_alp3_encoded','gn_encoded','wbn_encoded',
       'Specialist_surgical_per_1000_2008-18',
       'Completeness_of_birth_reg_2009-18',
      'Completeness_of_death_reg_2008-16'  
]
col_cat = []
df_train = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (traintest_df2['dayofyear']<=day_before_valid)]
df_valid = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (day_before_valid<traintest_df2['dayofyear']) & (traintest_df2['dayofyear']<=day_before_public)]
X_train = df_train[col_var2]
X_valid = df_valid[col_var2]
y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = xgb.DMatrix(np.asarray(X_train), y_train)
valid_data = xgb.DMatrix(np.asarray(X_valid), y_valid)

xgb2 = xgb.train( params, train_data, evals=[(train_data, 'train')], num_boost_round=10000,early_stopping_rounds=10, verbose_eval=False ) 


# In[81]:


y_true = df_valid['cases/day'].values
y_pred = np.exp(xgb2.predict(valid_data))-1
score = calc_score(y_true, y_pred)
print("{:.6f}".format(score))


# In[82]:


df_train = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (traintest_df2['dayofyear']<=day_before_public)]
df_valid = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (traintest_df2['dayofyear']<=day_before_public)]
X_train = df_train[col_var2]
X_valid = df_valid[col_var2]
y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = xgb.DMatrix(np.asarray(X_train), y_train)
valid_data = xgb.DMatrix(np.asarray(X_valid), y_valid)

xgb3 = xgb.train( params, train_data,  evals=[(train_data, 'train')], num_boost_round=10000,early_stopping_rounds=10, verbose_eval=False ) 


# In[83]:


# train model to predict fatalities/day
df_train = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (traintest_df2['dayofyear']<=day_before_public)]
df_valid = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (day_before_public<traintest_df2['dayofyear'])]
df_test = traintest_df2[pd.isna(traintest_df2['ForecastId'])==False]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)
train_data = xgb.DMatrix(np.asarray(X_train), y_train)
valid_data = xgb.DMatrix(np.asarray(X_valid), y_valid)

xgb4 = xgb.train( params, train_data, evals=[(train_data, 'train')], num_boost_round=10000,early_stopping_rounds=10, verbose_eval=False ) 


# In[84]:


# train with all data
df_train = traintest_df2[(pd.isna(traintest_df2['ForecastId']))]
df_valid = traintest_df2[(pd.isna(traintest_df2['ForecastId']))]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = xgb.DMatrix(np.asarray(X_train), y_train)
valid_data = xgb.DMatrix(np.asarray(X_valid), y_valid)

xgb5 = xgb.train( params, train_data, evals=[(train_data, 'train')], num_boost_round=10000,early_stopping_rounds=10, verbose_eval=False ) 


# In[85]:


# train model to predict cases/day
df_train = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (traintest_df2['dayofyear']<=day_before_public)]
df_valid = traintest_df2[(pd.isna(traintest_df2['ForecastId'])) & (day_before_public<traintest_df2['dayofyear'])]
X_train = df_train[col_var2]
X_valid = df_valid[col_var2]
y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = xgb.DMatrix(np.asarray(X_train), y_train)
valid_data = xgb.DMatrix(np.asarray(X_valid), y_valid)

xgb6 = xgb.train( params, train_data, evals=[(train_data, 'train')], num_boost_round=10000,early_stopping_rounds=10, verbose_eval=False ) 


# In[86]:


# train with all data
df_train = traintest_df2[(pd.isna(traintest_df2['ForecastId']))]
df_valid = traintest_df2[(pd.isna(traintest_df2['ForecastId']))]
X_train = df_train[col_var2]
X_valid = df_valid[col_var2]
y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)


train_data = xgb.DMatrix(np.asarray(X_train), y_train)
valid_data = xgb.DMatrix(np.asarray(X_valid), y_valid)

xgb7 = xgb.train( params, train_data, evals=[(train_data, 'train')], num_boost_round=10000,early_stopping_rounds=10, verbose_eval=False ) 



# In[87]:


# remove overlap for public LB prediction
df_tmp = traintest_df2[
    ((traintest_df2['dayofyear']<=day_before_public)  & (pd.isna(traintest_df2['ForecastId'])))
    | ((day_before_public<traintest_df2['dayofyear']) & (pd.isna(traintest_df2['ForecastId'])==False))].reset_index(drop=True)
df_tmp
df_tmp = df_tmp.drop([
    'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 
    'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',
    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal'
                               ],  axis=1)
traintest_df3 = []
for i, place in enumerate(places[:]):
    df_tmp2 = df_tmp[df_tmp['Country_Province']==place].reset_index(drop=True)
    df_tmp2 = do_aggregations(df_tmp2)
    traintest_df3.append(df_tmp2)
traintest_df3 = pd.concat(traintest_df3).reset_index(drop=True)
traintest_df3[traintest_df3['dayofyear']>day_before_public-2].head()


# In[88]:


# remove overlap for private LB prediction
df_tmp = traintest_df2[
    ((traintest_df2['dayofyear']<=day_before_private)  & (pd.isna(traintest_df2['ForecastId'])))
    | ((day_before_private<traintest_df2['dayofyear']) & (pd.isna(traintest_df2['ForecastId'])==False))].reset_index(drop=True)

df_tmp
df_tmp = df_tmp.drop([
    'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 
    'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',
    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal'
                               ],  axis=1)
traintest_df4 = []
for i, place in enumerate(places[:]):
    df_tmp2 = df_tmp[df_tmp['Country_Province']==place].reset_index(drop=True)
    df_tmp2 = do_aggregations(df_tmp2)
    traintest_df4.append(df_tmp2)
traintest_df4 = pd.concat(traintest_df4).reset_index(drop=True)
traintest_df4[traintest_df4['dayofyear']>day_before_private-2].head()


# In[89]:


# predict test data in public
# predict the cases and fatatilites one day at a time and use the predicts as next day's feature recursively.
df_preds = []
for i, place in enumerate(places[:]):
    df_interest = copy.deepcopy(traintest_df3[traintest_df3['Country_Province']==place].reset_index(drop=True))
    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    len_known = (df_interest['dayofyear']<=day_before_public).sum()
    len_unknown = (day_before_public<df_interest['dayofyear']).sum()
    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction
        X_valid = df_interest[col_var].iloc[j+len_known]
        X_valid2 = df_interest[col_var2].iloc[j+len_known]
        validata = xgb.DMatrix(X_valid)
        pred_f = xgb0.predict(validata)
        #xgb.plot_importance()
        validata2 = xgb.DMatrix(X_valid2)
        pred_c = xgb2.predict(validata2)
        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)
        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)
        df_interest['fatal/day'][j+len_known] = pred_f
        df_interest['cases/day'][j+len_known] = pred_c
        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f
        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c
#         print(df_interest['ConfirmedCases'][j+len_known-1], df_interest['ConfirmedCases'][j+len_known], pred_c)
        df_interest = df_interest.drop([
            'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 
            'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',
            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal'

                                       ],  axis=1)
        df_interest = do_aggregations(df_interest)
    if (i+1)%5==0:
        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)
    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)
    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)
    df_preds.append(df_interest)
df_preds = pd.concat(df_preds)


# In[90]:


# predict test data in public
df_preds_pri = []
for i, place in enumerate(places[:]):
    df_interest = copy.deepcopy(traintest_df4[traintest_df4['Country_Province']==place].reset_index(drop=True))
    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    len_known = (df_interest['dayofyear']<=day_before_private).sum()
    len_unknown = (day_before_private<df_interest['dayofyear']).sum()
    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction
        X_valid = df_interest[col_var].iloc[j+len_known]
        X_valid2 = df_interest[col_var2].iloc[j+len_known]
        validata = xgb.DMatrix(X_valid)
        pred_f = xgb5.predict(validata)
        validata2 = xgb.DMatrix(X_valid)
        pred_c = xgb7.predict(validata2)
        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)
        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)
        df_interest['fatal/day'][j+len_known] = pred_f
        df_interest['cases/day'][j+len_known] = pred_c
        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f
        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c
#         print(df_interest['ConfirmedCases'][j+len_known-1], df_interest['ConfirmedCases'][j+len_known], pred_c)
        df_interest = df_interest.drop([
            'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 
            'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',
            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal'
                                       ],  axis=1)
        df_interest = do_aggregations(df_interest)
    if (i+1)%5==0:
        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)
    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)
    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)
    df_preds_pri.append(df_interest)
df_preds_pri = pd.concat(df_preds_pri)


# In[91]:


# merge 2 preds
df_preds[df_preds['dayofyear']>day_before_private] = df_preds_pri[df_preds['dayofyear']>day_before_private]


# In[92]:


#df_preds.to_csv("df_preds.csv", index=None)


# In[93]:


# load sample submission
df_sub = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
print(len(df_sub))
df_sub.head()
# merge prediction with sub
df_sub = pd.merge(df_sub, traintest_df3[['ForecastId', 'Country_Province', 'dayofyear']])
df_sub = pd.merge(df_sub, df_preds[['Country_Province', 'dayofyear', 'cases_pred', 'fatal_pred']], on=['Country_Province', 'dayofyear',], how='left')
df_sub.head(10)


# In[94]:


# save
df_sub['ConfirmedCases'] = df_sub['cases_pred']
df_sub['Fatalities'] = df_sub['fatal_pred']
df_sub = df_sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]
df_sub.to_csv("submission.csv", index=None)
df_sub.head(10)

