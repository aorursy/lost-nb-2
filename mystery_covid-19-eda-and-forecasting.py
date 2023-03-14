#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from IPython.display import Markdown
from datetime import timedelta
from datetime import datetime

import plotly.express as px
import plotly.graph_objs as go
import pycountry
from plotly.offline import init_notebook_mode, iplot 
import plotly.offline as py
import plotly.express as ex
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import matplotlib.pyplot as plt
py.init_notebook_mode(connected=True)
plt.style.use("seaborn-talk")
plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['image.cmap'] = 'viridis'
import folium

from fbprophet import Prophet
from fbprophet.plot import plot_plotly

pd.set_option('display.max_rows', None)
from math import sin, cos, sqrt, atan2, radians
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn import preprocessing
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


# Load Data
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['ObservationDate'])
df.drop(['SNo','Last Update'],axis =1, inplace = True)
df['Active'] = df['Confirmed'] - (df['Recovered'] + df['Deaths'])
full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 
                         parse_dates=['Date'])
#train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
#test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
#submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
week5_train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
week5_test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
week5_sub = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')


# In[3]:


full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']
# replacing Mainland china with just China
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')
# filling missing values 
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']] = full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0)
# fixing datatypes
full_table['Recovered'] = full_table['Recovered'].astype(int)


# In[4]:


full_grouped = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
# new cases ======================================================
temp = full_grouped.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
mask = temp['Country/Region'] != temp['Country/Region'].shift(1)
temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan
# renaming columns
temp.columns = ['Country/Region', 'Date', 'New cases', 'New deaths', 'New recovered']
# =================================================================
# merging new values
full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])
# filling na with 0
full_grouped = full_grouped.fillna(0)
# fixing data types
cols = ['New cases', 'New deaths', 'New recovered']
full_grouped[cols] = full_grouped[cols].astype('int')
full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x<0 else x)


# In[5]:


country_wise = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)
# group by country
country_wise = country_wise.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()
# per 100 cases
country_wise['Deaths / 100 Cases'] = round((country_wise['Deaths']/country_wise['Confirmed'])*100, 2)
country_wise['Recovered / 100 Cases'] = round((country_wise['Recovered']/country_wise['Confirmed'])*100, 2)
country_wise['Deaths / 100 Recovered'] = round((country_wise['Deaths']/country_wise['Recovered'])*100, 2)
cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']
country_wise[cols] = country_wise[cols].fillna(0)


today = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)[['Country/Region', 'Confirmed']]
last_week = full_grouped[full_grouped['Date']==max(full_grouped['Date'])-timedelta(days=7)].reset_index(drop=True).drop('Date', axis=1)[['Country/Region', 'Confirmed']]
temp = pd.merge(today, last_week, on='Country/Region', suffixes=(' today', ' last week'))
# temp = temp[['Country/Region', 'Confirmed last week']]
temp['1 week change'] = temp['Confirmed today'] - temp['Confirmed last week']
temp = temp[['Country/Region', 'Confirmed last week', '1 week change']]
country_wise = pd.merge(country_wise, temp, on='Country/Region')
country_wise['1 week % increase'] = round(country_wise['1 week change']/country_wise['Confirmed last week']*100, 2)


# In[6]:


day_wise = full_grouped.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()
# number cases per 100 cases
day_wise['Deaths / 100 Cases'] = round((day_wise['Deaths']/day_wise['Confirmed'])*100, 2)
day_wise['Recovered / 100 Cases'] = round((day_wise['Recovered']/day_wise['Confirmed'])*100, 2)
day_wise['Deaths / 100 Recovered'] = round((day_wise['Deaths']/day_wise['Recovered'])*100, 2)
# no. of countries
day_wise['No. of countries'] = full_grouped[full_grouped['Confirmed']!=0].groupby('Date')['Country/Region'].unique().apply(len).values
# fillna by 0
cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']
day_wise[cols] = day_wise[cols].fillna(0)


# In[7]:


'''
train.rename(columns={'Country_Region':'Country','Province_State':'State'}, inplace=True)
test.rename(columns={'Country_Region':'Country','Province_State':'State'}, inplace=True)
train.rename(columns={'Province_State':'State'}, inplace=True)
test.rename(columns={'Province_State':'State'}, inplace=True)
train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)
test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)
'''


# In[8]:


'''
y1_xTrain = train.iloc[:, -2]
y1_xTrain.head()
y2_xTrain = train.iloc[:, -1]
y2_xTrain.head()

missing_value = "empty"
def fillState(state, country):
    if state == missing_value: return country
    return state
    
'''


# In[9]:


'''
train_x = train.copy()
train_x['State'].fillna(missing_value, inplace=True)
train_x['State'] = train_x.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)
train_x.loc[:, 'Date'] = train_x.Date.dt.strftime("%m%d")
train_x["Date"]  = train_x["Date"].astype(int)

test_x = test.copy()
test_x['State'].fillna(missing_value, inplace=True)
test_x['State'] = test_x.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)
test_x.loc[:, 'Date'] = test_x.Date.dt.strftime("%m%d")
test_x["Date"]  =test_x["Date"].astype(int')
'''


# In[10]:


week5_train = week5_train.drop(columns = ['County' , 'Province_State'])
week5_test = week5_test.drop(columns = ['County' , 'Province_State'])
week5_train['Date']= pd.to_datetime(week5_train['Date']).dt.strftime("%Y%m%d").astype(int)
week5_test['Date'] = pd.to_datetime(week5_test['Date']).dt.strftime("%Y%m%d").astype(int)


# In[11]:


date_wise_data = df[['Country/Region',"ObservationDate","Confirmed","Deaths","Recovered",'Active']]
date_wise_data['Date'] = date_wise_data['ObservationDate'].apply(pd.to_datetime, dayfirst=True)
date_wise_data = date_wise_data.groupby(["ObservationDate"]).sum().reset_index()
date_wise_data.rename({"ObservationDate": 'Date','Recovered':'Cured'}, axis=1,inplace= True) 
def formatted_text(string):
    display(Markdown(string))
#date_wise_data.to_csv('date_wise_data.csv')


# In[12]:


# Converting columns into numberic for Train ======================================================
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
w5_X = week5_train.iloc[:,1].values
week5_train.iloc[:,1] = labelencoder.fit_transform(w5_X.astype(str))
w5_X = week5_train.iloc[:,5].values
week5_train.iloc[:,5] = labelencoder.fit_transform(w5_X)

#Converting columns into numberic Test ======================================================
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
w5te_X = week5_test.iloc[:,1].values
week5_test.iloc[:,1] = labelencoder.fit_transform(w5te_X)
w5te_X = week5_test.iloc[:,5].values
week5_test.iloc[:,5] = labelencoder.fit_transform(w5te_X)

#Train & Test ======================================================
x = week5_train.iloc[:,1:6]
y = week5_train.iloc[:,6]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y, test_size = 0.2, random_state = 0 )


# In[13]:


#Adding Population Data
pop = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")
# select only population
pop = pop.iloc[:, :2]
# rename column names
pop.columns = ['Country/Region', 'Population']
# merged data
country_wise = pd.merge(country_wise, pop, on='Country/Region', how='left')
# update population
cols = ['Burma', 'Congo (Brazzaville)', 'Congo (Kinshasa)', "Cote d'Ivoire", 'Czechia', 
        'Kosovo', 'Saint Kitts and Nevis', 'Saint Vincent and the Grenadines', 
        'Taiwan*', 'US', 'West Bank and Gaza']
pops = [54409800, 89561403, 5518087, 26378274, 10708981, 1793000, 
        53109, 110854, 23806638, 330541757, 4543126]
for c, p in zip(cols, pops):
    country_wise.loc[country_wise['Country/Region']== c, 'Population'] = p
country_wise['Cases / Million People'] = round((country_wise['Confirmed'] / country_wise['Population']) * 1000000)


# In[14]:


temp = country_wise.copy()
temp = temp.iloc[:,:6]
temp = temp.sort_values('Confirmed',ascending=False).reset_index()
temp.style.background_gradient(cmap='Blues',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='Purples',subset=["Active"])                        .background_gradient(cmap='PuBu',subset=["New cases"])
                        


# In[15]:


sir_data = country_wise.copy()
sir_data["Susceptible"] = sir_data['Population'] - sir_data['Confirmed']
sir_data["Infected"] = sir_data['Confirmed'] - sir_data['Recovered'] - sir_data['Deaths']
#sir_data["Recovered"] = sir_data['Recovered']
sir_data["Fatal"] = sir_data.loc[:, 'Deaths']
response_variables = ["Susceptible", "Infected", "Recovered", "Fatal"]


# In[16]:


temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
temp1 = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])
fig = px.pie(temp1,
             values= 'value',labels=['Active Cases','Cured','Death'],
             names="variable",
             title="Current Situation of COVID-19 in the world",
             template="seaborn")
fig.update_traces(hoverinfo='label+percent',textinfo='value', textfont_size=14,
                  marker=dict(colors=['#263fa3','#cc3c2f','#2fcc41'], line=dict(color='#FFFFFF', width=2)))
fig.update_traces(textposition='inside')
#fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[17]:


temp = date_wise_data.copy()
fig = go.Figure(data=[
go.Bar(name='Deaths', x=temp['Date'], y=temp['Deaths'],marker_color='#ff0000'),
go.Bar(name='Recovered Cases', x=temp['Date'], y=temp['Cured'],marker_color='#2bad57'),
go.Bar(name='Confirmed Cases', x=temp['Date'], y=temp['Confirmed'],marker_color='#326ac7')])
fig.update_layout(barmode='stack')
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.update_layout(title_text='Distribution of COVID-19 Confirmed Cases,Recovered Cases and Deaths',
                  plot_bgcolor='rgb(275, 270, 273)')
fig.show()


# In[18]:


perday2 = date_wise_data.groupby(['Date'])['Confirmed','Cured','Deaths','Active'].sum().reset_index().sort_values('Date',ascending = True)
perday2['New Daily Confirmed Cases'] = perday2['Confirmed'].sub(perday2['Confirmed'].shift())
perday2['New Daily Confirmed Cases'].iloc[0] = perday2['Confirmed'].iloc[0]
perday2['New Daily Confirmed Cases'] = perday2['New Daily Confirmed Cases'].astype(int)
perday2['New Daily Cured Cases'] = perday2['Cured'].sub(perday2['Cured'].shift())
perday2['New Daily Cured Cases'].iloc[0] = perday2['Cured'].iloc[0]
perday2['New Daily Cured Cases'] = perday2['New Daily Cured Cases'].astype(int)
perday2['New Daily Deaths Cases'] = perday2['Deaths'].sub(perday2['Deaths'].shift())
perday2['New Daily Deaths Cases'].iloc[0] = perday2['Deaths'].iloc[0]
perday2['New Daily Deaths Cases'] = perday2['New Daily Deaths Cases'].astype(int)
perday2.to_csv('perday_daily_cases.csv')


# In[19]:


import plotly.express as px
fig = px.bar(perday2, x="Date", y="New Daily Confirmed Cases", barmode='group',height=500)
fig.update_layout(title_text='New COVID-19 cases reported daily all over the World',plot_bgcolor='rgb(275, 270, 273)')
fig.show()


# In[20]:


import plotly.express as px
fig = px.bar(perday2, x="Date", y="New Daily Cured Cases", barmode='group',height=500,
            color_discrete_sequence = ['#319146'])
fig.update_layout(title_text='New COVID-19 Recovered cases reported daily all over the world',plot_bgcolor='rgb(275, 270, 273)')
fig.show()


# In[21]:


import plotly.express as px
fig = px.bar(perday2, x="Date", y="New Daily Deaths Cases", barmode='group',height=500,
             color_discrete_sequence = ['#e31010'])
fig.update_layout(title_text='New COVID-19 Deaths reported daily all over the India',plot_bgcolor='rgb(275, 270, 273)')
fig.show()


# In[22]:


temp = date_wise_data.copy()
temp = date_wise_data.groupby('Date')['Confirmed', 'Deaths', 'Cured'].sum().reset_index()
fig = px.scatter(temp, x="Date", y="Confirmed", color="Confirmed",
                 size='Confirmed', hover_data=['Confirmed'],
                 color_discrete_sequence = ex.colors.cyclical.IceFire)
fig.update_layout(title_text='Trend of Daily Coronavirus Cases in India',
                  plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig.show()


# In[23]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=date_wise_data['Date'], y=date_wise_data['Confirmed'],
                    mode='lines+markers',marker_color='blue',name='Confimed Cases'))
fig.add_trace(go.Scatter(x=date_wise_data['Date'],y=date_wise_data['Active'], 
                mode='lines+markers',marker_color='purple',name='Active Cases'))
fig.add_trace(go.Scatter(x=date_wise_data['Date'], y=date_wise_data['Cured'],
                mode='lines+markers',marker_color='green',name='Recovered'))
fig.add_trace(go.Scatter(x=date_wise_data['Date'], y=date_wise_data['Deaths'], 
                mode='lines+markers',marker_color='red',name='Deaths'))
fig.update_layout(title_text='Trend of Novel Coronavirus Cases Globaly',plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig.show()


# In[24]:


cnf = '#263fa3' # confirmed - blue
act = '#fe9801' # active case - yellow
rec = '#21bf73' # recovered - green
dth = '#de260d' # death - red
tmp = date_wise_data.melt(id_vars="Date",value_vars=['Deaths','Cured' ,'Active','Confirmed'],
                 var_name='Case',value_name='Count')
fig = px.area(tmp, x="Date", y="Count",color='Case',
              title='Trend Over Weeks',color_discrete_sequence = [dth,rec,act,cnf])
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=550, height=600)
fig.show()


# In[25]:


temp = date_wise_data.copy()
temp['Recovery Rate'] = temp['Cured']/temp['Confirmed']*100
fig = go.Figure()
fig.add_trace(go.Scatter(x=temp['Date'], y=temp['Recovery Rate'],
                    mode='lines+markers',marker_color='green'))
fig.update_layout(title_text = 'Trend of Recovery Rate')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig.show()


# In[26]:


temp = date_wise_data.copy()
temp['Mortality Rate'] = temp['Deaths']/temp['Confirmed']*100
fig = go.Figure()
fig.add_trace(go.Scatter(x=temp['Date'], y=temp['Mortality Rate'],mode='lines+markers',marker_color='red'))
fig.update_layout(title_text = 'Trend of Mortality Rate')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig.show()


# In[27]:


temp = country_wise.sort_values('Active').tail(15).reset_index()
temp = temp.sort_values('Active',ascending=True)
fig = go.Figure(data=[
go.Bar(name='Active', y=temp['Country/Region'], x=temp['Active'], 
       orientation='h',marker_color='#0f5dbd'),
    go.Bar(name='Cured', y=temp['Country/Region'], x=temp['Recovered'], 
       orientation='h',marker_color='#319146'),
go.Bar(name='Death', y=temp['Country/Region'], x=temp['Deaths'], 
       orientation='h',marker_color='#e03216')])
fig.update_layout(barmode='stack',width=600, height=800)
#fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.update_layout(title_text='Active Cases,Cured,Deaths in Top 15 countries',
                  plot_bgcolor='rgb(275, 270, 273)')
fig.show()


# In[28]:


temp = country_wise.sort_values('New cases').tail(15).reset_index()
temp = temp.sort_values('New cases', ascending=False)
state_order = temp['Country/Region']
fig = px.bar(temp,x="New cases", y="Country/Region", color='Country/Region',color_discrete_sequence = ex.colors.cyclical.Edge,
             title=' Top 15 Countries by New Cases', orientation='h', text='New cases')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)')
fig.update_layout(template = 'plotly_white')
fig.show()


# In[29]:


temp = country_wise.sort_values('Confirmed').tail(15).reset_index()
temp = temp.sort_values('Confirmed', ascending=False)
state_order = temp['Country/Region']
fig = px.bar(temp,x="Confirmed", y="Country/Region", color='Country/Region',color_discrete_sequence = ex.colors.cyclical.IceFire,
             title=' Top 15 Countries by Number Confirmed cases', orientation='h', text='Confirmed')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)')
fig.update_layout(template = 'plotly_white')
fig.show()


# In[30]:


temp = country_wise.sort_values('Recovered').tail(15).reset_index()
temp = temp.sort_values('Recovered', ascending=False)
state_order = temp['Country/Region']
fig = px.bar(temp,x="Recovered", y="Country/Region", color='Country/Region',color_discrete_sequence = ex.colors.cyclical.Twilight, 
             title=' Top 15 Countries by Number Recovered cases', orientation='h', text='Recovered')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)')
fig.update_layout(template = 'plotly_white')
fig.show()


# In[31]:


temp = country_wise.sort_values('Deaths').tail(15).reset_index()
temp = temp.sort_values('Deaths', ascending=False)
state_order = temp['Country/Region']
fig = px.bar(temp,x="Deaths", y="Country/Region", color='Country/Region',
             title=' Top 15 Countries by Number Deaths', orientation='h', text='Deaths')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)')
fig.update_layout(template = 'plotly_white')
fig.show()


# In[32]:


fig = px.choropleth(country_wise, locations="Country/Region", 
                    locationmode='country names', color="Confirmed",
                    hover_name="Country/Region", hover_data=['Confirmed','Recovered','Deaths','Active'],
                    color_continuous_scale="peach", 
                    title='Current situation of COVID-19 Worldwide')
fig.update(layout_coloraxis_showscale=False)
fig.show()


# In[33]:


temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths'].max()
temp = temp.reset_index()
temp['Date'] = pd.to_datetime(temp['Date'])
temp['Date'] = temp['Date'].dt.strftime('%m/%d/%Y')
temp['size'] = temp['Confirmed'].pow(0.3)
fig = px.scatter_geo(temp,locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region",
                     range_color= [0, max(temp['Confirmed'])], animation_frame="Date", 
                     title='Spread of COVID-19 all over the world over time',
                     color_continuous_scale=px.colors.diverging.curl)
fig.update(layout_coloraxis_showscale=False)
fig.show()


# In[34]:


from folium.plugins import HeatMap, HeatMapWithTime
m = folium.Map(location=[54,15], zoom_start=2,tiles='cartodbpositron',height = 500,width = '95%')
HeatMap(data=full_table[['Lat', 'Long']], radius=15).add_to(m)
m


# In[35]:


'''
le = preprocessing.LabelEncoder()

train_x.Country = le.fit_transform(train_x.Country)
train_x['State'] = le.fit_transform(train_x['State'])

test_x.Country = le.fit_transform(test_x.Country)
test_x['State'] = le.fit_transform(test_x['State'])

countries = train_x.Country.unique()
'''


# In[36]:


'''
output = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in countries:
    states = train_x.loc[train_x.Country == country, :].State.unique()
    
    for state in states:
        # Train ======================================================
        train_X = train_x.loc[(train_x.Country == country) & (train_x.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]
        
        train_y1 = train_X.loc[:, 'ConfirmedCases']
        train_y2 = train_X.loc[:, 'Fatalities']
        
        train_X = train_X.loc[:, ['State', 'Country', 'Date']]
        
        train_X.Country = le.fit_transform(train_X.Country)
        train_X['State'] = le.fit_transform(train_X['State'])
        
        # Test ======================================================
        
        test_X = test_x.loc[(test_x.Country == country) & (test_x.State == state), ['State', 'Country', 'Date', 'ForecastId']]
        
        test_X_Id = test_X.loc[:, 'ForecastId']
        test_X = test_X.loc[:, ['State', 'Country', 'Date']]
        
        test_X.Country = le.fit_transform(test_X.Country)
        test_X['State'] = le.fit_transform(test_X['State'])
        
        # Data Fitting ======================================================
        xmodel1 = XGBRegressor(n_estimators=1000)
        xmodel1.fit(train_X,train_y1)
        y1_xpred = xmodel1.predict(test_X)
        
        xmodel2 = XGBRegressor(n_estimators=1000)
        xmodel2.fit(train_X, train_y2)
        y2_xpred = xmodel2.predict(test_X)
        
        data = pd.DataFrame({'ForecastId': test_X_Id, 'ConfirmedCases': y1_xpred, 'Fatalities': y2_xpred})
        output = pd.concat([output,data], axis=0)
        
output.ForecastId = output.ForecastId.astype('int')
#output.to_csv('submission.csv', index=False)
#output.head()
'''


# In[37]:


#Creating Pipleline ======================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline_dt = Pipeline([('scaler2' , StandardScaler()),
                        ('RandomForestRegressor: ', RandomForestRegressor())])
pipeline_dt.fit(x_train , y_train)
prediction = pipeline_dt.predict(x_test)

#score
score = pipeline_dt.score(x_test,y_test)
print('Score: ' + str(score))

#Error
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(prediction,y_test)
print('error value: ' + str(val_mae))

#predict
X_test = week5_test.iloc[:,1:6]
predictor = pipeline_dt.predict(X_test)
prediction_list = [x for x in predictor]

#submission
sub = pd.DataFrame({'ForecastId': week5_test.index , 'TargetValue': prediction_list})

p=sub.groupby(['ForecastId'])['TargetValue'].quantile(q=0.05).reset_index()
q=sub.groupby(['ForecastId'])['TargetValue'].quantile(q=0.5).reset_index()
r=sub.groupby(['ForecastId'])['TargetValue'].quantile(q=0.95).reset_index()

p.columns = ['ForecastId' , 'q0.05']
q.columns = ['ForecastId' , 'q0.5']
r.columns = ['ForecastId' , 'q0.95']

p = pd.concat([p,q['q0.5'] , r['q0.95']],1)

p['q0.05']=p['q0.05'].clip(0,10000)
p['q0.05']=p['q0.5'].clip(0,10000)
p['q0.05']=p['q0.95'].clip(0,10000)

p['ForecastId'] =p['ForecastId']+ 1

sub=pd.melt(p, id_vars=['ForecastId'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['ForecastId'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()


# In[38]:


import scipy
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0))) + 1
d_df = date_wise_data.copy()
p0 = (0,0,0)
def plot_logistic_fit_data(d_df, title, p0=p0):
    d_df = d_df.sort_values(by=['Date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['Confirmed']

    x = d_df['x']
    y = d_df['y']

    c2 = scipy.optimize.curve_fit(logistic,  x,  y,  p0=p0 )
    #y = logistic(x, L, k, x0)
    popt, pcov = c2

    x = range(1,d_df.shape[0] + int(popt[2]))
    y_fit = logistic(x, *popt)
    
    p_df = pd.DataFrame()
    p_df['x'] = x
    p_df['y'] = y_fit.astype(int)
    
    print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
    print("Predicted k (growth rate): " + str(float(popt[1])))
    print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")

    x0 = int(popt[2])
    
    traceC = go.Scatter(
        x=d_df['x'], y=d_df['y'],
        name="Confirmed",
        marker=dict(color="Red"),
        mode = "markers+lines",
        text=d_df['Confirmed'],
    )

    traceP = go.Scatter(
        x=p_df['x'], y=p_df['y'],
        name="Predicted",
        marker=dict(color="blue"),
        mode = "lines",
        text=p_df['y'],
    )
    
    trace_x0 = go.Scatter(
        x = [x0, x0], y = [0, p_df.loc[p_df['x']==x0,'y'].values[0]],
        name = "X0 - Inflexion point",
        marker=dict(color="black"),
        mode = "lines",
        text = "X0 - Inflexion point"
    )

    data = [traceC, traceP, trace_x0]

    layout = dict(title = 'Cumulative Conformed cases and logistic curve projection',
          xaxis = dict(title = 'Day since first case', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),
          hovermode = 'closest',plot_bgcolor='rgb(275, 270, 273)'
         )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='covid-logistic-forecast')
    
L = 250000
k = 0.25
x0 = 100
p0 = (L, k, x0)
plot_logistic_fit_data(d_df,'ALL')


# In[39]:


import datetime
import scipy
p0 = (0,0)
def plot_exponential_fit_data(d_df, title, delta, p0):
    d_df = d_df.sort_values(by=['Date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['Confirmed']

    x = d_df['x'][:-delta]
    y = d_df['y'][:-delta]

    c2 = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y,  p0=p0)

    A, B = c2[0]
    print(f'(y = Ae^(Bx)) A: {A}, B: {B}')
    x = range(1,d_df.shape[0] + 1)
    y_fit = A * np.exp(B * x)
    
    traceC = go.Scatter(
        x=d_df['x'][:-delta], y=d_df['y'][:-delta],
        name="Confirmed (included for fit)",
        marker=dict(color="Red"),
        mode = "markers+lines",
        text=d_df['Confirmed'],
    )

    traceV = go.Scatter(
        x=d_df['x'][-delta-1:], y=d_df['y'][-delta-1:],
        name="Confirmed (validation)",
        marker=dict(color="blue"),
        mode = "markers+lines",
        text=d_df['Confirmed'],
    )
    
    traceP = go.Scatter(
        x=np.array(x), y=y_fit,
        name="Projected values (fit curve)",
        marker=dict(color="green"),
        mode = "lines",
        text=y_fit,
    )

    data = [traceC, traceV, traceP]

    layout = dict(title = 'Cumulative Conformed cases and exponential curve projection',
          xaxis = dict(title = 'Day since first case', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),plot_bgcolor='rgb(275, 270, 273)',
          hovermode = 'closest'
         )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='covid-exponential-forecast')
p0 = (40, 0.2)
plot_exponential_fit_data(d_df, 'I', 7, p0)


# In[40]:


cnf = date_wise_data.copy()
Confirmed = cnf[['Date','Confirmed']]
Confirmed = date_wise_data.groupby('Date').sum()['Confirmed'].reset_index()
Confirmed.columns = ['ds','y']
Confirmed['ds'] = pd.to_datetime(Confirmed['ds'])
dth = date_wise_data.copy()
deaths = dth[['Date','Deaths']]
deaths = date_wise_data.groupby('Date').sum()['Deaths'].reset_index()
deaths.columns = ['ds','y']
deaths['ds'] = pd.to_datetime(deaths['ds'])


# In[41]:


m= Prophet(interval_width=0.99)
m.fit(Confirmed)
future = m.make_future_dataframe(periods=14)
future_confirmed = future.copy() # for non-baseline predictions later on
forecast = m.predict(future)
forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# In[42]:


fig = plot_plotly(m,forecast)
fig.update_layout(title_text = 'Confirmed cases Prediction using prophet')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
py.iplot(fig) 


# In[43]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=date_wise_data['Date'], y=date_wise_data['Confirmed'],
                    mode='lines+markers',marker_color='blue',name='Actual'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                    mode='lines+markers',marker_color='Orange',name='Predicted'))
fig.update_layout(title_text = 'Confirmed cases Predicted vs Actual using prophet')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig.show()


# In[44]:


md= Prophet(interval_width=0.99)
md.fit(deaths)
futured = md.make_future_dataframe(periods=14)
future_confirmed = futured.copy()
forecastd = md.predict(futured)
forecastd = forecastd[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# In[45]:


fig = plot_plotly(md, forecastd)
fig.update_layout(title_text = 'Deaths Prediction using prophet')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
py.iplot(fig) 


# In[46]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=date_wise_data['Date'], y=date_wise_data['Deaths'],
                    mode='lines+markers',marker_color='blue',name='Actual'))
fig.add_trace(go.Scatter(x=forecastd['ds'], y=forecastd['yhat_upper'],
                    mode='lines+markers',marker_color='red',name='Predicted'))
fig.update_layout(title_text = 'Deaths Predicted vs Actual using prophet')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig.show()

