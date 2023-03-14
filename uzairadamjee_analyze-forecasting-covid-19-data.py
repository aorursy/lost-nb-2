#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pmdarima


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
import plotly.graph_objects as go
from pmdarima import auto_arima    
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


#Changing Date column to datetime
train['Date']= pd.to_datetime(train['Date']) 
test['Date']= pd.to_datetime(test['Date']) 
#set index to date column
new_train = train.set_index(['Date'])
new_test = test.set_index(['Date'])


# In[7]:


new_train.head()


# In[8]:


new_train.isnull().sum()


# In[9]:


new_train[['Province_State']] = new_train[['Province_State']].fillna('')
new_train.isnull().sum()


# In[10]:


#dropping forcast id and id columns
new_test = new_test.drop(["ForecastId"], axis=1)
new_train = new_train.drop(["Id"], axis=1)


# In[11]:


# Creating a dataframe with total no of cases for every country
confirmiedcases = pd.DataFrame(train.groupby('Country_Region')['ConfirmedCases'].sum())
confirmiedcases['Country_Region'] = confirmiedcases.index
confirmiedcases.index = np.arange(1,185)
global_confirmiedcases = confirmiedcases[['Country_Region','ConfirmedCases']]
fig = px.bar(global_confirmiedcases.sort_values('ConfirmedCases',ascending=False)[:40][::-1],
             x='ConfirmedCases',y='Country_Region',title='Worldwide Confirmed Cases',text='ConfirmedCases', height=900, orientation='h')
fig.show()


# In[12]:


# Creating a dataframe with total no of cases for every country
confirmiedcases = pd.DataFrame(new_train.groupby('Country_Region')['Fatalities'].sum())
confirmiedcases['Country_Region'] = confirmiedcases.index
confirmiedcases.index = np.arange(1,185)
global_confirmiedcases = confirmiedcases[['Country_Region','Fatalities']]
fig = px.bar(global_confirmiedcases.sort_values('Fatalities',ascending=False)[:40][::-1],
             x='Fatalities',y='Country_Region',title='Worldwide Deaths',text='Fatalities', height=900, orientation='h')
fig.show()


# In[13]:


formated_gdf = train.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['ConfirmedCases'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="Country_Region", locationmode='country names', 
                     color="ConfirmedCases", size='size', hover_name="Country_Region", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="Date", 
                     title='CORONA: Spread Over Time From Jan 2020 to Apr 2020', color_continuous_scale="portland")
fig.show()


# In[14]:


formated_gdf = train.groupby(['Date', 'Country_Region'])['Fatalities'].sum()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Fatalities'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="Country_Region", locationmode='country names', 
                     color="Fatalities", size='size', hover_name="Country_Region", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="Date", 
                     title='CORONA: Spread Over Time From Jan 2020 to Apr 2020', color_continuous_scale="portland")
fig.show()


# In[15]:


new_train.columns


# In[16]:


countries = new_train['Country_Region'].unique()
for country in countries:
    if country == 'Turkey':
        train_df = new_train[new_train['Country_Region'] == country]
        test_df = new_test[new_test['Country_Region'] == country]

        #********* Farecasting ConfirmedCases ********

        X_train_conf = train_df['ConfirmedCases'].values
        p,d,q = auto_arima(X_train_conf).order
        
        #For trying out ARIMA
        #ARIMA(X_train_conf,order=(p,d,q))

        model_conf = SARIMAX(X_train_conf,order=(p,d,q),seasonal_order=(0,0,0,0))
        result_conf = model_conf.fit()
        fcast_conf = result_conf.predict(len(X_train_conf)-13,len(X_train_conf)+len(test_df)-14,typ='levels')
        test.loc[test['Country_Region']==country,'ConfirmedCases'] = np.rint(fcast_conf)
       
        
        #********* Farecasting Fatalities ********
        

        X_train_fat = train_df['Fatalities'].values
        p,d,q = auto_arima(X_train_fat).order
        model_fat = SARIMAX(X_train_fat,order=(p,d,q),seasonal_order=(0,0,0,0))
        result_fat = model_fat.fit()
        fcast_fat = result_fat.predict(len(X_train_fat)-13,len(X_train_fat)+len(test_df)-14,typ='levels')

        test.loc[test['Country_Region']==country,'Fatalities'] = np.rint(fcast_fat)
        


# In[17]:


#test.loc[test['Country_Region']=='Pakistan']


# In[18]:


turkey_data = test.loc[test['Country_Region']=='Turkey']
turkey_data.columns


# In[19]:


plot_turkey_data = turkey_data.filter(["Date","ConfirmedCases", "Fatalities"])
plot_turkey_data.head()


# In[20]:



fig = go.Figure(go.Scatter(x=plot_turkey_data['Date'],y=plot_turkey_data['ConfirmedCases'],
                      text='Total Confirmed Cases'))
fig.update_layout(title_text='Total Number of Coronavirus Cases by Date')
fig.update_yaxes(showticklabels=False)

fig.show()


# In[21]:



fig = go.Figure(go.Scatter(x=plot_turkey_data['Date'],y=plot_turkey_data['Fatalities'],
                      text='Total Confirmed Cases'))
fig.update_layout(title_text='Total Number Fatalities of Coronavirus by Date')
fig.update_yaxes(showticklabels=False)

fig.show()


# In[22]:


# download the latest data sets
global_confirmed_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
global_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
global_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')


# In[23]:


global_confirmed_cases.head()


# In[24]:


global_deaths.head()


# In[25]:


global_recovered.head()


# In[26]:


dates = global_confirmed_cases.columns[4:]


# In[27]:


cc_df = global_confirmed_cases.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Confirmed')
print(cc_df.head())


# In[28]:


# create complete data

cc_df = global_confirmed_cases.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Confirmed')


deaths_df = global_deaths.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Deaths')

recv_df = global_recovered.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Recovered')

print(cc_df.shape)
print(deaths_df.shape)
print(recv_df.shape)

complete_data = pd.merge(left=cc_df, right=deaths_df, how='left',
                      on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])
complete_data = pd.merge(left=complete_data, right=recv_df, how='left',
                      on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])

complete_data.head()


# In[29]:


# Active cases 
complete_data['Active'] = complete_data['Confirmed'] - complete_data['Recovered'] - complete_data['Deaths']


# In[30]:


#check for null/nan values

complete_data.isna().sum()


# In[31]:



complete_data['Recovered'] = complete_data['Recovered'].fillna(0)
complete_data['Recovered'] = complete_data['Recovered'].astype('int')
complete_data['Active'] = complete_data['Active'].fillna(0)
complete_data['Active'] = complete_data['Active'].astype('int')
complete_data.isna().sum()


# In[32]:


complete_data = complete_data.rename(columns={"Province/State":"State","Country/Region": "Country"})


# In[33]:


complete_data.loc[complete_data['Country'] == "US", "Country"] = "USA"

complete_data.loc[complete_data['Country'] == 'Korea, South', "Country"] = 'South Korea'

complete_data.loc[complete_data['Country'] == 'Taiwan*', "Country"] = 'Taiwan'

complete_data.loc[complete_data['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'

complete_data.loc[complete_data['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"

complete_data.loc[complete_data['Country'] == "Reunion", "Country"] = "Réunion"

complete_data.loc[complete_data['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'

complete_data.loc[complete_data['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'

complete_data.loc[complete_data['Country'] == 'Gambia, The', "Country"] = 'Gambia'


# In[34]:


df_date = complete_data.filter(["Date",  "Confirmed", "Deaths", "Recovered"])
df_date = df_date.groupby(df_date["Date"]).sum()
df_date.head()


# In[35]:


plt.figure(figsize=(15,6))
plt.plot(df_date, marker='o')
plt.title('Total Number of Coronavirus Cases by Date')
plt.legend(df1_date.columns)
plt.xticks(rotation=75)
plt.show()


# In[36]:


countries_grouped = complete_data.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
temp = countries_grouped[['Country', 'Deaths']]
temp = temp.sort_values(by='Deaths', ascending=False)
temp = temp.reset_index(drop=True)
temp = temp[temp['Deaths']>0]
temp.style.background_gradient(cmap='Pastel1_r')


# In[37]:


countries = complete_data['Country'].unique()
for country in countries:
    if(country == 'Turkey'):

        train_df = complete_data[complete_data['Country'] == country]
        data = train_df.Recovered.astype('int32').tolist()
        
        # fit model
        p,d,q = auto_arima(data).order
        model = SARIMAX(data, order=(p,d,q), seasonal_order=(0,0,0,0),measurement_error=True)#seasonal_order=(1, 1, 1, 1))
        model_fit = model.fit(disp=False)
        
        # make prediction
        predicted = model_fit.predict(len(data), len(data)+13)
       
        print(predicted)
       


# In[ ]:




