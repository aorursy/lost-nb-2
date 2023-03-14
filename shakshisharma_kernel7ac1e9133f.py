#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libararies
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
import collections
from datetime import timedelta
from datetime import datetime 
import scipy.stats as stats

import pycountry
import plotly
import plotly.io as pio
import plotly.express as px

from ipywidgets import interact
import statsmodels.api as sm


# In[2]:


#  Read datasets
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")


# In[3]:


#We are using weather data provided on Kaggle
weather=pd.read_csv("../input/weather-data/training_data_with_weather_info_week_4.csv")


# In[4]:


#We are using Tanu's dataset of population based on webscraping
population=pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")


# In[5]:


# Select required columns and rename few of them
population = population[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]
population.columns = ['Country (or dependency)', 'Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']


# In[6]:


# Replace United States by US
population.loc[population['Country (or dependency)']=='United States', 'Country (or dependency)'] = 'US'


# In[7]:


# Handling Urban Pop values
population['Urban Pop'] = population['Urban Pop'].str.rstrip('%')
p=population.loc[population['Urban Pop']!='N.A.', 'Urban Pop'].median()
population.loc[population['Urban Pop']=='N.A.', 'Urban Pop']= int(p)
population['Urban Pop'] = population['Urban Pop'].astype('int64')


# In[8]:


# Handling Med Age values
population.loc[population['Med Age']=='N.A.', 'Med Age'] = int(population.loc[population['Med Age']!='N.A.', 'Med Age'].mode()[0])
population['Med Age'] = population['Med Age'].astype('int64')


# In[9]:


train.head()


# In[10]:


print("Combined dataset")
corona_data = weather.merge(population, left_on='Country_Region', right_on='Country (or dependency)', how='left')
corona_data.shape


# In[11]:


#checking for null values
sns.heatmap(corona_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[12]:


#Drop Province/State 
corona_data.drop('Province_State', axis=1, inplace=True)


# In[13]:


#Drop Country or dependency
corona_data.drop('Country (or dependency)', axis=1, inplace=True)


# In[14]:


#checking for null values
sns.heatmap(corona_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[15]:


corona_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']] = corona_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']].fillna(0)


# In[16]:


#checking for null values
sns.heatmap(corona_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[17]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder() 
corona_data.iloc[:, 1] = labelencoder_X.fit_transform(corona_data.iloc[:, 1])


# In[18]:


corona_data['day']=pd.DatetimeIndex(corona_data['Date']).day
corona_data['year'] = pd.DatetimeIndex(corona_data['Date']).year
corona_data['month'] = pd.DatetimeIndex(corona_data['Date']).month
corona_data.head()


# In[19]:


corona_data['Population (2020)'] = corona_data['Population (2020)'].astype(int)


# In[20]:


# Manipulating the original dataframe
#train = pd.read_csv("train.csv")
countrydate_evolution = train[train['ConfirmedCases']>0]
countrydate_evolution = countrydate_evolution.groupby(['Date','Country_Region']).sum().reset_index()

# Creating the visualization
fig = px.choropleth(countrydate_evolution, locations="Country_Region", locationmode = "country names", color="ConfirmedCases", 
                    hover_name="Country_Region", animation_frame="Date", 
                   )

fig.update_layout(
    title_text = 'Global Spread of Coronavirus',
    title_x = 0.5,
    autosize=True,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()


# In[21]:


train_df=train
train_df.rename(columns={"Country_Region": "country", "Province_State": "province"}, inplace=True, errors="raise")
df = train_df.fillna('NA').groupby(['country','province','Date'])['ConfirmedCases','Fatalities'].sum()                           .groupby(['country','province']).max().sort_values(by='ConfirmedCases')                           .groupby(['country']).sum().sort_values(by='ConfirmedCases',ascending = False)

df = pd.DataFrame(df).reset_index()


df = pd.DataFrame(df)

df_new_cases = pd.DataFrame(train_df.fillna('NA').groupby(['country','Date'])['ConfirmedCases'].sum()                             .reset_index()).sort_values(['country','Date'])
df_new_cases.ConfirmedCases = df_new_cases.ConfirmedCases.diff().fillna(0)
df_new_cases = df_new_cases.loc[df_new_cases['Date'] == max(df_new_cases['Date']),['country','ConfirmedCases']]
df_new_cases.rename(columns={"ConfirmedCases": "NewCases"}, inplace=True, errors="raise")

df_new_deaths = pd.DataFrame(train_df.fillna('NA').groupby(['country','Date'])['Fatalities'].sum()                             .reset_index()).sort_values(['country','Date'])

df_new_deaths.Fatalities = df_new_deaths.Fatalities.diff().fillna(0)
df_new_deaths = df_new_deaths.loc[df_new_deaths['Date'] == max(df_new_deaths['Date']),['country','Fatalities']]

df_new_deaths.rename(columns={"Fatalities": "NewFatalities"}, inplace=True, errors="raise")

merged = df.merge(df_new_cases, left_on='country', right_on='country')            .merge(df_new_deaths, left_on='country', right_on='country')


merged.style.background_gradient(cmap="Blues", subset=['ConfirmedCases'])            .background_gradient(cmap="Reds", subset=['Fatalities'])            .background_gradient(cmap="Blues", subset=['NewCases'])            .background_gradient(cmap="Reds", subset=['NewFatalities'])


# In[22]:


import plotly.express as px
#df = px.data.gapminder()
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
#train_df.rename(columns={"Country_Region": "Country"}, inplace=True, errors="raise")
fig = px.scatter(train, x="ConfirmedCases", y="Fatalities",   
                 color="Country_Region",
                 hover_name="Province_State", log_x=True, size_max=60)
fig.update_layout(title_text='Confirmed COVID-19 cases vs Fatalities by country')
fig.show()


# In[23]:


df_by_date = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'India') &(df_by_date.Date >= '2020-03-01')].sort_values('ConfirmedCases',ascending = False), 
             x='Date', y='ConfirmedCases', color="ConfirmedCases", color_continuous_scale=px.colors.sequential.BuGn)
fig.update_layout(title_text='Confirmed COVID-19 cases per day in India')
fig.show()


# In[24]:


df_by_date = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'China') &(df_by_date.Date >= '2020-01-01')].sort_values('ConfirmedCases',ascending = False), 
             x='Date', y='ConfirmedCases', color="ConfirmedCases", color_continuous_scale=px.colors.sequential.BuGn)
fig.update_layout(title_text='Confirmed COVID-19 cases per day in China')
fig.show()


# In[25]:


df_by_date = pd.DataFrame(train.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'US') &(df_by_date.Date >= '2020-03-01')].sort_values('ConfirmedCases',ascending = False), 
             x='Date', y='ConfirmedCases', color="ConfirmedCases", color_continuous_scale=px.colors.sequential.BuGn)
fig.update_layout(title_text='Confirmed COVID-19 cases per day in US')
fig.show()


# In[26]:


# Interactive time series plot of fatalities
fig = px.line(train, x='Date', y='Fatalities', color="Country_Region", hover_name="Country_Region")
fig.update_layout(autosize=False,width=1000,height=500,title='Deaths Over Time for Each Country')
fig.show()


# In[27]:


corona_data['Active'] = corona_data['ConfirmedCases'] - corona_data['Fatalities'] 
 
group_data = corona_data.groupby(["Country_Region"])["Fatalities", "ConfirmedCases"].sum().reset_index()
group_data = group_data.sort_values(by='Fatalities', ascending=False)
group_data = group_data[group_data['Fatalities']>100]
plt.figure(figsize=(15, 5))
plt.plot(group_data['Country_Region'], group_data['Fatalities'],color='red')
plt.plot(group_data['Country_Region'], group_data['ConfirmedCases'],color='green')

 
plt.title('Total Deaths(>100), Confirmed Cases by Country')
plt.show()


# In[28]:


import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
 
grouped = corona_data.groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()
fig = px.line(grouped, x="Date", y="ConfirmedCases",
             title="Worldwide Confirmed Novel Coronavirus(COVID-19) Cases Over Date")
fig.show()


# In[29]:


train_df = pd.read_csv('../input/weather-data/training_data_with_weather_info_week_4.csv', parse_dates=['Date'])
train_df_conf = train_df[train_df["ConfirmedCases"]>=1]
train_df_conf=train_df_conf[["Country_Region","Date"]]
df= train_df_conf.groupby(["Country_Region"]).count()
df=df.sort_values("Date",ascending=False)
country_name = df.index.get_level_values('Country_Region')
corona_victims=[]
for i in range(len(df)):
    corona_victims.append(df["Date"][i])
cl = pd.DataFrame(corona_victims,columns=["Victim"]) # Converting List to Dateframe
df=df.head(80)
xlocs=np.arange(len(df))
df.plot.barh(color=[np.where(cl["Victim"]>20,"r","y")],figsize=(12,16))
plt.xlabel("Number of Confirmed Cases of Corona Virus",fontsize=12,fontweight="bold")
plt.ylabel("Country_Region",fontsize=12,fontweight="bold")
plt.title("No. of confirmed Corona Virus cases by country ",fontsize=14,fontweight="bold")
for i, v in enumerate(df["Date"][:]):
    plt.text(v+0.01,xlocs[i]-0.25,str(v))
plt.legend(country_name) # top affected country
plt.show()


# In[30]:


df_new = []
number_countries = 0
total_victims=0
for i in range(df["Date"].shape[0]):
    if df["Date"][i] > 100:
        df_new.append(df["Date"][i])
        total_victims = total_victims + df["Date"][i]
        number_countries=number_countries+1
print("Number of countries where Corona Victims are more than 100 :", number_countries,"\n")
print("Total Number of Victims:",total_victims,"\n")        
explode=np.zeros(number_countries)
explode[0]=0.1
explode[1]=0.1
explode[2]=0.2
fig = plt.gcf() # gcf stands for Get Current Figure
fig.set_size_inches(10,10)
plt.pie(df_new,explode=explode,autopct='%1.1f%%',shadow=True, labels=country_name[0:number_countries])
title = "Top"+str(number_countries) +" Countries by Confirmed Cases and their Contribution" 
plt.title(title,fontsize=12, fontweight="bold")
plt.legend(loc="lower right",bbox_to_anchor=(1.1,0),bbox_transform=plt.gcf().transFigure) # bbx required to place legend without overlapping
plt.show()


# In[31]:


corona_data.corr()['ConfirmedCases']


# In[32]:


#Attributes showing high correlation with dependent variables are not included
X_train=corona_data[['day','month','Population (2020)','Land Area','Med Age']]


# In[33]:


y_train=corona_data[['ConfirmedCases','Fatalities']]


# In[34]:


sns.heatmap(X_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[35]:


test_data = test.merge(population, left_on='Country_Region', right_on='Country (or dependency)', how='left')
test_data.shape


# In[36]:


test_data['day']=pd.DatetimeIndex(test_data['Date']).day
test_data['year'] = pd.DatetimeIndex(test_data['Date']).year
test_data['month'] = pd.DatetimeIndex(test_data['Date']).month
test_data.head()
test_data.drop('Province_State',axis=1,inplace=True)


# In[37]:


X_test=test_data[['day','month','Population (2020)','Land Area','Med Age']]


# In[38]:


X_test[['Population (2020)', 'Land Area', 'Med Age']] = X_test[['Population (2020)', 'Land Area', 'Med Age']].fillna(0)


# In[39]:


X_test.info()


# In[40]:


# Fitting XG Boost Regression to the dataset

from sklearn.multioutput import MultiOutputRegressor
#XGBoost Regressor
import xgboost as xgb
reg = xgb.XGBRegressor(n_estimators=100)


# In[41]:


regr_multirf = MultiOutputRegressor(xgb.XGBRegressor())

regr_multirf.fit(X_train, y_train)
reg_y_pred = regr_multirf.predict(X_test)


y_pred = np.round(reg_y_pred, 1)
    
y_pred = y_pred.astype(int)

#score=reg_y_pred.score(X_test, y_pred)
#print(score)


# In[42]:


#for multi-output
regr_multirf = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100))
regr_multirf.fit(X_train, y_train)


# In[43]:


y_multirf = regr_multirf.predict(X_test)
y_pred = np.round(y_multirf, 1)
y_multirf.shape


# In[44]:


y_pred = y_pred.astype(int)


# In[45]:


submission = pd.DataFrame(data = np.zeros((y_pred.shape[0],3)), columns = ['ForecastId', 'ConfirmedCases', 'Fatalities'])
submission.shape
y_pred1 = pd.DataFrame(y_pred)


# In[46]:


for i in range(0, len(submission)):
    submission.loc[i,'ForecastId'] = i + 1
    submission.loc[i,'ConfirmedCases'] = y_pred1.iloc[i, 0]
    submission.loc[i,'Fatalities'] = y_pred1.iloc[i, 1]


# In[47]:


submission['ForecastId'] = submission['ForecastId'].astype(int)
submission['ConfirmedCases'] = submission['ConfirmedCases'].astype(int)
submission['Fatalities'] = submission['Fatalities'].astype(int)


# In[48]:


submission


# In[49]:


submission.to_csv('submission.csv', index = False)


# In[50]:


submission.head()

