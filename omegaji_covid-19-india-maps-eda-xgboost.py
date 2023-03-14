#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


train_df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")


# In[3]:


train_df[train_df["Country_Region"]=="India"]


# In[4]:



india_df=train_df[train_df["Country_Region"]=="India"]

#india_df["day"]=india_df["Date"].apply(lambda x:int(x[-2:]) )
#india_df["Month"]=india_df["Date"].apply(months )
india_df["ConfirmedCases"]=india_df["ConfirmedCases"].apply(lambda x: int(x))


# In[5]:


from datetime import datetime
india_df["Date"]=india_df['Date'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

    
    


# In[ ]:





# In[6]:




india_df["week"]="week_"+ str(india_df["Date"].dt.week)


# In[7]:


india_df["week"]=india_df["Date"].dt.week.apply(lambda x: x)


# In[8]:



import matplotlib.pyplot as plt
import seaborn as sns


#fig.set_size_inches(12, 18)
sns.catplot(data=india_df.groupby(["week"]).max().reset_index(),x="week",y="ConfirmedCases",kind="bar")


# In[9]:


train_df["day"]=train_df["Date"].apply(lambda x:int(x[-2:]) )
#train_df["Month"]=train_df["Date"].apply(months )
train_df["ConfirmedCases"]=train_df["ConfirmedCases"].apply(lambda x: int(x))


# In[10]:


sns.catplot(data=india_df.groupby(["week"]).max().reset_index(),x="week",y="Fatalities",kind="bar")


# In[11]:


train_df=train_df.drop("Province_State",axis=1)
from datetime import datetime
from datetime import datetime
train_df["Date"]=train_df['Date'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

train_df["week"]="week_"+ str(train_df["Date"].dt.week)
train_df["week"]=train_df["Date"].dt.week.apply(lambda x: x)
train_df["day"]=train_df["Date"].dt.day.apply(lambda x: x)
train_df["month"]=train_df["Date"].dt.month.apply(lambda x: int(x))


# In[12]:


train_df=train_df.drop("Date",axis=1)


# In[13]:



import numpy as np 
import pandas as pd 
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

country_df=train_df.groupby(['Country_Region', 'week']).max().reset_index().sort_values('week', ascending=False)

country_df = country_df.drop_duplicates(subset = ['Country_Region'])
country_df = country_df[country_df['ConfirmedCases']>0]

data = dict(type='choropleth',
locations = country_df['Country_Region'],
locationmode = 'country names', z = country_df['ConfirmedCases'],
text = country_df['Country_Region'], colorbar = {'title':'CONFIRMED CASES'},
colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],    
reversescale = False
           )
layout = dict(title='COVID-19 CASES AROUND THE WORLD',
geo = dict(showframe = True, projection={'type':'mercator'}))
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap, validate=False)


# In[14]:


df_countrydate = train_df[train_df['ConfirmedCases']>0]
df_countrydate = df_countrydate.groupby(['week','Country_Region']).max().reset_index()
df_countrydate

fig = px.choropleth(df_countrydate, 
                    locations="Country_Region", 
                    locationmode = "country names",
                    color="ConfirmedCases", 
                    hover_name="Country_Region", 
                    animation_frame="week",
                   color_continuous_scale="Greens"
                   )
fig.update_layout(
    title_text = 'Global Spread of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()


# In[15]:


df_countrydate[df_countrydate["Country_Region"]=="India"]


# In[16]:


df_india=pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

df_india["Date"]=df_india['Date'].apply(lambda x:datetime.strptime(x, '%d/%m/%y'))

df_india["week"]="week_"+ str(df_india["Date"].dt.week)
df_india["week"]=df_india["Date"].dt.week.apply(lambda x: x)
df_india.head()
df_india_grouped=df_india.groupby(["State/UnionTerritory","week"]).max().reset_index().sort_values("week",ascending=False)
df_india_grouped=df_india_grouped.drop_duplicates(subset=["State/UnionTerritory"])


# In[17]:


df_india_grouped
fig = px.scatter(df_india_grouped, x="Confirmed", y="State/UnionTerritory", 
                 title="COVID CASES CONFIRMED IN INDIAN STATES",
                 labels={"COVID CASES CONFIRMED IN INDIAN STATES"} # customize axis label
                )

fig = px.bar(df_india_grouped, x='Confirmed', y='State/UnionTerritory',
             hover_data=['Confirmed', 'State/UnionTerritory'], color='Confirmed', orientation='h',
             text="Confirmed", height=1400)
fig.update_traces( textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='show')
fig.show()


# In[18]:


df_india_grouped["pending"]=df_india_grouped["Confirmed"]-df_india_grouped["Deaths"]-df_india_grouped["Cured"]


# In[19]:


'''df_india_grouped
labels=df_india_grouped["State/UnionTerritory"]
values=df_india_grouped["Confirmed"]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.show()'''


l=list(df_india_grouped["State/UnionTerritory"])
fig = make_subplots(rows=11, cols=3,subplot_titles=l,specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
a=1
b=1

for i in l:
    
    
    temp_df=df_india_grouped[df_india_grouped["State/UnionTerritory"]==i]
    #print(int(temp_df["Deaths"]))
    values=[int(temp_df["Deaths"]),int(temp_df["Cured"]),int(temp_df["pending"])]
    labels=["Deaths","Cured","pending"]
 
    #annot.append(dict(text=i,font_size=10, showarrow=False))
    
    fig.add_trace(go.Pie(labels=labels, textposition="inside",values=values, name=i),a, b)
    
    if b==3 and a<11:
        a=a+1
   
      
    if b+1>3:
        b=1
    else:
        b=b+1
   
    fig.update_traces(hole=.4)

fig.update_layout(
    
    height=1900,width=1000
)
fig.update(layout_title_text='StateWise analysis of Positive cases')


#fig = go.Figure(fig)
fig.show()
#iplot(fig)   


# In[20]:


testing_df=pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv")


# In[21]:


testing_df["Date"]=testing_df['Date'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

testing_df["week"]="week_"+ str(testing_df["Date"].dt.week)
testing_df["week"]=testing_df["Date"].dt.week.apply(lambda x: x)
testing_df.head()
testing_df_grouped=testing_df.groupby(["State","week"]).max().reset_index().sort_values("week",ascending=False)


# In[22]:


testing_df_grouped=testing_df_grouped.drop_duplicates(subset=["State"])


# In[23]:


states=list(testing_df_grouped["State"])

fig = go.Figure(data=[
    
    go.Bar(name='Negative', x=states, y=list(testing_df_grouped["Negative"])),
    go.Bar(name='Positive', x=states, y=list(testing_df_grouped["Positive"])),
])

fig.update_layout(barmode='stack')
fig.show()


# In[24]:


import geopandas as gpd
shapefile="/kaggle/input/india-shape/ind_shape/IND_adm1.shp"
gdf=gpd.read_file(shapefile)[["NAME_1","geometry"]]

gdf.columns = ['states','geometry']
gdf.loc[31,"states"]="Telengana"
gdf.loc[34,"states"]="Uttarakhand"
gdf.loc[25,"states"]="Odisha"

#gdf[gdf["states"]=="Orissa"]


# In[ ]:





# In[25]:


merged_grouped = gdf.merge(df_india_grouped, left_on = 'states', right_on = 'State/UnionTerritory').drop(["Date"],axis=1)
import json
merged_json_grouped = json.loads(merged_grouped.to_json())
json_data_grouped = json.dumps(merged_json_grouped)


# In[ ]:





# In[26]:


from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar,LabelSet
from bokeh.palettes import brewer
from bokeh.models import Slider, HoverTool
geosource = GeoJSONDataSource(geojson = json_data_grouped)
palette = brewer['YlGnBu'][8]
palette = palette[::-1]
color_mapper = LinearColorMapper(palette = palette, low = 0, high = max(merged_grouped["Confirmed"]))

tick_labels = {'0': '0', '100': '100', '200':'200', '400':'400', '800':'800', '1200':'1200', '1400':'1400','1800':'1800', '2000': '2000'}
hover = HoverTool(tooltips = [ ('states','@states'),('Confirmed_Cases', '@Confirmed')])
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)

p = figure(title = 'CoronaVirus Confirmed States(HOVER MOUSE FOR INFO)', plot_height = 600 , plot_width = 950, toolbar_location = None,tools=[hover])
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None


p.patches('xs','ys', source = geosource,fill_color = {'field' :'Confirmed', 'transform' : color_mapper},name="states",
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
labels = LabelSet(x='xs', y='ys', text='states',
              x_offset=5, y_offset=5, source=geosource)

p.add_layout(color_bar, 'below')
output_notebook()
#Display figure.
show(p)


# In[27]:


country_df=df_india.groupby(["week","State/UnionTerritory"]).max().reset_index()
country_df.drop(["Date","ConfirmedIndianNational","ConfirmedForeignNational","Deaths","Cured","Time"],axis=1,inplace=True)


# In[28]:



shapefile="/kaggle/input/india-shape/ind_shape/IND_adm1.shx"
gdf=gpd.read_file(shapefile)[["NAME_1","geometry"]]
gdf["geometry"]=gdf["geometry"].simplify(0.02, preserve_topology=True)
gdf
gdf.columns = ['states','geometry']
gdf.loc[31,"states"]="Telengana"
gdf.loc[34,"states"]="Uttarakhand"
gdf.loc[25,"states"]="Odisha"
merged_grouped = gdf#.merge(df_india_grouped[["State/UnionTerritory","geo"]], left_on = 'states', right_on = 'State/UnionTerritory')

merged_json_grouped = json.loads(merged_grouped.to_json())
json_data_grouped = json.dumps(merged_json_grouped)
for i in merged_json_grouped["features"]:
    i["id"]=i["properties"]["states"]


# In[29]:


country_df=df_india.groupby(["State/UnionTerritory","week"]).max().reset_index().sort_values("week",ascending=True)

country_df=country_df.drop(['Sno', 'Date', 'Time',
       'ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured',
       'Deaths',],axis=1)


# In[30]:


fig = px.choropleth(country_df, geojson=merged_json_grouped,
                    locations="State/UnionTerritory", 
                    
                    color="Confirmed", 
                    hover_name="State/UnionTerritory", 
                    animation_frame="week",
                   color_continuous_scale=["yellow","orange","red"],
                     labels={'Confirmed':'Confirmed'}
                    
                
                         
                      
                   )
fig.update_geos(fitbounds="locations", visible=False,projection_type="natural earth")   
fig.update_layout(
    title_text = 'India Spread of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
 
fig.show()



# In[31]:


newtestdf=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
newtestdf


# In[32]:


train_df.head()


# In[33]:


train_df.columns


# In[34]:


from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
train_df['ConfirmedCases'] = train_df['ConfirmedCases'].apply(int)
train_df['Fatalities'] = train_df['Fatalities'].apply(int)
cases = train_df.ConfirmedCases
fatal=train_df.Fatalities

lb = LabelEncoder()

del train_df["Fatalities"]
del train_df["ConfirmedCases"]
#del train_df["Id"]
train_df['Country_Region'] = lb.fit_transform(train_df['Country_Region'])
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df.drop(["Id","week"],axis=1).values)


# In[35]:


from xgboost import XGBRegressor
model = XGBRegressor(n_estimators = 2500 , random_state = 0 , max_depth = 27)
model.fit(X_train,cases)


# In[36]:


model.score(X_train,cases)


# In[ ]:





# In[37]:


newtestdf["Date"]=newtestdf['Date'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

newtestdf["week"]="week_"+ str(newtestdf["Date"].dt.week)
newtestdf["week"]=newtestdf["Date"].dt.week.apply(lambda x: x)
newtestdf["day"]=newtestdf["Date"].dt.day.apply(lambda x: x)
newtestdf["month"]=newtestdf["Date"].dt.month.apply(lambda x: int(x))
newtestdf['Country_Region'] = lb.fit_transform(newtestdf['Country_Region'])
newtestdf=newtestdf.drop(["Province_State","Date"],axis=1)
newtestdf


# In[ ]:





# In[38]:


X_test = scaler.fit_transform(newtestdf.drop(["ForecastId","week"],axis=1).values)
cases_pred = model.predict(X_test)


# In[39]:


cases_pred = np.around(cases_pred,decimals = 0)
x_train_cas = []
for i in range(len(X_train)):
    x = list(X_train[i])
    x.append(cases[i])
    x_train_cas.append(x)


# In[40]:


x_train_cas = np.array(x_train_cas)
model = XGBRegressor(n_estimators = 2500 , random_state = 0 , max_depth = 27)
model.fit(x_train_cas,fatal)


# In[41]:


x_test_cas = []
for i in range(len(X_test)):
    x = list(X_test[i])
    x.append(cases_pred[i])
    x_test_cas.append(x)
x_test_cas[0]


# In[42]:


x_test_cas = np.array(x_test_cas)
fatalities_pred =model.predict(x_test_cas)
fatalities_pred = np.around(fatalities_pred,decimals = 0)


# In[43]:



submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
submission['ConfirmedCases'] = cases_pred
submission['Fatalities'] = fatalities_pred
submission.to_csv("submission.csv" , index = False)


# In[44]:


THANK YOU!!!!


# In[ ]:





# In[ ]:




