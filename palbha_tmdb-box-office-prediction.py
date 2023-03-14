#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
sns.palplot(sns.color_palette())
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from collections import Counter
import os
print(os.listdir("../input"))
# plotly standard imports
import plotly.graph_objs as go
import plotly.plotly as py
# Cufflinks wrapper on plotly
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
cf.go_offline(connected=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
# Set global theme
cf.set_config_file(world_readable=True, theme='pearl')
# Any results you write to the current directory are saved as output.


# In[2]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
        
train = text_to_dict(train)
test = text_to_dict(test)


# In[3]:


train.release_date.head()


# In[4]:


train.head(3)


# In[5]:


#Lets check at the basic information about our data
train.info()


# In[6]:


#Lets check the basic overview of our numeric data 
print("Shape of train data ",train.shape , "\nShape of test data ",test.shape )

train.describe()


# In[7]:


#Lets check how many columns have null value in it

null_columns=train.columns[train.isnull().any()]

train[null_columns].isnull().sum()


# In[8]:


#Lets draw the same 
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
print(missing_data.head(20))
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index.values, y=missing_data["Missing Ratio"])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[9]:


train['belongs_to_collection'][0]


# In[10]:


train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)


# In[11]:


train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)
train['all_genres'] = train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')


# In[12]:


list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)


# In[13]:


# basically counter gives count for each genre most_common() order it from high to low ...

Counter([i for j in list_of_genres for i in j]).most_common() 
#We can see that there are overall 20 genres 
#Lets see if any particular movie is from the most popular genre for this purpose lets take top 10 genre 
top_genre=Counter([i for j in list_of_genres for i in j]).most_common()[:10]
#print(top_genre)
#lets only the genre name
top_genre=[i[0] for i in top_genre]
print(top_genre)
for g in top_genre:
    train['genre_' + g] = train['all_genres'].apply(lambda x: 1 if g in x else 0)
#Similary we will do for other columns as well


# In[14]:


train['num_lang'] = train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)
train['all_lang'] = train['spoken_languages'].apply(lambda x: ' '.join(sorted([i['iso_639_1'] for i in x])) if x != {} else '')


# In[15]:


list_of_lang = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
print(list_of_lang[:10])


# In[16]:



print("Total languages ",len(Counter([i for j in list_of_lang for i in j]).most_common() ))
#here we have approximately 56 langauges so we can consider may be top 20 as popular languages

top_lang=Counter([i for j in list_of_lang for i in j]).most_common()[:20]
#print(top_genre)
#lets only the genre name
top_lang=[i[0] for i in top_lang]
print(top_lang)
for g in top_lang:
    train['lang_' + g] = train['all_lang'].apply(lambda x: 1 if g in x else 0)
#Similary we will do for other columns as well


# In[17]:


train['num_pcount'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)
train['all_pcount'] = train['production_countries'].apply(lambda x: ','.join(sorted([i['name'] for i in x])) if x != {} else '')


# In[18]:


list_of_prod_country = list(train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
print(list_of_prod_country[:10])


# In[19]:


print("Lets check how many Countries are there in totel ",len(Counter([i for j in list_of_prod_country for i in j]).most_common()
                                                                        ))

#Since we have in total 3695 ocmpanies lets check if a particular movies is produced by any of the top 50 companies

top_pcount=Counter([i for j in list_of_prod_country for i in j]).most_common()[:30]
top_pcount=[i[0] for i in top_pcount]
print(top_pcount[:20])

for g in top_pcount:
    train['pcount_' + g] = train['all_pcount'].apply(lambda x: 1 if g in x else 0)


# In[20]:


train['num_pc'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)
train['all_pc'] = train['production_companies'].apply(lambda x: ','.join(sorted([i['name'] for i in x])) if x != {} else '')


# In[21]:


list_of_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
print(list_of_lang[:10])


# In[22]:


train['num_words'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)
train['all_words'] = 
train['Keywords'].apply(lambda x: ','.join(sorted([i['name'] for i in x])) if x != {} else '')


# In[23]:


print("Lets check how many production companies are there in totel ",len(Counter([i for j in list_of_companies for i in j]).most_common()
                                                                        ))

#Since we have in total 3695 ocmpanies lets check if a particular movies is produced by any of the top 50 companies

top_pc=Counter([i for j in list_of_lang for i in j]).most_common()[:50]
top_pc=[i[0] for i in top_pc]
print(top_pc[:20])


# In[24]:


def check_pc(x):
    flag=0
    for k in x.split(","):
        if k in top_pc:
            flag=1
    return flag  

train["pop_pc"]=train["all_pc"].apply(check_pc)


# In[25]:


train["num_cast"]=train['cast'].apply(lambda x: len(x) if x != {} else 0)
train["num_cast_0"]=train["cast"].apply(lambda x: len([1 for i in x if i['gender'] == 0]))
train["num_cast_1"]=train["cast"].apply(lambda x: len([1 for i in x if i['gender'] == 1]))
train["num_cast_2"]=train["cast"].apply(lambda x: len([1 for i in x if i['gender'] == 2]))
train["all_cast"]=train["cast"].apply(lambda x: ','.join(sorted([i['name'] for i in x])) if x != {} else '')


# In[26]:


list_of_cast = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
print(list_of_cast[:1])


# In[27]:


print("Total Cast ",len(Counter([i for j in list_of_cast for i in j]).most_common() ))
#here we have approximately 56 langauges so we can consider may be top 20 as popular languages

top_cast=Counter([i for j in list_of_cast for i in j]).most_common()[:30]
top_cast=[i[0] for i in top_cast]
print(top_cast)
for g in top_cast:
    train['cast_' + g] = train['all_cast'].apply(lambda x: 1 if g in x else 0)
#Similary we will do for other columns as well


# In[28]:


#Lets take a look at what all details are present in the Crew column
train["crew"][0]


# In[29]:


train["num_crew"]=train['crew'].apply(lambda x: len(x) if x != {} else 0)
train["num_crew_0"]=train["crew"].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
train["num_crew_1"]=train["crew"].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
train["num_crew_2"]=train["crew"].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
train["all_crew"]=train["crew"].apply(lambda x: ','.join(sorted([i['name'] for i in x])) if x != {} else '')


# In[30]:


list_of_crew = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
print(list_of_crew[:1])


# In[31]:


print("Total Crew ",len(Counter([i for j in list_of_crew for i in j]).most_common() ))
Counter([i for j in list_of_crew for i in j]).most_common()


# In[32]:


top_crew=Counter([i for j in list_of_crew for i in j]).most_common()[:30]
top_crew=[i[0] for i in top_cast]
for g in top_crew:
    train['crew_' + g] = train['all_crew'].apply(lambda x: 1 if g in x else 0)


# In[33]:


list_of_crew_dept = list(train['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)
print("Total crew dept : ",len(Counter([i for j in list_of_crew_dept for i in j])))
top_crew_dept=Counter([i for j in list_of_crew_dept for i in j]).most_common(10)
top_crew_dept=[i[0] for i in top_crew_dept]
print(top_crew_dept)
for j in top_crew_dept:
    train['dept_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 


# In[34]:


train['release_date']=train['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%y'))
print(train.release_date.head())
train["release_year"]=train.release_date.dt.year
train["month"]=train.release_date.dt.month
train["week_day"]=train.release_date.dt.dayofweek


# In[35]:


print("Null values is Runtime are : ",train["runtime"].isnull().sum())
train["runtime"].fillna(train["runtime"].median(),inplace=True)


# In[36]:


#Lets see how runtime is affected by year and genre as well
train['runtime'].iplot(
    kind='hist',
    bins=30,
    xTitle='Runtime',
    linecolor='black',
    yTitle='count',
    title='Runtime Distribution')


# In[37]:


#Lets see if runtime affects the revenue somehow

#train[['runtime', 'revenue']].iplot(
train.iplot(
    x='runtime',
    y='revenue',
    xTitle='RunTime',
    yTitle='Revenue',
    text='title',
    mode='markers',
    title='Revenue vs RunTime')


# In[38]:


#train[['runtime', 'revenue']].iplot(
train.iplot(
    x='runtime',
    y='popularity',
    xTitle='RunTime',
    yTitle='Popularity',
    text='title',
    mode='markers',
    title='Popularity vs RunTime')


# In[39]:


runtime_mean=train.groupby("release_year")["runtime"].mean()
runtime_mean.iplot(
    kind='bar',
    xTitle='Year',
    yTitle='Average',
    title='Yearly Average Runtime')


# In[40]:





# In[40]:


train['has_homepage'] = 0
train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1
#test['has_homepage'] = 0
#test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1


# In[41]:


print('Number of casted persons in films')
train['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)


# In[42]:


train.columns.values


# In[43]:


#Lets create a copy of the train data and keep it as it is for future purpse
train_copy=train.copy(deep=True)
train.drop("id",axis=1,inplace=True)


# In[44]:


#Lets check the basic correlation plot

#We know that amoung our 23 columns including the target column we only have 4 numeric feature which are not corrlated as such .. 
#Only budget has a correlation with revenue which is something bound to happen and good for our model
plt.figure(figsize=(10,10))
corr = train.corr()
#corr.index = train.columns
sns.heatmap(corr, annot = True, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


# In[45]:


#Lets take a look at budget column
sns.boxplot("budget",data=train)
#Lets check at the actual quantile values 
print("Quantile values \n ",train.budget.quantile([0.25,0.5,.75,.8,.95,.97,.99,.995]))


# In[46]:


#Lets check how manu values have zero in budget
print("Values where budget is zero i.e NA",train[train.budget==0].budget.count())

#Lets see how can we impute these value or handle it 


# In[47]:


train = train.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status'], axis=1)


# In[48]:


train.columns.values


# In[49]:



train_x,test_x,train_y,test_y=train_test_split(train.drop(["revenue"],axis=1),train["revenue"],test_size=0.3)


# In[50]:


train_x.dtypes.unique()


# In[51]:


model=LinearRegression()
model.fit(train_x.select_dtypes(exclude="object"),train_y)


# In[52]:


pred=model.predict(train_x.select_dtypes(exclude="object"))
mean_absolute_error(pred,train_y)

