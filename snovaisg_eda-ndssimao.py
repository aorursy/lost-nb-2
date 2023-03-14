#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast
import operator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[3]:


train.head(2)


# In[4]:


train.shape


# In[5]:


dict_columns = ["belongs_to_collection","genres","production_companies",                 "production_countries","spoken_languages","Keywords","cast","crew"]


# In[6]:


def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df


# In[7]:


train = text_to_dict(train)
test = text_to_dict(test)


# In[8]:


train.belongs_to_collection.apply(lambda x: 1 if len(x) > 0 else 0).value_counts()


# In[9]:


train.belongs_to_collection.apply(lambda x: len(x)).value_counts()


# In[10]:


train.belongs_to_collection.iloc[0]


# In[11]:


movies_per_collection = {}
for x in train.belongs_to_collection:
    if x == {}:
        continue
    name = x[0]["name"]
    if name not in movies_per_collection:
        movies_per_collection[name] = 1
    else:
        movies_per_collection[name] += 1


# In[12]:


sorted_x = sorted(movies_per_collection.items(), key=operator.itemgetter(1),reverse=True)
sorted_x


# In[13]:


train["CollectionName"] = train.belongs_to_collection.apply(lambda x: x[0]["name"] if x != {} else 0)
train["hasCollection"] = train.belongs_to_collection.apply(lambda x: len(x) if x!= {} else 0)

test["CollectionName"] = test.belongs_to_collection.apply(lambda x: x[0]["name"] if x != {} else 0)
test["hasCollection"] = test.belongs_to_collection.apply(lambda x: len(x) if x!= {} else 0)

train = train.drop("belongs_to_collection", axis=1)
test = test.drop("belongs_to_collection", axis=1)

train.loc[:,["CollectionName","hasCollection"]].head(3)


# In[14]:


train.genres.iloc[0]


# In[15]:


train.genres.iloc[1]


# In[16]:


list_of_genres = list(train.genres.apply(lambda x: [genre["name"] for genre in x] if x != {} else []))
list_of_genres = [i for j in list_of_genres for i in j]
list_of_genres


# In[17]:


# Generate a word cloud image
wordcloud = WordCloud(background_color="white",collocations=False,
                      width=1200, height=1000).generate(" ".join(list_of_genres))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[18]:


Counter(list_of_genres).most_common(10)


# In[19]:


train["nGenres"] = train.genres.apply(lambda x: len(x) if x != {} else 0)
test["nGenres"] = test.genres.apply(lambda x: len(x) if x != {} else 0)

unique_genres = set(list_of_genres)
for g in unique_genres:
    train[g] = train.genres.apply(lambda x: 1 if g in [genre["name"] for genre in x] else 0)
    test[g] = test.genres.apply(lambda x: 1 if g in [genre["name"] for genre in x] else 0)

train = train.drop("genres", axis=1)
test = test.drop("genres", axis=1)

train.head(3)


# In[20]:


list_of_languages = list(train.original_language)

# Generate a word cloud image
wordcloud = WordCloud(background_color="white",collocations=False,
                      width=1200, height=1000).generate(" ".join(list_of_languages))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[21]:


most_common_oglanguages = train.original_language.value_counts().head(10).index.values
for language in most_common_oglanguages:
    train[language] = train.original_language.apply(lambda x: 1 if x == language else 0)
    test[language] = test.original_language.apply(lambda x: 1 if x == language else 0)
train["other_og_language"] = train.original_language.apply(lambda x: 1 if x not in most_common_oglanguages else 0)
test["other_og_language"] = test.original_language.apply(lambda x: 1 if x not in most_common_oglanguages else 0)

train = train.drop("original_language", axis = 1)
test = test.drop("original_language", axis = 1)
train.head(5)


# In[22]:


sns.distplot(train.popularity)


# In[23]:


train.popularity = (train.popularity - train.popularity.mean() ) / train.popularity.std()
sns.distplot(train.popularity)


# In[24]:


train.head()


# In[25]:


train.production_companies[0]


# In[26]:


number_of_production_companies = train.production_companies.apply(lambda x: len(x))
sns.scatterplot(x=number_of_production_companies, y=train.revenue)


# In[27]:


train["n_production_companies"] = train.production_companies.apply(lambda x: len(x))
test["n_production_companies"] = test.production_companies.apply(lambda x: len(x))


# In[28]:


production_companies_list = train.production_companies.apply(lambda companies: [company["name"] for company in companies] if companies != {} else []).tolist()
production_companies_list[:3]


# In[29]:


production_companies_list = [company for x in production_companies_list for company in x]
production_companies_list[:3]


# In[30]:


# remove whitespaces to prepare for wordcloud
production_companies_list_wordcloud = [company.replace(" ","_") for company in production_companies_list]
production_companies_list_wordcloud[:3]


# In[31]:


# Generate a word cloud image
wordcloud = WordCloud(background_color="white",collocations=False,
                      width=1200, height=1000).generate(" ".join(production_companies_list_wordcloud))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[32]:


Counter(production_companies_list).most_common(10)


# In[33]:


# how many are there
len(set(production_companies_list))


# In[34]:


most_common_production_companies = Counter(production_companies_list).most_common(15)

for common_company,_ in most_common_production_companies:
    train[common_company] = train.production_companies.apply(lambda companies: 1                     if common_company in [company["name"] for company in companies] else 0)
    test[common_company] = test.production_companies.apply(lambda companies: 1                      if common_company in [company["name"] for company in companies] else 0)

train["other_production_company"] = train.production_companies.apply(lambda companies: 1                     if not any(company in most_common_production_companies for company in companies) else 0)

test["other_production_company"] = test.production_companies.apply(lambda companies: 1                     if not any(company in most_common_production_companies for company in companies) else 0)

train = train.drop("production_companies",axis=1)
test = test.drop("production_companies",axis=1)


# In[35]:


train.head(5)


# In[36]:


train.production_countries.apply(lambda x: len(x)).unique()


# In[37]:


production_countries_list = train.production_countries.apply(lambda countries: [country['iso_3166_1'] for country in countries])
production_countries_list = [country for x in production_countries_list for country in x]
unique_production_countries_list = set([country for x in production_countries_list for country in x])
revenue_country = {}
for unique_country in unique_production_countries_list:
    revenue_country[unique_country] = 0
    revenue_country[unique_country] += np.log(train.loc[:,["production_countries","revenue"]]         .apply(lambda x: x.revenue if unique_country in             [country['iso_3166_1'] for country in x.production_countries] else 0 ,axis=1).mean())


# In[38]:


sorted_revenue_country = sorted(revenue_country.items(), key=operator.itemgetter(1),reverse=True)
plt.figure(figsize=(16,10))
plt.title("Log of Mean Revenue per country")
plt.xlabel("Log of Mean revenue")
plt.ylabel("Country")
sns.barplot(y= [country[0] for country in sorted_revenue_country], x= [country[1] for country in sorted_revenue_country])


# In[39]:


prod_country_list = train.production_countries.apply(lambda c: [country["iso_3166_1"] for country in c] if c != {} else []).tolist()
prod_country_list = [country for x in prod_country_list for country in x]
prod_countrylist = [country.replace(" ", "_") for country in prod_country_list]
# Generate a word cloud image
wordcloud = WordCloud(background_color="white",collocations=False,
                      width=1500, height=800).generate(" ".join(prod_country_list))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[40]:


train.production_countries = train.production_countries     .apply(lambda countries: [country["iso_3166_1"] for country in countries] if countries != {} else [])


# In[41]:


common_production_countries = np.array(Counter(production_countries_list).most_common(20))[:,0]

for top_country in common_production_countries:
    train[top_country] = train.production_countries.apply(lambda countries: 1 if             top_country in countries else 0)
    
    test[top_country] = test.production_countries.apply(lambda countries: 1 if         top_country in countries else 0)

train["other_country"] = train.production_countries.apply(lambda countries:         1 if not any([country in common_production_countries         for country in countries]) else 0)

test["other_country"] = test.production_countries.apply(lambda countries:         1 if not any([country in common_production_countries         for country in countries]) else 0)

train = train.drop("production_countries",axis=1)
test = test.drop("production_countries",axis=1)


# In[42]:


train.release_date = pd.to_datetime(train.release_date,infer_datetime_format=True)
test.release_date = pd.to_datetime(train.release_date,infer_datetime_format=True)
train["release_month"] = train.release_date.apply(lambda x: x.month)
test["release_month"] = test.release_date.apply(lambda x: x.month)
train = train.drop("release_date",axis=1)
test = test.drop("release_date",axis=1)


# In[43]:


train = train.drop(["original_title", "overview", "poster_path","homepage",                    "imdb_id","status","title","crew","CollectionName","tagline","Keywords"],axis=1)
test = test.drop(["original_title", "overview", "poster_path","homepage",                    "imdb_id","status","title","crew","CollectionName","tagline","Keywords"],axis=1)


# In[44]:


train.spoken_languages = train.spoken_languages.apply(lambda languages: [language['iso_639_1'] for language in languages])


# In[45]:


spoken_laguages_list = train.spoken_languages.tolist()
spoken_laguages_list = [language for x in spoken_laguages_list for language in x]
unique_spoken_laguages_list = list(set(spoken_laguages_list))
top_spoken_languages = Counter(spoken_laguages_list).most_common(15)

for top_language in top_spoken_languages:
    train[top_language] = train.spoken_languages.apply(lambda languages: 1 if  top_language in languages else 0)
    test[top_language] = test.spoken_languages.apply(lambda languages: 1 if  top_language in languages else 0)

train["other_spoken_languages"] = train.spoken_languages.apply(lambda languages:                                 1 if not any([language in top_spoken_languages for language in languages]) else 0)

test["other_spoken_languages"] = test.spoken_languages.apply(lambda languages:                                 1 if not any([language in top_spoken_languages for language in languages]) else 0)

train = train.drop("spoken_languages",axis=1)
test = test.drop("spoken_languages",axis=1)
    


# In[46]:


train.cast = train.cast.apply(lambda cast: [person['name'] for person in cast] if cast != {} else [])
test.cast = test.cast.apply(lambda cast: [person['name'] for person in cast] if cast != {} else [])


# In[47]:


Counter(actors_list).most_common(25)


# In[48]:


actors_list = train.cast.tolist()
actors_list = [actor for x in actors_list for actor in x]
most_common_actors = np.array(Counter(actors_list).most_common(20))[:,0]


for top_actor in most_common_actors:
    train[top_actor] = train.cast.apply(lambda people: 1 if  top_actor in people else 0)
    test[top_actor] = test.cast.apply(lambda people: 1 if  top_actor in people else 0)

train["other_actors"] = train.cast.apply(lambda people:                                 1 if not any([person in most_common_actors for person in people]) else 0)

test["other_actors"] = test.cast.apply(lambda people:                                 1 if not any([person in most_common_actors for person in people]) else 0)

train = train.drop("cast",axis=1)
test = test.drop("cast",axis=1)


# In[49]:


train.runtime = (train.runtime - train.runtime.mean()) / train.runtime.std()
test.runtime = (test.runtime - test.runtime.mean()) / test.runtime.std()


# In[50]:


train.budget = (train.budget - train.budget.mean()) / train.budget.std()
test.budget = (test.budget - test.budget.mean()) / test.budget.std()


# In[51]:


train = train.fillna(0)
test= test.fillna(0)


# In[52]:


X_train, X_test, y_train, y_test =          train_test_split(train.drop("revenue",axis=1), train.revenue, test_size=0.33, random_state=42)

reg = DecisionTreeRegressor()
reg.fit(X_train,y_train)
pred = reg.predict(X_test)

