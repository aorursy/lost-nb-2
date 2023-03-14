#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import seaborn as sns 
from scipy import stats
from scipy.stats import norm,skew
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import xgboost as xgb
import catboost as catb
import operator
import time
import ast
from collections import Counter
import itertools
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[3]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[4]:


print('The shape of train data is:', train.shape)
print('The shape of test data is:', test.shape)


# In[5]:


print('The columns present in train data are:', train.columns)
print('The columns present in test data are:', test.columns)


# In[6]:


train.describe()


# In[7]:


train.info()


# In[8]:


total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
miss = print(missing_data)


# In[9]:


fig = plt.figure(figsize=(15,8))
plt.title("Distriution of missing values")
train.isna().sum().sort_values(ascending=True).plot(kind='bar', colors='Red', fontsize=10)


# In[10]:


pd.set_option("display.max_columns", 100)
train.head(5)


# In[11]:


train.dropna().shape


# In[12]:


train[['revenue', 'budget', 'runtime']].describe()


# In[13]:


train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning
train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          
train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs
train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven
train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 
train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty
train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood
train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II
train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada
train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol
train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times
train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman
train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   
train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 
train.loc[train['id'] == 1542,'budget'] = 1              # All at Once
train.loc[train['id'] == 1542,'budget'] = 15800000       # Crocodile Dundee II
train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp
train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit
train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon
train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed
train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget
train.loc[train['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers
train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus
train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams
train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D
train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture


# In[14]:


test.loc[test['id'] == 3889,'budget'] = 15000000       # Colossal
test.loc[test['id'] == 6733,'budget'] = 5000000        # The Big Sick
test.loc[test['id'] == 3197,'budget'] = 8000000        # High-Rise
test.loc[test['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2
test.loc[test['id'] == 5704,'budget'] = 4300000        # French Connection II
test.loc[test['id'] == 6109,'budget'] = 281756         # Dogtooth
test.loc[test['id'] == 7242,'budget'] = 10000000       # Addams Family Values
test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family
test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage
test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee


# In[15]:


train.runtime[train.id == 391] = 86                     #Il peor natagle de la meva vida
train.runtime[train.id == 592] = 90                     #А поутру они проснулись
train.runtime[train.id == 925] = 95                     #¿Quién mató a Bambi?
train.runtime[train.id == 978] = 93                     #La peggior settimana della mia vita
train.runtime[train.id == 1256] = 92                    #Cipolla Colt
train.runtime[train.id == 1542] = 93                    #Все и сразу
train.runtime[train.id == 1875] = 86                    #Vermist
train.runtime[train.id == 2151] = 108                   #Mechenosets
train.runtime[train.id == 2499] = 108                   #Na Igre 2. Novyy Uroven
train.runtime[train.id == 2646] = 98                    #同桌的妳
train.runtime[train.id == 2786] = 111                   #Revelation
train.runtime[train.id == 2866] = 96                    #Tutto tutto niente niente

# TEST
test.runtime[test.id == 4074] = 103                     #Shikshanachya Aaicha Gho
test.runtime[test.id == 4222] = 93                      #Street Knight
test.runtime[test.id == 4431] = 100                     #Плюс один
test.runtime[test.id == 5520] = 86                      #Glukhar v kino
test.runtime[test.id == 5845] = 83                      #Frau Müller muss weg!
test.runtime[test.id == 5849] = 140                     #Shabd
test.runtime[test.id == 6210] = 104                     #Le dernier souffle
test.runtime[test.id == 6804] = 145                     #Chaahat Ek Nasha..
test.runtime[test.id == 7321] = 87                      #El truco del manco


# In[16]:


power_six = train.id[train.budget > 1000][train.revenue < 100]

for k in power_six :
    train.loc[train['id'] == k,'revenue'] =  train.loc[train['id'] == k,'revenue'] * 1000000


# In[17]:


from scipy import stats
from scipy.stats import norm


# In[18]:


def visualize_distribution(y):
    sns.distplot(y,fit=norm)
    mu,sigma=norm.fit(y)
    plt.legend(["Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})".format(mu,sigma)])
    plt.title("Distribution of revenue")
    plt.ylabel("Frequency")
    plt.show()
    
    
def visualize_probplot(y):
    stats.probplot(y,plot=plt)
    plt.show()


# In[19]:


visualize_distribution(test.budget)
visualize_probplot(test.budget)


# In[20]:


train['budget'] = np.log1p(train['budget'])
test['budget'] = np.log1p(test['budget'])

train['popularity'] = np.log1p(train['popularity'])
test['popularity'] = np.log1p(test['popularity'])


# In[21]:


visualize_distribution(train.budget)
visualize_probplot(train.budget)


# In[22]:


visualize_distribution(train.revenue)
visualize_probplot(train.revenue)


# In[23]:


train = train.drop(['imdb_id', 'poster_path'], axis = 1)
test = test.drop(['imdb_id', 'poster_path'], axis = 1)


# In[24]:


import ast


# In[25]:


train.loc[train["cast"].notnull(),"cast"]=train.loc[train["cast"].notnull(),"cast"].apply(lambda x : ast.literal_eval(x))
train.loc[train["crew"].notnull(),"crew"]=train.loc[train["crew"].notnull(),"crew"].apply(lambda x : ast.literal_eval(x))

test.loc[test["cast"].notnull(),"cast"]=test.loc[test["cast"].notnull(),"cast"].apply(lambda x : ast.literal_eval(x))
test.loc[test["crew"].notnull(),"crew"]=test.loc[test["crew"].notnull(),"crew"].apply(lambda x : ast.literal_eval(x))


# In[26]:


train.loc[train["cast"].notnull(),"cast"]=train.loc[train["cast"].notnull(),"cast"].apply(lambda x : [y["name"] for y in x if y["order"]<6]) 

test.loc[test["cast"].notnull(),"cast"]=test.loc[test["cast"].notnull(),"cast"].apply(lambda x : [y["name"] for y in x if y["order"]<6]) 


# In[27]:


def get_DirProdExP(df):
    df["Director"]=[[] for i in range(df.shape[0])]
    df["Producer"]=[[] for i in range(df.shape[0])]
    df["Executive Producer"]=[[] for i in range(df.shape[0])]

    df["Director"]=df.loc[df["crew"].notnull(),"crew"]    .apply(lambda x : [y["name"] for y in x if y["job"]=="Director"])

    df["Producer"]=df.loc[df["crew"].notnull(),"crew"]    .apply(lambda x : [y["name"] for y in x if y["job"]=="Producer"])

    df["Executive Producer"]=df.loc[df["crew"].notnull(),"crew"]    .apply(lambda x : [y["name"] for y in x if y["job"]=="Executive Producer"])
    
    return df


# In[28]:


train = get_DirProdExP(train)
test = get_DirProdExP(test)


# In[29]:


print ('budget: ' + str(sum(train['budget'].isna())) + ', popularity: ' + str(sum(train['popularity'].isna())) + 
      ', runtime: ' + str(sum(train['runtime'].isna())) + ', revenue: ' + str(sum(train['revenue'].isna())))


# In[30]:


pair = ['budget', 'popularity', 'runtime', 'revenue']
sns.pairplot(train[pair].dropna())


# In[31]:


print("raw format:", train['spoken_languages'].iloc[0])

train['spoken_languages'] = train['spoken_languages'].apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
test['spoken_languages'] = test['spoken_languages'].apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))

train.head().spoken_languages


# In[32]:


train['nb_spoken_languages'] = train.spoken_languages.apply(len)
test['nb_spoken_languages'] = test.spoken_languages.apply(len)

train['english_spoken'] = train.spoken_languages.apply(lambda x: 'en' in x)
test['english_spoken'] = test.spoken_languages.apply(lambda x: 'en' in x)


# In[33]:


train['nb_spoken_languages'].value_counts()


# In[34]:


all_languages = pd.concat([train.original_language, test.original_language], axis=0).value_counts()
all_languages[all_languages > 10]


# In[35]:


main_languages = list(all_languages[all_languages>20].index)


# In[36]:


dict_language = dict(zip(main_languages, range(1, len(main_languages)+1)))
dict_language['other'] = 0


# In[37]:


train.original_language = train.original_language.apply(lambda x: x if x in main_languages else 'other')
test.original_language = test.original_language.apply(lambda x: x if x in main_languages else 'other')


# In[38]:


train['language'] = train.original_language.apply(lambda x: dict_language[x])
test['language'] = test.original_language.apply(lambda x: dict_language[x])


# In[39]:


train.genres = train.genres.apply(lambda x: list(map(lambda d: list(d.values())[1], ast.literal_eval(x)) if isinstance(x, str) else []))
test.genres = test.genres.apply(lambda x: list(map(lambda d: list(d.values())[1], ast.literal_eval(x)) if isinstance(x, str) else []))

train.genres.head()


# In[40]:


plt.bar(train.genres.apply(len).value_counts().sort_index().keys(), train.genres.apply(len).value_counts().sort_index())


# In[41]:


for v in train[train.genres.apply(len)==7][['title', 'genres']].values:
    print('film:', v[0], '\ngenres:', *v[1], '\n')


# In[42]:


genres = Counter(itertools.chain.from_iterable(pd.concat((train.genres, test.genres), axis=0).values))
genres


# In[43]:


get_ipython().run_cell_magic('time', '', "temp_train = train[['id', 'genres']]\ntemp_test = test[['id', 'genres']]\n\nfor g in genres:\n    temp_train[g] = temp_train.genres.apply(lambda x: 1 if g in x else 0)\n    temp_test[g] = temp_test.genres.apply(lambda x: 1 if g in x else 0)\n    \nX_train = temp_train.drop(['genres', 'id'], axis=1).values\nX_test = temp_test.drop(['genres', 'id'], axis=1).values\n\n# Number of features we want for genres\nn_comp_genres = 3\n\n# Build the SVD pipeline\nsvd = make_pipeline(\n    TruncatedSVD(n_components=n_comp_genres),\n    Normalizer(norm='l2', copy=False)\n)\n\n# Here are our new features\nf_train = svd.fit_transform(X_train)\nf_test = svd.transform(X_test)")


# In[44]:


temp_train.head(3)


# In[45]:


my_genres = [g for g in genres if g!= 'TV Movie']
my_genres


# In[46]:


train = pd.concat([train, temp_train.iloc[:,1:]], axis=1) 
train.drop(train.columns[-1],axis=1, inplace = True)

test = pd.concat([test, temp_test.iloc[:,1:]], axis=1) 
test.drop(test.columns[-1], axis=1, inplace = True)


# In[47]:


train.Keywords = train.Keywords.apply(lambda x: list(map(lambda d: list(d.values())[1], ast.literal_eval(x)) if isinstance(x, str) else []))
test.Keywords = test.Keywords.apply(lambda x: list(map(lambda d: list(d.values())[1], ast.literal_eval(x)) if isinstance(x, str) else []))


# In[48]:


train['nb_keywords'] = train.Keywords.apply(len)
test['nb_keywords'] = test.Keywords.apply(len)


# In[49]:


train.production_companies = train.production_companies.apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
test.production_companies = test.production_companies.apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))


# In[50]:


production_companies = Counter(itertools.chain.from_iterable(pd.concat((train.production_companies, test.production_companies), axis=0).values))
print("Number of different production companies:", len(production_companies))


# In[51]:


train['nb_production_companies'] = train.production_companies.apply(len)
test['nb_production_companies'] = test.production_companies.apply(len)


# In[52]:


get_ipython().run_cell_magic('time', '', "print('Applying SVD on production companies to create reduced features')\n\n# Factorizing all the little production companies into an 'other' variable\nbig_companies = [p for p in production_companies if production_companies[p] > 30]\ntrain.production_companies = train.production_companies.apply(lambda l: list(map(lambda x: x if x in big_companies else 'other', l)))\n\ntemp_train = train[['id', 'production_companies']]\ntemp_test = test[['id', 'production_companies']]\n\nfor p in big_companies + ['other']:\n    temp_train[p] = temp_train.production_companies.apply(lambda x: 1 if p in x else 0)\n    temp_test[p] = temp_test.production_companies.apply(lambda x: 1 if p in x else 0)\n    \nX_train = temp_train.drop(['production_companies', 'id'], axis=1).values\nX_test = temp_test.drop(['production_companies', 'id'], axis=1).values\n\n# Number of features we want for genres\nn_comp_production_companies = 3\n\n# Build the SVD pipeline\nsvd = make_pipeline(\n    TruncatedSVD(n_components=n_comp_production_companies),\n    Normalizer(norm='l2', copy=False)\n)\n\n# Here are our new features\nf_train = svd.fit_transform(X_train)\nf_test = svd.transform(X_test)\n\nfor i in range(n_comp_production_companies):\n    train['production_companies_reduced_{}'.format(i)] = f_train[:, i]\n    test['production_companies_reduced_{}'.format(i)] = f_test[:, i]")


# In[53]:


train[['production_companies_reduced_0', 'production_companies_reduced_1', 'production_companies_reduced_2']].head(3)


# In[54]:


train.production_countries = train.production_countries.apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
test.production_countries = test.production_countries.apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))


# In[55]:


production_countries = Counter(itertools.chain.from_iterable(pd.concat((train.production_countries, test.production_countries), axis=0).values))
print("Number of different production companies:", len(production_countries))


# In[56]:


get_ipython().run_cell_magic('time', '', "print('Applying SVD on production countries to create reduced features')\n\n# Factorizing all the little production companies into an 'other' variable\nbig_countries = [p for p in production_countries if production_countries[p] > 30]\ntrain.production_countries = train.production_countries.apply(lambda l: list(map(lambda x: x if x in big_countries else 'other', l)))\n\ntemp_train = train[['id', 'production_countries']]\ntemp_test = test[['id', 'production_countries']]\n\nfor p in big_countries + ['other']:\n    temp_train[p] = temp_train.production_countries.apply(lambda x: 1 if p in x else 0)\n    temp_test[p] = temp_test.production_countries.apply(lambda x: 1 if p in x else 0)\n    \nX_train = temp_train.drop(['production_countries', 'id'], axis=1).values\nX_test = temp_test.drop(['production_countries', 'id'], axis=1).values\n\n# Number of features we want for genres\nn_comp_production_countries = 3\n\n# Build the SVD pipeline\nsvd = make_pipeline(\n    TruncatedSVD(n_components=n_comp_production_countries),\n    Normalizer(norm='l2', copy=False)\n)\n\n# Here are our new features\nf_train = svd.fit_transform(X_train)\nf_test = svd.transform(X_test)\n\nfor i in range(n_comp_production_countries):\n    train['production_countries_reduced_{}'.format(i)] = f_train[:, i]\n    test['production_countries_reduced_{}'.format(i)] = f_test[:, i]")


# In[57]:


train[['production_countries_reduced_0', 'production_countries_reduced_1', 'production_countries_reduced_2']].head(3)


# In[58]:


test.loc[test.release_date.isna(), 'release_date'] = '05/01/00'


# In[59]:


#Train
train['release_date'] = pd.to_datetime(train['release_date'], format='%m/%d/%y')
train['Year'] = train.release_date.dt.year
train['Month'] = train.release_date.dt.month
train['Day'] = train.release_date.dt.day
train['dayofweek'] = train.release_date.dt.dayofweek 
train['quarter'] = train.release_date.dt.quarter   
#Test
test['release_date'] = pd.to_datetime(test['release_date'], format='%m/%d/%y')
test['Year'] = test.release_date.dt.year
test['Month'] = test.release_date.dt.month
test['Day'] = test.release_date.dt.day
test['dayofweek'] = test.release_date.dt.dayofweek 
test['quarter'] = test.release_date.dt.quarter  


# In[60]:


dummies = pd.get_dummies(train['Month'] ,drop_first=True).rename(columns=lambda x: 'Month' + str(x))
dummies2 = pd.get_dummies(test['Month'] ,drop_first=True).rename(columns=lambda x: 'Month' + str(int(x)))
train = pd.concat([train, dummies], axis=1)
test = pd.concat([test, dummies2], axis = 1)


# In[61]:


ddow = pd.get_dummies(train['dayofweek'] ,drop_first=True).rename(columns=lambda x: 'dayofweek' + str(x))
ddow2 = pd.get_dummies(test['dayofweek'] ,drop_first=True).rename(columns=lambda x: 'dayofweek' + str(int(x)))
train = pd.concat([train, ddow], axis=1)
test = pd.concat([test, ddow2], axis = 1)


# In[62]:


print ('Train: ' + str(max(train.Year)) + ' Test: ' + str(max(test.Year)))


# In[63]:


#Train
train.loc[train['Year'] > 2018, 'Year'] = train.loc[train['Year'] > 2018, 'Year'].apply(lambda x: x - 100)
#Test
test.loc[test['Year'] > 2018, 'Year'] = test.loc[test['Year'] > 2018, 'Year'].apply(lambda x: x - 100)


# In[64]:


test.Year.describe()


# In[65]:


data_plot = train[['revenue', 'Year']]
money_Y = data_plot.groupby('Year')['revenue'].sum()

money_Y.plot(figsize=(15,8))
plt.xlabel("Year of release")
plt.ylabel("revenue")
plt.xticks(np.arange(1960,2015,5))

plt.show()


# In[66]:


f,ax = plt.subplots(figsize=(18, 10))
plt.bar(train.Month, train.revenue, color = 'Red')
plt.xlabel("Month of release")
plt.ylabel("revenue")
plt.show()


# In[67]:


f,ax = plt.subplots(figsize=(15, 10))
plt.bar(train.dayofweek, train.revenue, color = 'Red')
plt.xlabel("Dayofweek of release")
plt.ylabel("revenue")
plt.show()


# In[68]:


def fuzzy_feat(df):
    
    df['Ratiobudgetbypopularity'] = df['budget']/df['popularity']
    df['RatiopopularitybyYear'] = df['popularity']/df['Year']
    df['RatoioruntimebyYear'] = df['runtime']/df['Year']
    
    
    df['budget_runtime_ratio'] = df['budget']/df['runtime'] 
    df['budget_Year_ratio'] = df['budget']/df['Year']
    
    return df


# In[69]:


train = fuzzy_feat(train)
test = fuzzy_feat(test)


# In[70]:


# NAs

train['has_homepage'] = np.where(train['homepage'].isna(), 0, 1)
train ['has_collection'] = np.where(train['belongs_to_collection'].isna(), 0, 1)

test['has_homepage'] = np.where(test['homepage'].isna(), 0, 1)
test ['has_collection'] = np.where(test['belongs_to_collection'].isna(), 0, 1)

train['has_tagline'] = np.where (train['tagline'].isna(), 0, 1)
test['has_tagline'] = np.where (test['tagline'].isna(), 0, 1)

#Fix Strange occurences

train['title_different'] = np.where(train['original_title'] == train['title'], 0, 1)
test['title_different'] = np.where(test['original_title'] == test['title'], 0, 1)

train['isReleased'] = np.where(train['status'] != 'Released', 0, 1)
test['isReleased'] = np.where(test['status'] != 'Released', 0, 1)


# In[71]:


features = ['budget', 
            'popularity', 
            'runtime', 
            'nb_spoken_languages', 
            'nb_production_companies',
            'english_spoken', 
            'language',
            'has_homepage', 'has_collection', 'isReleased', 'has_tagline', 'title_different',
            'Day',
            'quarter', 'Year',
            'nb_keywords', 
            'Month2', 'Month3',  'Month4', 'Month5',  'Month6', 'Month7',
            'Ratiobudgetbypopularity', 'RatiopopularitybyYear',
            'RatoioruntimebyYear', 'budget_runtime_ratio', 'budget_Year_ratio',
            'Month8', 'Month9',  'Month10', 'Month11', 'Month12']


# In[72]:


features += [col for col in train.columns if 'dayofweek' in col and col != "dayofweek"]
features += my_genres
features += ['production_companies_reduced_{}'.format(i) for i in range(n_comp_production_companies)]
features += ['production_countries_reduced_{}'.format(i) for i in range(n_comp_production_countries)]
X = train[features]
X['revenue'] = train.revenue


# In[73]:


X.columns


# In[74]:


cor_features = X[['revenue', 'budget',  'popularity', 'runtime', 'nb_spoken_languages', 'nb_production_companies',
            'Day', 'quarter', 'Year','nb_keywords' ]]
f,ax = plt.subplots(figsize=(20, 12))
sns.heatmap(cor_features.corr(), annot=True, linewidths=.7, fmt= '.2f',ax=ax)
plt.show()


# In[75]:


X = X.drop(['revenue'], axis = 1)
y = train.revenue.apply(np.log1p)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12, shuffle=True)


# In[76]:


params = {'objective': 'reg:linear', 
          'eta': 0.01, 
          'max_depth': 6, 
          'min_child_weight': 3,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'colsample_bylevel': 0.50, 
          'gamma': 1.45, 
          'eval_metric': 'rmse', 
          'seed': 12, 
          'silent': True    
}

# create dataset for xgboost
xgb_data = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_test, y_test), 'valid')]


# In[77]:


print('Starting training...')
xgb_model = xgb.train(params, 
                  xgb.DMatrix(X_train, y_train),
                  5000,  
                  xgb_data, 
                  verbose_eval=200,
                  early_stopping_rounds=200)


# In[78]:


xgb_model_full = xgb.XGBRegressor(objective  = 'reg:linear', 
          eta = 0.01, 
          max_depth = 6,
          min_child_weight = 3,
          subsample = 0.8, 
          colsample_bytree = 0.8,
          colsample_bylevel = 0.50, 
          gamma = 1.45, 
          eval_metric = 'rmse',
          seed = 12, n_estimators = 2000)


# In[79]:


xgb_model_full.fit (X.values, y)


# In[80]:


catmodel = catb.CatBoostRegressor(iterations=10000, 
                                 learning_rate=0.01, 
                                 depth=5, 
                                 eval_metric='RMSE',
                                 colsample_bylevel=0.7,
                                 bagging_temperature = 0.2,
                                 metric_period = None,
                                 early_stopping_rounds=200,
                                 random_seed=12)


# In[81]:


ti=time.time()
catmodel.fit(X, y, 
             eval_set=(X_train, y_train), 
             verbose=500, 
             use_best_model=True)

print("Number of minutes of training of model_cal = {:.2f}".format((time.time()-ti)/60))

cat_pred_train=catmodel.predict(X)
cat_pred_train[cat_pred_train<0]=0


# In[82]:


fea_imp = pd.DataFrame({'imp': catmodel.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 12))
plt.savefig('catboost_feature_importance.png') 


# In[83]:


fig, ax = plt.subplots(figsize=(20,12))
xgb.plot_importance(xgb_model, max_num_features=30, height = 0.8, ax = ax)
plt.title('XGBOOST Features (avg over folds)')
plt.show()


# In[84]:


train_pred = xgb_model.predict(xgb.DMatrix(X), ntree_limit=xgb_model.best_ntree_limit)
plt.figure(figsize=(32,15))
plt.plot(y[:500],label="Real")
plt.plot(train_pred[:500],label="train_pred")
plt.legend(fontsize=15)
plt.title("Real and predicted revenue of first 500 entries of train set",fontsize=24)
plt.show()


# In[85]:


plt.figure(figsize=(32,15))
plt.plot(y[:500],label="Real")
plt.plot(cat_pred_train[:500],label="train_pred")
plt.legend(fontsize=15)
plt.title("Real and predicted revenue of first 500 entries of train set",fontsize=24)
plt.show()


# In[86]:


plt.figure(figsize=(35,18))
plt.plot(y[:600],label="Real", color = "red")
plt.plot(xgb_model.predict(xgb.DMatrix(X), ntree_limit=xgb_model.best_ntree_limit)[:600],label="xgb", color = "blue")
plt.plot(cat_pred_train[:600],label="catb", color = "green")
plt.legend(fontsize=15)
plt.title("Real and predicted revenue of first 500 entries of train set",fontsize=24)
plt.show()


# In[87]:


X_test = test[features]
xgb_pred = np.expm1(xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit))
pd.DataFrame({'id': test.id, 'revenue': xgb_pred}).to_csv('xgbsubmission.csv', index=False)


# In[88]:


xgb_pred[0]


# In[89]:


xgb_pred_f = np.expm1(xgb_model_full.predict(X_test.values))
pd.DataFrame({'id': test.id, 'revenue': xgb_pred_f}).to_csv('xgbfullsubmission.csv', index=False)
xgb_pred_f[0]


# In[90]:


X_test = test[features]
catb_pred = np.expm1(catmodel.predict(X_test.values))
pd.DataFrame({'id': test.id, 'revenue': catb_pred}).to_csv('catbsubmission.csv', index=False)


# In[91]:


catb_pred[0]


# In[92]:


ens_pred = 0.3*xgb_pred_f + 0.7*catb_pred
pd.DataFrame({'id': test.id, 'revenue': ens_pred}).to_csv('enssubmission.csv', index=False)


# In[93]:


ens_pred[0]


# In[94]:


pd.DataFrame({'id': test.id, 'revenue': ens_pred}).head()

