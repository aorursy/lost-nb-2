#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import linear_model
from collections import Counter


# In[2]:


train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")


# In[3]:


test.shape


# In[4]:


train.shape


# In[5]:


train.info()


# In[6]:


train['interest_level'].value_counts(normalize=True)


# In[7]:


x = train['street_address'].value_counts()
x = x.where(x>50).dropna()
x = x.index.tolist()
z = train[['street_address','interest_level']].where(train['street_address']     .apply(lambda y: y in x)).dropna()


# In[8]:


a = train[train['display_address']=='Water Street']['bedrooms'].count()

def countess(x):
    return train[train['display_address'] == x]['bedrooms'].count()

#train['display_address'][:100]
#train['display_address'].apply(countess)

#train['display_address'][:2000].map(countess)
train['display_address'].value_counts().index.tolist()

train[(train['display_address']=='Water Street')&(train['interest_level']=='high')]['bedrooms'].count()

train[['display_address','interest_level']]#.value_counts()


# In[9]:


z


# In[10]:


train[['street_address','interest_level']].unique()


# In[11]:


z.groupby(['street_address', 'interest_level']).agg({'interest_level': 'count'}).head(10)


# In[12]:


z.groupby(['street_address', 'interest_level'])


# In[13]:


z.groupby(['street_address', 'interest_level'])


# In[14]:


interest_rating = ['low','medium','high']

for i in interest_rating:
    train[i] = train['interest_level'].apply(lambda x: 1 if x==i else 0)


# In[15]:


train.street_address.unique().shape


# In[16]:


#feature_list = [item.lower() for item in feature_list]
#cl = Counter(feature_list)

display_address_list=[]

da = Counter(train.display_address)

for key in da:
    if da[key]>175:
        display_address_list.append(key)
        print(key)

len(display_address_list)


# In[17]:


#feature_list = [item.lower() for item in feature_list]
#cl = Counter(feature_list)

street_address_list=[]

sa = Counter(train.street_address)

for key in sa:
    if da[key]>120:
        street_address_list.append(key)
        print(key)

len(street_address_list)


# In[18]:


for i in display_address_list:
    train[i] = train['display_address'].apply(lambda x: 1 if x==i else 0)
    test[i] = test['display_address'].apply(lambda x: 1 if x==i else 0)


# In[19]:


for i in street_address_list:
    train[i] = train['street_address'].apply(lambda x: 1 if x==i else 0)
    test[i] = test['street_address'].apply(lambda x: 1 if x==i else 0)


# In[20]:


train.head()


# In[21]:


train['interest_level'].value_counts(normalize=True)


# In[22]:


print(train.shape[0])
print(train.shape[0]/8)


# In[23]:


x = train['features'].tolist()

feature_list = []

for i in range(0,len(x),6169):
    print(i+6169)
    feature_listadd = []
    for j in range(i,i+6169):#len(x)):
        feature_listadd = feature_listadd + x[j]
    feature_list = feature_list + feature_listadd
    
#lowercase is necessary, otherwise matches will be missed when there are differences in case
feature_list = [item.lower() for item in feature_list]
cl = Counter(feature_list)


# In[24]:


#Keeping features that occur at least 1,000 times yields the following list

feature_list = []

for key in cl:
    if cl[key]>1000:
        feature_list.append(key)
        print(key)

feature_list


# In[25]:


for i in feature_list:
    train[i] = train['features'][:train.shape[0]].apply         (lambda x: 1 if i in [y.lower() for y in x] else 0)


# In[26]:


for i in feature_list:
    test[i] = test['features'][:test.shape[0]].apply         (lambda x: 1 if i in [y.lower() for y in x] else 0)


# In[27]:


#Combining 'pre-war' & 'prewar'

train['prewar'] = train['prewar'] + train['pre-war']
del train ['pre-war']
test['prewar'] = test['prewar'] + test['pre-war']
del test ['pre-war']


# In[28]:


mapping = {'low': 0, 'medium': 1, 'high':2}
train['interest_level'] = train['interest_level'].apply(lambda x: mapping.get(x))


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[30]:


feature_list.remove('pre-war')
feature_list.append('bedrooms')
feature_list.append('bathrooms')
feature_list.append('price')
feature_list.append('latitude')
feature_list.append('longitude')


# In[31]:


feature_list = feature_list + display_address_list + street_address_list


# In[32]:


feature_list


# In[33]:


modeltrain, modeltest = train_test_split(train, test_size = 0.2,random_state=0)

#For testing
Xmodeltrain = modeltrain[feature_list]
Ymodeltrain = modeltrain['interest_level']

Xmodeltest = modeltest[feature_list]
Ymodeltest = modeltest['interest_level']

#For fitting all the data
Xtrain = train[feature_list]
Ytrain = train['interest_level']

#For prediction from the test tile
Xtest = test[feature_list]


# In[34]:


#Models
regr = linear_model.LinearRegression()
random_forest = RandomForestClassifier(n_estimators=500)
gaussian = GaussianNB()
logreg = linear_model.LogisticRegression()
kneighbors = KNeighborsClassifier()

#Model List
Models = [regr,random_forest,gaussian,logreg,kneighbors]

#Fitting
for i in Models:
    i.fit(Xmodeltrain,Ymodeltrain)


# In[35]:


def runscore(x):
    #print('\n')
    print('Test Score: ' + str(x.score(Xmodeltest, Ymodeltest)))
    print('Train Score: ' + str(x.score(Xmodeltrain, Ymodeltrain)))


# In[36]:


#Run Scores
for model in Models:
    print('\n')
    print(model)
    runscore(model)


# In[37]:


for model in Models:
    model.fit(Xtrain, Ytrain)


# In[38]:


for model in Models:
    if model not in [regr]:
        test['interest_level']=model.predict(Xtest)
        print('\n')
        print(model)
        print(test['interest_level'].value_counts(normalize=True))


# In[39]:


test['interest_level']=random_forest.predict(Xtest)


# In[40]:


test['interest_level'].value_counts(normalize=True)


# In[41]:


maplow = {0: .75, 1: .20, 2:.05}
mapmedium = {0: .125, 1: .75, 2:.125}
maphigh = {0: .05, 1: .15, 2:.80}

test['low'] = test['interest_level'].apply(lambda x: maplow.get(x))
test['medium'] = test['interest_level'].apply(lambda x: mapmedium.get(x))
test['high'] = test['interest_level'].apply(lambda x: maphigh.get(x))


# In[42]:


submission = test[['listing_id','high','medium','low']]


# In[43]:


#submission = submission.set_index('listing_id')
submission.head(10)


# In[44]:


submission.to_csv('submission13.csv', index=False)


# In[45]:




