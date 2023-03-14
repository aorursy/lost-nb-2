#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("."))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
print ('Datasets:' , 'train:' , train.shape , 'test:' , test.shape)


# In[3]:


#train = train.dropna()
print ('Datasets:' , 'train:' , train.shape , 'test:' , test.shape)


# In[4]:


train.info()


# In[5]:


# Several columns (e.g. genres) are lists of values - split them to dictionaries for easier processing
import ast
for c in ['genres', 'production_companies', 'production_countries', 'spoken_languages', 
          'Keywords', 'cast', 'crew']:
    train[c] = train[c].apply(lambda x: [''] if pd.isna(x) else [str(j['name']) for j in (eval(x))])
    test[c]  = test[c].apply(lambda x: [''] if pd.isna(x) else [str(j['name']) for j in (eval(x))])


# In[6]:


train.head()


# In[7]:


train["cast_len"] = train.loc[train["cast"].notnull(),"cast"].apply(lambda x : len(x))
train["crew_len"] = train.loc[train["crew"].notnull(),"crew"].apply(lambda x : len(x))

train["production_companies_len"]=train.loc[train["production_companies"].notnull(),"production_companies"].apply(lambda x : len(x))

train["production_countries_len"]=train.loc[train["production_countries"].notnull(),"production_countries"].apply(lambda x : len(x))

train["Keywords_len"]=train.loc[train["Keywords"].notnull(),"Keywords"].apply(lambda x : len(x))
train["genres_len"]=train.loc[train["genres"].notnull(),"genres"].apply(lambda x : len(x))

train['original_title_letter_count'] = train['original_title'].str.len() 
train['original_title_word_count'] = train['original_title'].str.split().str.len() 
train['title_word_count'] = train['title'].str.split().str.len()
train['overview_word_count'] = train['overview'].str.split().str.len()
train['tagline_word_count'] = train['tagline'].str.split().str.len()


# In[8]:


test["cast_len"] = test.loc[test["cast"].notnull(),"cast"].apply(lambda x : len(x))
test["crew_len"] = test.loc[test["crew"].notnull(),"crew"].apply(lambda x : len(x))

test["production_companies_len"]=test.loc[test["production_companies"].notnull(),"production_companies"].apply(lambda x : len(x))

test["production_countries_len"]=test.loc[test["production_countries"].notnull(),"production_countries"].apply(lambda x : len(x))

test["Keywords_len"]=test.loc[test["Keywords"].notnull(),"Keywords"].apply(lambda x : len(x))
test["genres_len"]=test.loc[test["genres"].notnull(),"genres"].apply(lambda x : len(x))

test['original_title_letter_count'] = test['original_title'].str.len() 
test['original_title_word_count'] = test['original_title'].str.split().str.len() 
test['title_word_count'] = test['title'].str.split().str.len()
test['overview_word_count'] = test['overview'].str.split().str.len()
test['tagline_word_count'] = test['tagline'].str.split().str.len()


# In[9]:


train.loc[train["homepage"].notnull(),"homepage"]=1
train["homepage"]=train["homepage"].fillna(0)  # Note that we only need to know if the film has a webpage or not!

train["in_collection"]=1
train.loc[train["belongs_to_collection"].isnull(),"in_collection"]=0

train["has_tagline"]=1
train.loc[train["tagline"].isnull(),"has_tagline"]=0

train["title_different"]=1
train.loc[train["title"]==train["original_title"],"title_different"]=0

train["isReleased"]=1
train.loc[train["status"]!="Released","isReleased"]=0


# In[10]:


test.loc[test["homepage"].notnull(),"homepage"]=1
test["homepage"]=test["homepage"].fillna(0)  # Note that we only need to know if the film has a webpage or not!

test["in_collection"]=1
test.loc[test["belongs_to_collection"].isnull(),"in_collection"]=0

test["has_tagline"]=1
test.loc[test["tagline"].isnull(),"has_tagline"]=0

test["title_different"]=1
test.loc[test["title"]==test["original_title"],"title_different"]=0

test["isReleased"]=1
test.loc[test["status"]!="Released","isReleased"]=0


# In[11]:


train['release_year'] = train['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)[2]
train.loc[ (train['release_year'] <= 19) & (train['release_year'] < 100), "release_year"] += 2000
train.loc[ (train['release_year'] > 19)  & (train['release_year'] < 100), "release_year"] += 1900


# In[12]:


test['release_year'] = test['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)[2]
test.loc[ (test['release_year'] <= 19) & (test['release_year'] < 100), "release_year"] += 2000
test.loc[ (test['release_year'] > 19)  & (test['release_year'] < 100), "release_year"] += 1900


# In[13]:


train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


# In[14]:


train.describe()


# In[15]:


train.head()


# In[16]:


from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([train.original_language, test.original_language], ignore_index=True))
word_index  = tokenizer.word_index
print(word_index)


# In[17]:


train['original_lang_int'] = [word_index[j] for j in train.original_language]
test['original_lang_int'] = [word_index[j] for j in test.original_language]


# In[18]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([train.genres, test.genres], ignore_index=True))
word_index  = tokenizer.word_index
print(word_index)


# In[19]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([train.spoken_languages, test.spoken_languages], ignore_index=True))
word_index  = tokenizer.word_index
print(word_index)


# In[20]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([train.cast, test.cast], ignore_index=True))
word_index  = tokenizer.word_index
#print(word_index)


# In[21]:


train.head()


# In[22]:


def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 10 }
    )


# In[23]:


import seaborn as sns
plot_correlation_map(train.drop(columns=['id']))


# In[24]:


features = train.drop(columns=['id','revenue']).select_dtypes(include=[np.number]).columns.tolist()
features


# In[25]:


import seaborn as sns
#sns.pairplot(train.drop(columns=['id']))


# In[26]:


import math
features=['budget', 'homepage', 'popularity', 'runtime', 'in_collection', 'release_year', 'cast_len', 'crew_len', 'production_companies_len','revenue']
X_train = train.filter(features, axis=1)
X_test = test.filter(features, axis=1)
X_train = X_train[X_train.budget>10000]
X_train = X_train[X_train.runtime>10]
Y_train = X_train.filter(['revenue'], axis=1)

X_train['popularity'] = X_train['popularity'].apply(lambda x: math.log(1.0+x))
X_train['budget'] = X_train['budget'].apply(lambda x: math.log(1.0+x))
X_train['revenue'] = X_train['revenue'].apply(lambda x: math.log(1.0+x))

X_test['popularity'] = X_test['popularity'].apply(lambda x: math.log(1.0+x))
X_test['budget'] = X_test['budget'].apply(lambda x: math.log(1.0+x))

Y_train['revenue'] = Y_train['revenue'].apply(lambda x: math.log(1.0+x))


# In[27]:


plot_correlation_map(X_train)
sns.pairplot(X_train)


# In[28]:


X_train.describe()


# In[29]:


#training data
X_train_model = X_train.drop(columns=['revenue']).values
Y_train_model = Y_train.values

#test data with no y values!
X_test_model  = X_test.values

print("X_train data =", X_train_model.shape)
print("Y_train data =", Y_train_model.shape)
print("X_test data  =", X_test_model.shape)


# In[30]:


def PlotTraining(x,y):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.show()


# In[31]:


from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X_train_model,Y_train_model)
y_pred_svr = svr_rbf.predict(X_train_model)


# In[32]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_train_model, y_pred_svr))


# In[33]:


PlotTraining(Y_train_model,y_pred_svr)


# In[34]:


Y_test_pred_svr = svr_rbf.predict(X_test_model)


# In[35]:


Y_test_pred_svr


# In[36]:


Y_test_pred_svr = np.exp(Y_test_pred_svr)


# In[37]:


Y_test_pred_svr


# In[38]:


from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_model, Y_train_model)
y_pred_knn = knn.predict(X_train_model)


# In[39]:


print(mean_absolute_error(Y_train_model, y_pred_knn))


# In[40]:


PlotTraining(Y_train_model,y_pred_knn)


# In[41]:


Y_test_pred_knn = knn.predict(X_test_model)


# In[42]:


Y_test_pred_knn


# In[43]:


Y_test_pred_knn = np.exp(Y_test_pred_knn)


# In[44]:


Y_test_pred_knn


# In[45]:


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators = 500, max_depth=5)
gbr.fit(X_train_model,Y_train_model)
y_pred_gbr = gbr.predict(X_train_model)


# In[46]:


print(mean_absolute_error(Y_train_model, y_pred_gbr))


# In[47]:


PlotTraining(Y_train_model,y_pred_gbr)


# In[48]:


feature_import = pd.DataFrame(data=gbr.feature_importances_, index=X_train.drop(columns=['revenue']).columns.values, columns=['values'])
feature_import.sort_values(['values'], ascending=False, inplace=True)
feature_import.transpose()


# In[49]:


Y_test_pred_gbr = gbr.predict(X_test_model)


# In[50]:


Y_test_pred_gbr


# In[51]:


Y_test_pred_gbr = np.exp(Y_test_pred_gbr)


# In[52]:


Y_test_pred_gbr


# In[53]:


from sklearn.ensemble import RandomForestRegressor
rfg = RandomForestRegressor(n_estimators=500, max_features='sqrt', min_samples_split=4)
rfg.fit(X_train_model, Y_train_model)
y_pred_rfg = rfg.predict(X_train_model)


# In[54]:


print(mean_absolute_error(Y_train_model, y_pred_rfg))


# In[55]:


PlotTraining(Y_train_model,y_pred_rfg)


# In[56]:


feature_import = pd.DataFrame(data=rfg.feature_importances_, index=X_train.drop(columns=['revenue']).columns.values, columns=['values'])
feature_import.sort_values(['values'], ascending=False, inplace=True)
feature_import.transpose()


# In[57]:


from sklearn.tree import export_graphviz
import pydot
feature_list = X_train.drop(columns=['revenue']).columns.values
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 4, max_features='sqrt')
rf_small.fit(X_train_model, Y_train_model)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');


# In[58]:


Y_test_pred_rfg = rfg.predict(X_test_model)


# In[59]:


Y_test_pred_rfg


# In[60]:


Y_test_pred_rfg = np.exp(Y_test_pred_rfg)


# In[61]:


Y_test_pred_rfg


# In[62]:


test = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({
    "id" : test.id
})

submission['revenue'] = Y_test_pred_knn
submission.to_csv("submission_knn.csv", index=False)


# In[63]:


test = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({
    "id" : test.id
})

submission['revenue'] = Y_test_pred_rfg
submission.to_csv("submission_rfg.csv", index=False)


# In[64]:


test = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({
    "id" : test.id
})

submission['revenue'] = Y_test_pred_gbr
submission.to_csv("submission_gbr.csv", index=False)


# In[65]:


test = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({
    "id" : test.id
})

submission['revenue'] = Y_test_pred_svr
submission.to_csv("submission_svr.csv", index=False)


# In[66]:


test = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({
    "id" : test.id
})

submission['revenue'] = (Y_test_pred_rfg+Y_test_pred_knn.reshape(-1))/2
submission.to_csv("submission_rfg_knn.csv", index=False)


# In[67]:


submission.head()

