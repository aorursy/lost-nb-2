#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import skew, skewtest, norm
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelBinarizer
from scipy import sparse

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Ridge, Lasso, SGDRegressor, ElasticNet
from sklearn.metrics import  make_scorer,  mean_squared_error
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


# In[ ]:


Start_time = time.time()

#train_data = pd.read_csv('train.tsv', sep = "\t")   
#test_data = pd.read_csv('test.tsv', sep='\t')   
train_data = pd.read_csv('../input/train.tsv', sep='\t')
test_data = pd.read_csv('../input/test.tsv', sep='\t')


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


sns.distplot(train_data['price'], bins = 20, fit = norm)


# In[ ]:


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
sns.distplot(np.log(train_data['price'].values+1), axlabel = 'Log(price)', label = 'log(trip_duration)', bins = 50, color="y")
plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


# In[ ]:


train_data['price'] = np.log1p(train_data['price'])
train_data['shipping'] = np.log1p(train_data['shipping'])
test_data['shipping'] = np.log1p(test_data['shipping'])


# In[ ]:


y = train_data['price']


# In[ ]:


def if_catname(row):
    
    """function to give if category, brand or description name is there or not"""
    if row == row:
        return 0
    else:
        return 1
    
train_data['Category_missing'] = train_data.category_name.apply(lambda row : if_catname(row))
train_data['Brand_missing'] = train_data.brand_name.apply(lambda row : if_catname(row))
train_data['Item_missing'] = train_data.item_description.apply(lambda row : if_catname(row))
test_data['Category_missing'] = test_data.category_name.apply(lambda row : if_catname(row))
test_data['Brand_missing'] = test_data.brand_name.apply(lambda row : if_catname(row))
test_data['Item_missing'] = test_data.item_description.apply(lambda row : if_catname(row))


# In[ ]:


train_data.head()


# In[ ]:


train_data["category_name"].fillna("None/None/None", inplace=True)
test_data["category_name"].fillna("None/None/None", inplace=True)

train_data["brand_name"].fillna("None", inplace=True)
test_data["brand_name"].fillna("None", inplace=True)

train_data["item_description"].fillna("None", inplace=True)
test_data["item_description"].fillna("None", inplace=True)


# In[ ]:


ID_train = train_data['train_id']
ID_test = test_data['test_id']


# In[ ]:


train_data.drop("train_id", axis = 1, inplace = True)


test_data.drop("test_id", axis = 1, inplace = True)


train_data.drop("price", axis = 1, inplace = True)


# In[ ]:


print(train_data.shape)
print(test_data.shape)
ntrain = train_data.shape[0]
ntest = test_data.shape[0]


# In[ ]:


Combined_data = pd.concat([train_data,test_data]).reset_index(drop=True)


# In[ ]:


print("Combined size is : {}".format(Combined_data.shape))


# In[ ]:



categorical_features = Combined_data.select_dtypes(include = ["object"]).columns
numerical_features = Combined_data.select_dtypes(exclude = ["object"]).columns
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))


# In[ ]:


Combined_data_numerical = Combined_data[numerical_features]


# In[ ]:


categorical_features


# In[ ]:


count = CountVectorizer(min_df=10)
X_name = count.fit_transform(Combined_data['name'])
X_name.shape


# In[ ]:


vector_brand = LabelBinarizer(sparse_output=True)
X_brand = vector_brand.fit_transform(Combined_data['brand_name'])
X_brand.shape


# In[ ]:


vector_L0 = LabelBinarizer(sparse_output=True)
X_L0 = vector_brand.fit_transform(Combined_data['category_name'])
X_L0.shape


# In[ ]:


num_features = Combined_data_numerical.values
num_features.shape


# In[ ]:



tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=50000)
X_description = tfidf_vec.fit_transform(Combined_data['item_description'])
X_description.shape


# In[ ]:


X = sparse.hstack((X_name, X_brand, X_L0, num_features, X_description )).tocsr()
X.shape


# In[ ]:


train = X[:ntrain]
test = X[ntrain:]
train.shape


# In[ ]:


def evaluate_model(X, y, algorithm):
    
    X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.20, random_state = 1)
    
    print(algorithm)
    print()
    print('Train - Mean Squared Error')
    print((np.sqrt(-cross_val_score(algorithm, X_train, y_train, scoring="neg_mean_squared_error", cv = 2)).mean()))
    print()
    print('Test - Mean Squared Error')
    print((np.sqrt(-cross_val_score(algorithm, X_test, y_test, scoring="neg_mean_squared_error", cv = 2)).mean()))
    print()

    #pipe.fit(X_train, y_train)
    #y_train_pred = pipe.predict(X_train)
    #y_test_pred = pipe.predict(X_test)
       


# In[ ]:


#pipe = make_pipeline(ElasticNet())
#evaluate_model(train, y, pipe)
#pipe = make_pipeline(Ridge(solver='sag', alpha=0.02))
#evaluate_model(train, y, pipe)


# In[ ]:


ridge = Ridge(solver='sag', fit_intercept=True, alpha=0.02, max_iter=200, normalize=False, tol=0.01)
ridge.fit(train, y)


# In[ ]:


#elastic = ElasticNet(tol=0.01)
#elastic.fit(train, y)


# In[ ]:


labels_ridge = np.expm1(ridge.predict(test))


# In[ ]:


#labels_elastic = np.expm1(elastic.predict(test))


# In[ ]:


pd.DataFrame({'test_id': ID_test , 'price': labels_ridge}).to_csv('MercariPredictions.csv', index =False) 


# In[ ]:


end_time = time.time()


# In[ ]:


(end_time - Start_time) 


# In[ ]:




