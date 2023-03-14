#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_csv("../input/train.tsv", delimiter = '\t', nrows=10000)
print ("train.shape: " + str(train.shape))
train.head()


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10, 6))
plt.hist(train["price"])
plt.xlabel("price[$]")
plt.ylabel("count")
plt.title("Price Histogram")
plt.show()


# In[3]:


plt.figure(figsize=(10, 6))
plt.hist(train["price"], bins=500)
plt.xlim(0, 200)
plt.xlabel("price[$]")
plt.ylabel("count")
plt.title("Price Histogram")
plt.show()


# In[4]:


import time
import seaborn as sns
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
train["general_cat"], train["sub_cat1"], train["sub_cat2"] =     zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.drop("category_name", axis=1)
start_time = time.time()
plt.figure(figsize=(16, 8))
ax = sns.violinplot(x="general_cat", y="price", data=train, inner=None)
ax = sns.swarmplot(x="general_cat", y="price", data=train, edgecolor="gray", hue="sub_cat1")
plt.xticks(rotation=30)
plt.ylim(0, 200)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.25), ncol=5)
plt.show()
print('Showing graph took {} secs.'.format(time.time() - start_time))


# In[5]:


train.sort_values(by="price", ascending=False).head(1)


# In[6]:


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['sub_cat1'].fillna(value='missing', inplace=True)
    dataset['sub_cat2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)
handle_missing_inplace(train)
train['brand_name'].value_counts().head()


# In[7]:


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:750]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:450]
    pop_category2 = dataset['sub_cat1'].value_counts().loc[lambda x: x.index != 'missing'].index[:450]
    pop_category3 = dataset['sub_cat2'].value_counts().loc[lambda x: x.index != 'missing'].index[:450]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['sub_cat1'].isin(pop_category2), 'sub_cat1'] = 'missing'
    dataset.loc[~dataset['sub_cat2'].isin(pop_category3), 'sub_cat2'] = 'missing'
cutting(train)
train['brand_name'].value_counts().head()


# In[8]:


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['sub_cat1'] = dataset['sub_cat1'].astype('category')
    dataset['sub_cat2'] = dataset['sub_cat2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
to_categorical(train)
train.dtypes


# In[9]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=5)
X_name = cv.fit_transform(train['name'])
print (X_name.shape)
occ = np.asarray(X_name.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': cv.get_feature_names(), 'occurrences': occ})
counts_cv.sort_values(by='occurrences', ascending=False).head(10)


# In[10]:


cv = CountVectorizer(min_df=5)
combine_category = [train["general_cat"], train["sub_cat1"], train["sub_cat2"]]
X_category1 = cv.fit_transform(train['general_cat'])
print ("----general_cat----")
print (X_category1.shape)
occ = np.asarray(X_category1.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': cv.get_feature_names(), 'occurrences': occ})
print (counts_cv.sort_values(by='occurrences', ascending=False).head())
X_category2 = cv.fit_transform(train['sub_cat1'])
print ("----sub_cat1----")
print (X_category2.shape)
occ = np.asarray(X_category2.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': cv.get_feature_names(), 'occurrences': occ})
print (counts_cv.sort_values(by='occurrences', ascending=False).head())
X_category3 = cv.fit_transform(train['sub_cat2'])
print ("----sub_cat2----")
print (X_category3.shape)
occ = np.asarray(X_category3.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': cv.get_feature_names(), 'occurrences': occ})
print (counts_cv.sort_values(by='occurrences', ascending=False).head())


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=1000,
                         ngram_range=(1, 3),
                         stop_words='english')
X_description = tv.fit_transform(train['item_description'])
weights = np.asarray(X_description.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': tv.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(10)


# In[12]:


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(train['brand_name'])
occ = np.asarray(X_brand.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': lb.classes_, 'occurrences': occ})
counts_cv.sort_values(by='occurrences', ascending=False).head(10)


# In[13]:


from scipy.sparse import csr_matrix
X_dummies = csr_matrix(pd.get_dummies(train[['item_condition_id', 'shipping']],
                                          sparse=True).values)
X_dummies


# In[14]:


from scipy.sparse import hstack
sparse_merge = hstack([X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name]).tocsr()
sparse_merge.shape


# In[15]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = sparse_merge
y = np.log1p(train["price"])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 144) 

modelR = Ridge(alpha=.5, copy_X=True, fit_intercept=True, max_iter=100,
      normalize=False, random_state=101, solver='auto', tol=0.01)
modelR.fit(train_X, train_y)
predsR = modelR.predict(test_X)

def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

rmsleR = rmsle(predsR, test_y)
print ("Ridge Regression RMSLE = " + str(rmsleR))


# In[16]:


import lightgbm as lgb

train_XL1, valid_XL1, train_yL1, valid_yL1 = train_test_split(train_X, train_y, test_size = 0.1, random_state = 144) 
d_trainL1 = lgb.Dataset(train_XL1, label=train_yL1, max_bin=8192)
d_validL1 = lgb.Dataset(valid_XL1, label=valid_yL1, max_bin=8192)
watchlistL1 = [d_trainL1, d_validL1]
paramsL1 = {
        'learning_rate': 0.65,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4
    }
modelL1 = lgb.train(paramsL1, train_set=d_trainL1, num_boost_round=8000, valid_sets=watchlistL1, early_stopping_rounds=5000, verbose_eval=500) 
predsL1 = modelL1.predict(test_X)
rmsleL1 = rmsle(predsL1, test_y)
print ("LightGBM1 RMSLE = " + str(rmsleL1))


# In[17]:


train_XL2, valid_XL2, train_yL2, valid_yL2 = train_test_split(train_X, train_y, test_size = 0.1, random_state = 101) 
d_trainL2 = lgb.Dataset(train_XL2, label=train_yL2, max_bin=8192)
d_validL2 = lgb.Dataset(valid_XL2, label=valid_yL2, max_bin=8192)
watchlistL2 = [d_trainL2, d_validL2]
paramsL2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 140,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': 4
    }
modelL2 = lgb.train(paramsL2, train_set=d_trainL2, num_boost_round=5500, valid_sets=watchlistL2, early_stopping_rounds=5000, verbose_eval=500) 
predsL2 = modelL2.predict(test_X)
rmsleL2 = rmsle(predsL2, test_y)
print ("LightGBM2 RMSLE = " + str(rmsleL2))


# In[18]:


preds = predsR*0.3 + predsL1*0.35 + predsL2*0.35
rmsle = rmsle(preds, test_y)
print ("Total RMSLE = " + str(rmsle))


# In[19]:


actual_price = np.expm1(test_y)
preds_price = np.expm1(preds)

plt.figure(figsize=(12,10))
cm = plt.cm.get_cmap('winter')
x_diff = np.clip(100 * ((preds_price - actual_price) / actual_price), -75, 75)
plt.scatter(x=actual_price, y=preds_price, c=x_diff, s=10, cmap=cm)
plt.colorbar()
plt.plot([0, 100], [0, 100], 'k--', lw=1)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Prices [$]')
plt.ylabel('Predicted Prices [$]')
plt.show()

