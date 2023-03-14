#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import json

import scipy as sp
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
tqdm.pandas()
import gc
gc.collect()
from functools import partial
from math import sqrt
from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from collections import Counter

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from collections import Counter

import lightgbm as lgb
np.random.seed(369)


# In[ ]:


df_train = pd.read_csv('../input/train/train.csv')
df_breed = pd.read_csv('../input/breed_labels.csv')
df_color = pd.read_csv('../input/color_labels.csv')
df_state = pd.read_csv('../input/state_labels.csv')
df_test = pd.read_csv('../input/test/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.isna().sum()


# In[ ]:


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings
def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


# In[ ]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[ ]:


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# In[ ]:


train_desc = df_train.Description.fillna("none").values
test_desc = df_test.Description.fillna("none").values

svd_n_components = 200

tfv = TfidfVectorizer(min_df=2,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        )
    
# Fit TFIDF
tfv.fit(list(train_desc))
X =  tfv.transform(train_desc)
X_test = tfv.transform(test_desc)

svd = TruncatedSVD(n_components=svd_n_components)
svd.fit(X)
print(svd.explained_variance_ratio_.sum())
print(svd.explained_variance_ratio_)
X = svd.transform(X)
X = pd.DataFrame(X, columns=['svd_{}'.format(i) for i in range(svd_n_components)])
df_train = pd.concat((df_train, X), axis=1)
X_test = svd.transform(X_test)
X_test = pd.DataFrame(X_test, columns=['svd_{}'.format(i) for i in range(svd_n_components)])
df_test = pd.concat((df_test, X_test), axis=1)


# In[ ]:


df_test.shape


# In[ ]:


train_sentiment_path = '../input/train_sentiment/'
test_sentiment_path = '../input/test_sentiment/'
train_meta_path = '../input/train_metadata/'
test_meta_path = '../input/test_metadata/'


# In[ ]:


def get_sentiment(pet_id, json_dir):
    try:
        with open(json_dir + pet_id + '.json') as f:
            data = json.load(f)
        return pd.Series((data['documentSentiment']['magnitude'], data['documentSentiment']['score']))
    except FileNotFoundError:
        return pd.Series((np.nan, np.nan))


# In[ ]:


df_train[['desc_magnitude', 'desc_score']] = df_train['PetID'].progress_apply(lambda x: get_sentiment(x, train_sentiment_path))
df_test[['desc_magnitude', 'desc_score']] = df_test['PetID'].progress_apply(lambda x: get_sentiment(x, test_sentiment_path))


# In[ ]:


#df_train['Name'].fillna(0, inplace=True)
#df_train['Name'] = np.where(df_train.Name == 'No Name Yet', 0, 1)
#df_test['Name'].fillna(0, inplace=True)
#df_test['Name'] = np.where(df_test.Name == 'No Name Yet', 0, 1)


# In[ ]:


df_train.head()


# In[ ]:


print(df_train.shape, df_test.shape)


# In[ ]:


df_train.isna().sum()


# In[ ]:


train_petid = df_train['PetID']
test_petid = df_test['PetID']


# In[ ]:


#including rescuer id
#df_train.drop(['Name','Description', 'PetID'], axis=1, inplace=True)
#df_test.drop(['Name','Description',  'PetID'], axis=1, inplace=True)

df_train.drop(['Name','Description', 'RescuerID', 'PetID'], axis=1, inplace=True)
df_test.drop(['Name','Description', 'RescuerID', 'PetID'], axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


magnitude_std = df_train.desc_magnitude.std()
magnitude_mean = df_train.desc_magnitude.mean()
score_std = df_train.desc_score.std()
score_mean = df_train.desc_score.mean()
df_train['desc_magnitude'].fillna(np.random.normal(magnitude_mean, magnitude_std), inplace=True)
df_train['desc_score'].fillna( np.random.normal(score_mean, score_std), inplace=True)
df_test['desc_magnitude'].fillna(np.random.normal(magnitude_mean, magnitude_std), inplace=True)
df_test['desc_score'].fillna(np.random.normal(score_mean, score_std), inplace=True)


# In[ ]:


#category_columns = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']
#numerical_columns = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']


# In[ ]:


#df_train[category_columns] = df_train[category_columns].astype('category')
#df_train['AdoptionSpeed'] = df_train['AdoptionSpeed'].astype('category')
#df_test[category_columns] = df_test[category_columns].astype('category')


# In[ ]:


#df_train[numerical_columns] = df_train[numerical_columns].astype('float64')
#df_test[numerical_columns] = df_test[numerical_columns].astype('float64')


# In[ ]:


df_train.info()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df_train.drop(['AdoptionSpeed'], axis=1)
y = df_train.AdoptionSpeed


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from pandas.api.types import is_string_dtype, is_numeric_dtype
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def apply_cats(df, trn):
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)


# In[ ]:


#train_cats(X_train)
#apply_cats(X_val,X_train)
model_rf = RandomForestClassifier()
model_rf.fit(X_train,y_train)
model_rf.score(X_train,y_train)


# In[ ]:


model_rf.score(X_val,y_val)


# In[ ]:


print(model_rf.predict(X_train))


# In[ ]:


def score(model,X,y):
    optR = OptimizedRounder()
    optR.fit(model.predict(X), y)
    coefficients = optR.coefficients()
    pred_y_k = optR.predict(model.predict(X), coefficients)
    print("Valid Counts = ", Counter(y))
    print("Predicted Counts = ", Counter(pred_y_k))
    print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(y, pred_y_k)
    print("QWK = ", qwk)
    print("RMSE = ",rmse(y, model.predict(X)))
    print("mean Accuracy: ", model.score(X,y))


# In[ ]:


print('Train score: ', score(model_rf,X_train,y_train))


# In[ ]:


print('valid scoe: ', score(model_rf,X_val,y_val))


# In[ ]:


model_rf = RandomForestClassifier(n_estimators=20,n_jobs=-1)
model_rf.fit(X_train,y_train)


# In[ ]:


print('Train score: ', score(model_rf,X_train,y_train))
print('valid scoe: ', score(model_rf,X_val,y_val))


# In[ ]:


model_rf = RandomForestClassifier(n_estimators=40,n_jobs=-1)
model_rf.fit(X_train,y_train)
print('Train score: ', score(model_rf,X_train,y_train))
print('valid scoe: ', score(model_rf,X_val,y_val))


# In[ ]:


model_rf = RandomForestClassifier(n_estimators=80,n_jobs=-1)
model_rf.fit(X_train,y_train)
print('Train score: ', score(model_rf,X_train,y_train))
print('valid scoe: ', score(model_rf,X_val,y_val))


# In[ ]:


model_rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
model_rf.fit(X_train,y_train)
print('Train score: ', score(model_rf,X_train,y_train))
print('valid scoe: ', score(model_rf,X_val,y_val))


# In[ ]:





# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
features = X_train.columns
importances = model_rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
#table.sort_values('indices')


# In[ ]:


model_gb = GradientBoostingClassifier(n_estimators=100,)
model_gb.fit(X_train,y_train)
print('Train score: ', score(model_gb,X_train,y_train))
print('valid scoe: ', score(model_gb,X_val,y_val))


# In[ ]:


model_gb = GradientBoostingClassifier(n_estimators=150)
model_gb.fit(X_train,y_train)
print('Train score: ', score(model_gb,X_train,y_train))
print('valid scoe: ', score(model_gb,X_val,y_val))


# In[ ]:


#training on whole data set
model_gb = GradientBoostingClassifier(n_estimators=180)
model_gb.fit(X,y)
print('Train score: ', score(model_gb,X,y))
print('Train score: ', score(model_gb,X_train,y_train))
print('valid scoe: ', score(model_gb,X_val,y_val))


# In[ ]:


X_train.head()


# In[ ]:


#model_gb.predict_log_proba(X_train)


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
features = X_train.columns
importances = model_gb.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


#XGBoost
model_xgb = XGBClassifier(learning_rate=0.09,n_estimators=150,objective='multi:softprob',n_jobs=-1,eval_metric= 'mlogloss',gamma=0.2)
model_xgb.fit(X_train, y_train)
print('Train score: ', score(model_xgb,X_train,y_train))
print('valid score: ', score(model_xgb,X_val,y_val))


# In[ ]:





# In[ ]:


import lightgbm as lgbm
params_lgbm = {'num_leaves': 25,
         'min_data_in_leaf': 90, 
         'objective':'multiclass',
         'num_class': 5,
         'max_depth': 9,
         'learning_rate': 0.03,
         "boosting": "gbdt",
         "feature_fraction": 0.9980062052116254,
         "bagging_freq": 1,
         "bagging_fraction": 0.844212672233457,
         "bagging_seed": 11,
         "metric": 'multi_logloss',
         "lambda_l1": 0.12757257166471625,
         "random_state": 133,
         "verbosity": -1
              }


# In[ ]:


lgbm_train = lgbm.Dataset(X_train, y_train, categorical_feature=category_columns)
lgbm_valid = lgbm.Dataset(X_val, y_val, categorical_feature=category_columns)


# In[ ]:


model_lgbm = lgbm.train(params_lgbm, lgbm_train, 50000, valid_sets=[lgbm_valid],  verbose_eval= 5000, categorical_feature=category_columns, early_stopping_rounds = 500)


# In[ ]:


print((np.argmax(model_lgbm.predict(X_train), axis=1) == y_train).sum() / y_train.shape[0])
print((np.argmax(model_lgbm.predict(X_val), axis=1) == y_val).sum() / y_val.shape[0])


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
features = X_train.columns
importances = model_lgbm.feature_importance()
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


# Checking result of two best models
val_lgbm = model_lgbm.predict(X_val)
val_gb = model_gb.predict_log_proba(X_val)
val_mixed = (val_gb + val_lgbm) / 2
(np.argmax(val_mixed, axis=1) == y_val).sum() / y_val.shape[0]


# In[ ]:


test_pred = model_xgb.predict(df_test) # other than lgbm
#test_pred =np.argmax(model_lgbm.predict(df_test), axis=1)  # for lgbm
#test_pred = np.argmax((model_lgbm.predict(df_test) + model_gb.predict_log_proba(df_test))/2, axis=1) # mean of 2 best models
test_petid = pd.DataFrame(test_petid)
submission = test_petid.join(pd.DataFrame(test_pred, columns=['AdoptionSpeed']))
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:




