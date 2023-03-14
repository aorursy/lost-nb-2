#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

import scipy as sp
import pandas as pd
import numpy as np

from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder, FunctionTransformer

from collections import Counter

import lightgbm as lgb

from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_pandas import DataFrameMapper
from nltk.stem.snowball import SnowballStemmer
np.random.seed(369)

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

class SmallModalityAsOthers(BaseEstimator, TransformerMixin):
    """Group small frequencies modality as an other modality

    Parameters
    ----
    cols: columns name to apply modality fgroup
    threshold: int - number threshold to switch a modality to "other"

    Attributes
    ----
    Return pandas dataframe with new modality "other" for given columns
    """
    def __init__(self, cols, threshold=10, other_name="other"):
        self.cols = cols
        self.threshold = threshold
        self.other_name = other_name

    def fit(self, df, y=None, **fit_params):
        self.modality_to_others = dict()
        for col in self.cols:
            table = df.loc[:, col].value_counts().reset_index()
            self.modality_to_others[col] =                list(table.loc[table.loc[:, col] < self.threshold, "index"])
        return self

    def transform(self, df, **transform_params):
        for col in self.cols:
            df.loc[df.loc[:, col].map(lambda x: x in self.modality_to_others[col]),
                   col] =\
                self.other_name
        return df
    
class NumberOfRowByValue(BaseEstimator, TransformerMixin):
    """Count number of rows for a specific categorical variable

    Parameters
    ----
    col_groupby: column  name to groupby from
    
    Attributes
    ----
    Return pandas dataframe with news column. 
    Number of rows which take this specific categorical variable in training
    """
    def __init__(self, col_groupby, new_col_name="n_row_by_category"):
        self.col_groupby = col_groupby
        self.new_col_name = new_col_name
    def fit(self, df, y=None, **fit_params):
        self.n_row =            df.groupby(self.col_groupby).size().reset_index()
        self.n_row.columns = [self.col_groupby, self.new_col_name]
        return self
    def transform(self, df, **transform_params):
        df =            df.merge(self.n_row, how="left", on=self.col_groupby)
        return df

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d

def _get_unseen():
    """Basically just a static method
    instead of a class attribute to avoid
    someone accidentally changing it."""
    return 99999


class SafeLabelEncoder(LabelEncoder):
    """An extension of LabelEncoder that will
    not throw an exception for unseen data, but will
    instead return a default value of 99999

    Attributes
    ----------

    classes_ : the classes that are encoded
    """

    def transform(self, y):
        """Perform encoding if already fit.

        Parameters
        ----------

        y : array_like, shape=(n_samples,)
            The array to encode

        Returns
        -------

        e : array_like, shape=(n_samples,)
            The encoded array
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        classes = np.unique(y)
        # _check_numpy_unicode_bug(classes)

        # Check not too many:
        unseen = _get_unseen()
        if len(classes) >= unseen:
            raise ValueError('Too many factor levels in feature. Max is %i' % unseen)

        e = np.array([
                         np.searchsorted(self.classes_, x) if x in self.classes_ else unseen
                         for x in y
                         ])

        return e
    
class CrossFeatures(BaseEstimator, TransformerMixin):
    """ Create new column as multiplication between two columns
    Attributes
    ----------
    cols_tuple: list of string tuple (a, b) such as result will be a * b

    """
    def __init__(self, cols_tuple=None):
        self.cols_tuple = cols_tuple
    def fit(self, df, y=None, **fit_params):
        return self
    def transform(self, df, **transform_params):
        for a, b in self.cols_tuple:
            df.loc[:, str(a) + "_MULTIPLICATED_BY_" + str(b)] =                pd.to_numeric(df.loc[:, a], errors='coerce') * pd.to_numeric(df.loc[:, b], errors='coerce')
        return df
    
class DivBetweenCols(BaseEstimator, TransformerMixin):
    """ Create new column as division between two columns
    Attributes
    ----------
    cols_tuple: list of string tuple (a, b) such as result will be a / b

    """
    def __init__(self, cols_tuple=None):
        self.cols_tuple = cols_tuple
    def fit(self, df, y=None, **fit_params):
        return self
    def transform(self, df, **transform_params):
        for a, b in self.cols_tuple:
            df.loc[:, str(a) + "_DIVIDED_BY_" + str(b)] =                pd.to_numeric(df.loc[:, a], errors='coerce') / pd.to_numeric(df.loc[:, b], errors='coerce')
        return df
    
class ColumnsSelector(BaseEstimator, TransformerMixin):
    """ Create new Dataframe with columns selected
    Attributes
    ----------
    colnames_list: list of string - columns name

    """
    def __init__(self, colnames_list=None):
        self.colnames_list = colnames_list
    def fit(self, df, y=None, **fit_params):
        return self
    def transform(self, df, **transform_params):
        if len(self.colnames_list) == 1:
            return(df.loc[:, self.colnames_list[0]])
        else:
            return df.loc[:, self.colnames_list]
    
class MeanYByCategories(BaseEstimator, TransformerMixin):
    """For columns that should not accept negative value:
        if negative value is found then value become nan

    Parameters
    ----
    cols: list of column names that sould not accept negative values

    Attributes
    ----
    Return pandas dataframe with cols columns contain missing value if value are below 0
    """
    def __init__(self, col, new_col_name):
        self.col = col
        self.new_col_name = new_col_name
    def fit(self, df, y=None, **fit_params):
        full_df = pd.concat([df.reset_index(drop=True),
                             pd.Series(y).reset_index(drop=True)], axis=1, ignore_index=True)
        full_df.columns = list(df.columns) + ["y_target"]
        self.median_by_cat =            full_df.groupby(self.col)["y_target"].mean().reset_index()
        return self
    def transform(self, df, **transform_params):
        df =            df.merge(self.median_by_cat, how="left", on=self.col)
        df.drop(self.col, axis=1, inplace=True)
        df = df.rename(columns={"y_target":self.new_col_name})
        return df
    

class CreateGroupByFeature(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----
    cols: list of column names to compute mean
    by: column to groupby

    Attributes
    ----
    Return pandas dataframe wnew columns as average values
    """
    def __init__(self, col, by, percentile=False):
        self.col = col
        self.by = by
        self.percentile = percentile
    def fit(self, df, y=None, **fit_params):
        operations = ["max", "mean", "std"]
        self.median_by_cat =            df.groupby(self.by)[self.col].agg(operations).reset_index()
        self.median_by_cat.columns =            [self.by] + [x + "_" + y + "_" + self.by for x in self.col for y in operations]
        if self.percentile:
            self.percentile_table =                df.groupby(self.by)[self.col].quantile([0.1, 0.25, 0.75, 0.9]).reset_index()
            self.percentile_table =                self.percentile_table.pivot(self.by, columns="level_1")
            self.percentile_table.columns = [x + str(y) + "_by_" + self.by for x in self.col for y in [0.1, 0.25, 0.75, 0.9]]
        return self
    def transform(self, df, **transform_params):
        df =            df.merge(self.median_by_cat, how="left", on=self.by)
        # df.drop(self.by, axis=1, inplace=True)
        if self.percentile:
            df =                df.merge(self.percentile_table, how="left", on=self.by)
        # df = df.rename(columns={"y_target":self.new_col_name})
        return df



# In[2]:


print('Train')
train = pd.read_csv("../input/train/train.csv")
print(train.shape)

print('Test')
test = pd.read_csv("../input/test/test.csv")
print(test.shape)

print('Breeds')
breeds = pd.read_csv("../input/breed_labels.csv")
print(breeds.shape)

print('Colors')
colors = pd.read_csv("../input/color_labels.csv")
print(colors.shape)

print('States')
states = pd.read_csv("../input/state_labels.csv")
print(states.shape)

target = train['AdoptionSpeed']
train_id = train['PetID']
test_id = test['PetID']
train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)
test.drop(['PetID'], axis=1, inplace=True)


# In[3]:


def name_size(name):
    if pd.notnull(name):
        return(len(name))
    else:
        return(0)
    
def n_words_in_name(name):
    if pd.notnull(name):
        return(len(name.split(" ")))
    else:
        return(0)
    
def hasNumbers(inputString):
    if pd.notnull(inputString):
        return int(any(char.isdigit() for char in inputString))
    else:
        return(0)

def n_punctuation(inputString):
    if pd.notnull(inputString):
        return sum([j in string.punctuation for j in inputString])
    else:
        return(0)
    
COLUMNS_TO_DROP = ["RescuerID"]
CATEGORICAL_FEATURES = ["Gender", "Vaccinated", "Dewormed", 
                        "Sterilized", "State",
                        "Name", "MaturitySize", "Type",
                        "Health"]

COLUMNS_WITH_SMALL_MODALITY =    ["Breed1", "Breed2", "Color1", "Color2", "Color3", "Name"]

# Create transformers instance
to_other = SmallModalityAsOthers(cols=COLUMNS_WITH_SMALL_MODALITY, other_name=-1, threshold=30)
name_frequencies = NumberOfRowByValue(col_groupby="Name", new_col_name="name_frequency")
cross_feature = CrossFeatures(cols_tuple=[("Age", "Health")])
div_features = DivBetweenCols(cols_tuple=[("Age", "Health")])
median_Y = MeanYByCategories(col="Type", new_col_name="Type_y_frequencies")
median_Y_by_breed1 = MeanYByCategories(col="Breed1", new_col_name="Breed1")
median_Y_by_breed2 = MeanYByCategories(col="Breed2", new_col_name="Breed2")
median_Y_by_color1 = MeanYByCategories(col="Color1", new_col_name="Color1")
median_Y_by_color2 = MeanYByCategories(col="Color2", new_col_name="Color2")
median_Y_by_color3 = MeanYByCategories(col="Color3", new_col_name="Color3")
grouper = CreateGroupByFeature(col=["Fee", "Quantity"], by="Breed1", percentile=False)
grouper2 = CreateGroupByFeature(col=["Fee", "Quantity"], by="Breed2", percentile=False)
stemmer = SnowballStemmer("english")

all_data = [train, test]

for i in range(len(all_data)):
    # Name size
    all_data[i].loc[:, "name_size"] =        all_data[i].Name.map(name_size)
    
    all_data[i].loc[:, "n_words_in_name"] =        all_data[i].Name.map(n_words_in_name)
    
    all_data[i].loc[:, "name_contain_number"] =        all_data[i].Name.map(hasNumbers)
    
    all_data[i].loc[:, "n_punctuations_in_name"] =        all_data[i].Name.map(n_punctuation)
    
    all_data[i].loc[:, "photo_and_video"] =        all_data[i].PhotoAmt + all_data[i].VideoAmt
   

all_data[0] = median_Y.fit_transform(all_data[0], target)
all_data[1] = median_Y.transform(all_data[1])

all_data[0] = median_Y_by_breed1.fit_transform(all_data[0], target)
all_data[1] = median_Y_by_breed1.transform(all_data[1])
all_data[0] = median_Y_by_breed2.fit_transform(all_data[0], target)
all_data[1] = median_Y_by_breed2.transform(all_data[1])
all_data[0] = median_Y_by_color1.fit_transform(all_data[0], target)
all_data[1] = median_Y_by_color1.transform(all_data[1])
all_data[0] = median_Y_by_color2.fit_transform(all_data[0], target)
all_data[1] = median_Y_by_color2.transform(all_data[1])
all_data[0] = median_Y_by_color3.fit_transform(all_data[0], target)
all_data[1] = median_Y_by_color3.transform(all_data[1])
        
all_data[0] = name_frequencies.fit_transform(all_data[0])
all_data[1] = name_frequencies.transform(all_data[1])
all_data[1].name_frequency = all_data[1].name_frequency.fillna(0)
all_data[0].name_frequency = all_data[0].name_frequency.fillna(0)

# Label encode Name
name_le = SafeLabelEncoder()
all_data[0].Name = name_le.fit_transform(all_data[0].Name.fillna("missing"))
all_data[1].Name = name_le.transform(all_data[1].Name.fillna("missing"))

all_data[0] = to_other.fit_transform(all_data[0])
all_data[1] = to_other.transform(all_data[1])

# cross and div feature
all_data[0] = cross_feature.fit_transform(all_data[0])
all_data[1] = cross_feature.transform(all_data[1])

all_data[0] = div_features.fit_transform(all_data[0])
all_data[1] = div_features.transform(all_data[1])

all_data[0] = all_data[0].drop(COLUMNS_TO_DROP, axis=1)
all_data[1] = all_data[1].drop(COLUMNS_TO_DROP, axis=1)

all_data[0] = grouper.fit_transform(all_data[0])
all_data[1] = grouper.transform(all_data[1])
all_data[0] = grouper2.fit_transform(all_data[0])
all_data[1] = grouper2.transform(all_data[1])

for i in range(len(all_data)):
    all_data[i].loc[:, "relative_fee"] =        all_data[i].Fee / all_data[i].Fee_mean_Breed1
    all_data[i].loc[:, "relative_fee2"] =        all_data[i].Fee / all_data[i].Fee_mean_Breed2
    all_data[i].loc[:, "relative_Quantity"] =        all_data[i].Fee / all_data[i].Quantity_mean_Breed1
    all_data[i].loc[:, "relative_Quantity2"] =        all_data[i].Fee / all_data[i].Quantity_mean_Breed2
    
    all_data[i].Description = all_data[i].Description.fillna("None")
    all_data[i].Description =        all_data[i].Description.map(lambda x: stemmer.stem(x))

train = all_data[0]
test = all_data[1]


# In[4]:


# lgbm = lgb.LGBMClassifier(categorical_features=CATEGORICAL_FEATURES)
lgbm = lgb.LGBMClassifier(categorical_features=CATEGORICAL_FEATURES)
description_selector = ColumnsSelector(colnames_list=["Description"])
nodescription_selector =    ColumnsSelector(colnames_list=[x for x in train.columns if x !="Description"])
tf_idf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                        norm="l1", max_df=0.9, min_df=0.05)
lda = LatentDirichletAllocation(n_jobs=4)

pipe = Pipeline([("union", FeatureUnion([
                ('tf_idf', 
                  Pipeline([('extract_field',
                              description_selector),
                            ('tfidf', 
                              tf_idf),
                           ("lda", lda)])),
                ('no_tfidf',
                  nodescription_selector)])),
                 ("classifier", lgbm)])

# pipe = Pipeline([("classifier", lgbm)])

RECOMPUTE_BEST_PARAMS = True

# Find best param with random search
param_dist = {"classifier__num_leaves": sp_randint(2, 20),
              "classifier__learning_rate": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "classifier__max_bin": sp_randint(13, 23),
              "classifier__bagging_freq": [20, 21, 22, 23, 24, 25, 26, 27],
              "classifier__max_depth": sp_randint(100, 250),
              "classifier__feature_fraction": [0.7, 0.8, 0.9, 1],
             "classifier__n_estimators":sp_randint(150, 800),
             "classifier__reg_alpha":[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

best_params = {'classifier__bagging_freq': 26,
             'classifier__feature_fraction': 0.9,
             'classifier__learning_rate': 0.3,
             'classifier__max_bin': 21,
             'classifier__max_depth': 219,
             'classifier__n_estimators': 304,
             'classifier__num_leaves': 3,
             'classifier__reg_alpha': 0.6}

if RECOMPUTE_BEST_PARAMS:
    n_iter_search = 20
    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist,
                                        n_iter=n_iter_search, cv=5,
                                      scoring="neg_log_loss",
                                      n_jobs=4)

    random_search.fit(train, target)
    
    best_params = random_search.best_params_


# In[5]:


# affect best params
pipe = pipe.set_params(**best_params)

# k-fold to evaluate
kf = ShuffleSplit(n_splits=5, test_size=0.20, random_state=50)

qwk_train_list = []
qwk_test_list = []
pipe_list = []
for train_index, test_index in kf.split(train):
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    if RECOMPUTE_BEST_PARAMS:
        random_search.fit(X_train, y_train)
        train_predictions = random_search.predict(X_train)
        test_predictions = random_search.predict(X_test)
        pipe_list.append(random_search)
    else:
        pipe.fit(X_train, y_train)

        train_predictions = pipe.predict(X_train)
        test_predictions = pipe.predict(X_test)
        pipe_list.append(pipe)
    
    qwk_train = quadratic_weighted_kappa(y_train, train_predictions)
    qwk_test = quadratic_weighted_kappa(y_test, test_predictions)
    
    print("QWK train = " + str(qwk_train))
    print("QWK test = " + str(qwk_test))
    
    qwk_train_list = qwk_train_list + [qwk_train]
    qwk_test_list = qwk_test_list + [qwk_test]
    
print("Average of QWK train = " + str(sum(qwk_train_list) / len(qwk_train_list)))
print("Average of QWK test = " + str(sum(qwk_test_list) / len(qwk_test_list)))


# In[6]:


import matplotlib.pyplot as plt

if RECOMPUTE_BEST_PARAMS == False:
    feat_imp =        pd.concat([pd.Series(["lda_component" + str(i) for i in range(1, 10)] + 
                             list(train.columns)), 
                   pd.Series(pipe.named_steps["classifier"].feature_importances_)], axis=1)

    feat_imp.columns = ["var", "importance"]

    feat_imp = feat_imp.sort_values(by="importance", ascending=False)

    feat_imp.plot.bar(x="var", y="importance")


# In[7]:


test_predictions_proba = pipe_list[0].predict_proba(test)
for temp_pipe in pipe_list[1:]:
    # temp_pipe.fit(train, target)
    test_predictions_proba = test_predictions_proba + temp_pipe.predict_proba(test)

test_predictions_proba = test_predictions_proba / 5
test_predictions = np.apply_along_axis(np.argmax, arr=test_predictions_proba, axis=1)

submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
submission.head()


# In[8]:


submission.to_csv('submission.csv', index=False)


# In[9]:




