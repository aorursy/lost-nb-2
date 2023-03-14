#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import json
from gc import collect
from functools import partial
from collections import Counter, defaultdict
from math import sqrt
from operator import itemgetter

from joblib import Parallel, delayed

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import dill
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix

import lightgbm as lgb

kernel = True
nthread = 6


# In[2]:


# load data
train_df = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv", encoding="utf-8")
test_df = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv", encoding="utf-8")

breed_labels_df = pd.read_csv("../input/petfinder-adoption-prediction/breed_labels.csv")
breed_labels_df = pd.concat([pd.DataFrame([{"BreedID": 0, "Type": 0, "BreedName": "None"}]), breed_labels_df])
color_labels_df = pd.read_csv("../input/petfinder-adoption-prediction/color_labels.csv")
state_labels_df = pd.read_csv("../input/petfinder-adoption-prediction/state_labels.csv")

smpsb_df = pd.read_csv("../input/petfinder-adoption-prediction/test/sample_submission.csv")


# In[3]:


# Since train and test are splited by RescuerID, I used GroupKFold for validation.
petid_map = {v: i for i, v in enumerate(pd.concat([train_df["PetID"], test_df["PetID"]]))}
rescuerid_encoder = LabelEncoder().fit(pd.concat([train_df["RescuerID"], test_df["RescuerID"]]))

for group, (_, group_idx) in enumerate(GroupKFold(n_splits=10).split(train_df,
                                                                     train_df["AdoptionSpeed"],
                                                                     rescuerid_encoder.transform(train_df["RescuerID"]))):
    train_df.loc[group_idx, "group"] = group


# In[4]:


# metrix

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

def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

from copy import deepcopy
param = {'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread":nthread,
         "verbosity": -1}

def make_lgb_oofs(X_train, y_train, group, X_test, params, repeat=1, seedkeys=["bagging_seed", "seed"]):
    #folds = StratifiedKFold(n_splits=5, random_state=2434, shuffle=True)
    params = deepcopy(params)
    train_oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))

    for j in range(repeat):
        for key in seedkeys:
            params[key] = 2434 + j
        for i in range(5):
            dev_idx = np.where((group//2) != i)[0]
            val_idx = np.where((group//2) == i)[0]
            dev_data = lgb.Dataset(X_train[dev_idx], label=y_train[dev_idx])
            val_data = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])

            num_rounds = 10000
            clf = lgb.train(params,
                            dev_data,
                            num_rounds,
                            valid_sets=[dev_data, val_data],
                            verbose_eval=100,
                            early_stopping_rounds=200)
            train_oof[val_idx] += clf.predict(X_train[val_idx]) / repeat
            test_pred += clf.predict(X_test) / 5 / repeat
        

    return train_oof, test_pred


# In[5]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights = 'quadratic')
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = np.percentile(X, [2.73, 23.3, 50.3, 72]) # <= keypoint
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')
    
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds
    
    def coefficients(self):
        return self.coef_['x']


# In[6]:


# [2.73, 23.3, 50.3, 72] is
(np.add.accumulate(np.bincount(train_df.AdoptionSpeed)) / train_df.shape[0])[:4] * 100


# In[7]:


def load_metadata(path):
    file = path.split("/")[-1]
    pet_id = file[:-5].split("-")[0]
    file_id = file[:-5].split("-")[1]
    
    with open(path, encoding="utf-8") as f:
        jfile = json.loads(f.read())
    response = {"labels": [],
                "text": {"PetID": pet_id,
                         "FileID": file_id,
                         "description": ""}}
    
    if "labelAnnotations" in jfile.keys():
        for anot in jfile["labelAnnotations"]:
            response["labels"].append({"PetID": pet_id,
                                       "FileID": file_id,
                                       "description": anot["description"],
                                       "score": anot["score"]})

    if "imagePropertiesAnnotation" in jfile.keys():
        colors = np.zeros((10, 1, 3), dtype=np.uint8)
        scores = np.zeros(10)
        fractions = np.zeros(10)
        getscore = itemgetter("score")
        for i, color in enumerate(sorted(jfile['imagePropertiesAnnotation']["dominantColors"]["colors"],
                                         key=getscore,
                                         reverse=True)
                                 ):

            for j, c in enumerate(["red", "green", "blue"]):
                if not color["color"].get(c) is None:
                    colors[i, 0, j] = color["color"][c] 
                
            scores[i] = color["score"]
            fractions[i] = color["pixelFraction"]
        hsv = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV_FULL)
        response["property"] = {"PetID": pet_id,
                                "FileID": file_id,
                                "top_red": colors[0, 0, 0],
                                "top_green": colors[0, 0, 1],
                                "top_blue": colors[0, 0, 2],
                                "top_score": scores[0],
                                "top_fraction": fractions[0],
                                "top_hue": hsv[0, 0, 0],
                                "top_saturation": hsv[0, 0, 1],
                                "top_brightness": hsv[0, 0, 2],
                                "top3_score": scores[:3].sum(),
                                "top3_fraction": fractions[:3].sum(),
                                "top3_area": np.linalg.norm(np.cross((colors[1] - colors[0])[0], (colors[2] - colors[0])[0])),
                                "top10_fraction": fractions.sum(),
                                "top10_score": scores.sum()}

    if 'cropHintsAnnotation' in jfile.keys():
        tmp = jfile["cropHintsAnnotation"]["cropHints"][0]
        response["crop"] = {"PetID": pet_id,
                            "FileID": file_id,
                            "confidence": tmp["confidence"]}
        if not tmp.get("importanceFraction") is None:
            response["crop"]["importanceFraction"] = tmp["importanceFraction"]
    
    if 'textAnnotations' in jfile.keys():
        for anot in jfile["textAnnotations"]:
            response["text"]["description"] += anot["description"] + " "
    
    if "faceAnnotations" in jfile.keys():
        faceanot = jfile["faceAnnotations"][0]
        response["face"] = {"PetID": pet_id,
                            "FileID": file_id,
                            "detectionConfidence": faceanot['detectionConfidence'],
                            'landmarkingConfidence': faceanot['landmarkingConfidence'],
                            }
    
    return response


# In[8]:


get_ipython().run_cell_magic('time', '', 'metadata_path = [dir_ + file for dir_ in ["../input/petfinder-adoption-prediction/train_metadata/",\n                                          "../input/petfinder-adoption-prediction/test_metadata/"]\n                                 for file in os.listdir(dir_)]\n\nresults = Parallel(n_jobs=-1, verbose=0)([delayed(load_metadata)(path) for path in metadata_path])\n\nlabels = []\nproperties = []\ncrops = []\nfaces = []\ntexts = []\nfor res in results:\n    if not res.get("labels") is None:\n        labels.extend(res["labels"])\n    if not res.get("property") is None:\n        properties.append(res["property"])\n    if not res.get("crop") is None:\n        crops.append(res["crop"])\n    if not res.get("face") is None:\n        faces.append(res["face"])\n    if not res.get("text") is None:\n        texts.append(res["text"])\n\nlabels_df = pd.DataFrame(labels)\nproperties_df = pd.DataFrame(properties)\ncrops_df = pd.DataFrame(crops)\nfaces_df = pd.DataFrame(faces)\ntexts_df = pd.DataFrame(texts)')


# In[9]:


# sentiment ver.
def load_sentiments(path):
    file = path.split("/")[-1]
    pet_id = path.split("/")[-1][:-5]
    
    with open(path, encoding="utf-8") as f:
        jfile = json.loads(f.read())
    
    cnt = 0
    score = []
    magnitude = []
    for sent in jfile.get("sentences"):
        cnt += 1
        score.append(sent["sentiment"]["score"])
        magnitude.append(sent["sentiment"]["magnitude"])

    result = {"PetID": pet_id,
              "documentSentiment_score": jfile['documentSentiment']["score"],
              "documentSentiment_magnitude": jfile['documentSentiment']["magnitude"],
              "language": jfile["language"],
              "sentense_score_mean": np.mean(score),
              "sentense_score_min": np.min(score),
              "sentense_score_std": np.std(score),
              "sentense_magnitude_mean": np.mean(magnitude),
              "sentense_magnitude_min": np.min(magnitude),
              "sentense_magnitude_std": np.std(magnitude),
             }
    return result


# In[10]:


get_ipython().run_cell_magic('time', '', 'sentiment_path = [dir_ + file for dir_ in ["../input/petfinder-adoption-prediction/train_sentiment/",\n                                           "../input/petfinder-adoption-prediction/test_sentiment/"]\n                                  for file in os.listdir(dir_)]\nsentiment_df = pd.DataFrame(Parallel(n_jobs=-1, verbose=0)([delayed(load_sentiments)(path) for path in sentiment_path]))')


# In[11]:


train_newmeta_df = train_df[["PetID"]]
test_newmeta_df = test_df[["PetID"]]

# labelAnnotations
labels_global_score = labels_df.groupby("PetID")["score"].agg(["mean", "max", "min", "std"])
labels_global_score.columns = ["labels_global_score_" + col for col in labels_global_score.columns]

train_newmeta_df = train_newmeta_df.merge(labels_global_score.reset_index(),
                          on="PetID",
                          how="left")
test_newmeta_df = test_newmeta_df.merge(labels_global_score.reset_index(),
                        on="PetID",
                        how="left")


# imagePropertiesAnnotation
properties_df.iloc[:, 2:] = (properties_df.iloc[:, 2:] - properties_df.iloc[:, 2:].mean())/properties_df.iloc[:, 2:].std()
profile_properties_df = properties_df[properties_df["FileID"] == "1"].drop("FileID", axis=1)
profile_properties_df.columns = ["profile_properties_" + col if col != "PetID" else col for col in profile_properties_df.columns]

train_newmeta_df = train_newmeta_df.merge(profile_properties_df,
                                          on="PetID",
                                          how="left")
test_newmeta_df = test_newmeta_df.merge(profile_properties_df,
                                        on="PetID",
                                        how="left")


properties_agg = properties_df.groupby("PetID").agg({"top_score": ["mean", "std"],
                                                     "top10_score": ["mean", "std"],
                                                     "top_fraction": ["mean", "std"],
                                                     "top10_fraction": ["mean", "std"]})
properties_agg.columns = ["property_agg_" + "_".join(col) for col in properties_agg.columns]

train_newmeta_df = train_newmeta_df.merge(properties_agg,
                          on="PetID",
                          how="left")
test_newmeta_df = test_newmeta_df.merge(properties_agg,
                        on="PetID",
                        how="left")

# cropHintsAnnotation
profile_crops_df = crops_df[crops_df["FileID"] == "1"].drop("FileID", axis=1)
train_newmeta_df = train_newmeta_df.merge(profile_crops_df,
                                          on="PetID",
                                          how="left")
test_newmeta_df = test_newmeta_df.merge(profile_crops_df,
                                        on="PetID",
                                        how="left")

# faceAnnotations
faces_df.columns = ["FileID", "PetID", "face_crop_detectionConfidence", "face_crop_landmarkingConfidence"]
profile_faces_df = faces_df[faces_df["FileID"] == "1"].drop("FileID", axis=1)
train_newmeta_df = train_newmeta_df.merge(profile_faces_df,
                                          on="PetID",
                                          how="left")
test_newmeta_df = test_newmeta_df.merge(profile_faces_df,
                                        on="PetID",
                                        how="left")


# In[12]:


# agged_features
train_newmeta_df.head()


# In[13]:


texts_agg = texts_df.groupby("PetID")["description"].sum().reset_index()
texts_agg.columns = ["PetID", "metadata_description"]

train_df = train_df.merge(texts_agg[["PetID", "metadata_description"]],
                          on="PetID",
                          how="left")
test_df = test_df.merge(texts_agg[["PetID", "metadata_description"]],
                        on="PetID",
                        how="left")


# In[14]:


train_df["Description"] = train_df["Description"].fillna("none") + " " + train_df["metadata_description"].fillna("none")
test_df["Description"] = test_df["Description"].fillna("none") + " " + test_df["metadata_description"].fillna("none")


# In[15]:


# data cleansing
import re

all_text = pd.concat([train_df["Description"], test_df["Description"]]).fillna("none").values
shorted_forms = {"i'm":"i am","i'll":"i will","i'd":"i had","i've":"i have","you're":"you are","you'll":"you will","you'd":"you had","you've":"you have","he's":"he has","he'll":"he will","he'd":"he had","she's":"she has","she'll":"she will","she'd":"she had","it's (or ‘tis)":"it is","it'll":"it will","it'd":"it had","it's":"it is","we're":"we are","we'll":"we will","we'd":"we had","we've":"we have","they're":"they are","they'll":"they will","they'd":"they had","they've":"they have","that's":"that has","that'll":"that will","that'd":"that had","who's":"who has","who'll":"who will","who'd":"who had","what's/what're":"what is/what are","what'll":"what will","what'd":"what had","what's":"what is","where's":"where has","where'll":"where will","where'd":"where had","when's":"when has","when'll":"when will","when'd":"when had","why's":"why has","why'll":"why will","why'd":"why had","how's":"how has","how'll":"how will","how'd":"how had","what're":"what are","isn't":"is not","aren't":"are not","wasn't":"was not","weren't":"were not","haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not","wouldn't":"would not","don't":"do not","doesn't":"does not","didn't":"did not","can't":"cannot","couldn't":"could not","shouldn't":"should not","mightn't":"might not","mustn't":"must not"}
noalphabet = set()
for text in tqdm(all_text):
    noalphabet.update(list(re.sub("[0-9a-zA-Z\s]", "", text)))

cleaned_texts = []
noalphabet_count = []
repwords = "|".join(map(re.escape, noalphabet))
for text in all_text:
    text = text.lower()
    for k, v in shorted_forms.items():
        text = text.replace(k, v)
    noalphabet_count.append(len(re.findall(repwords, text)))
    text = re.sub(repwords, " ", text)
    cleaned_texts.append(re.sub("([0-9]+)", "", text))


# In[16]:


# description
tfv = TfidfVectorizer(min_df=3,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')

tfv.fit(cleaned_texts)
X_tfv = tfv.transform(cleaned_texts)

svd = TruncatedSVD(n_components=150, random_state=2434)
svd.fit(X_tfv)
X_desc_tfv_svd = svd.transform(X_tfv)

svd = TruncatedSVD(n_components=16, random_state=2434)
svd.fit(X_tfv)
X_desc_tfv_svd_mini = svd.transform(X_tfv)


# In[17]:


labels_agg = labels_df.groupby(["PetID", "description"])["score"].max().reset_index()

label_agg_text = labels_agg.groupby("PetID")["description"]                           .apply(lambda x: " ".join(x))                           .reset_index()                           .rename(columns={"description": "label_description_1"})

train_df = train_df.merge(label_agg_text,
                          on="PetID",
                          how="left")
test_df = test_df.merge(label_agg_text,
                        on="PetID",
                        how="left")

# spsp as dammy
label_agg_text = labels_agg.groupby("PetID")["description"]                           .apply(lambda x: " spsp spsp ".join(x))                           .reset_index()                           .rename(columns={"description": "label_description_2"})

train_df = train_df.merge(label_agg_text,
                          on="PetID",
                          how="left")
test_df = test_df.merge(label_agg_text,
                        on="PetID",
                        how="left")


# In[18]:


# space version
lebeldesc_texts = pd.concat([train_df["label_description_1"], test_df["label_description_1"]]).fillna("none").values

tfv = TfidfVectorizer(min_df=3,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')

tfv.fit(lebeldesc_texts)
X_tfv = tfv.transform(lebeldesc_texts)
print("shape is", X_tfv.shape)

svd = TruncatedSVD(n_components=70, random_state=2434)#NMF(n_components=150, random_state=2434, shuffle=True, verbose=True)
svd.fit(X_tfv)

X_labeldesc_tfv_svd = svd.transform(X_tfv)


# In[19]:


X_train = X_labeldesc_tfv_svd[:len(train_df)]
X_test = X_labeldesc_tfv_svd[len(train_df):]
y_train = train_df["AdoptionSpeed"]
group = train_df["group"]

param = {'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread":nthread,
         "verbosity": -1}

train_oof, test_oof = make_lgb_oofs(X_train, y_train, group, X_test, param)


# In[20]:


print(rmse(y_train, train_oof))


# In[21]:


# dammy version
lebeldesc_texts = pd.concat([train_df["label_description_2"], test_df["label_description_2"]]).fillna("none").values

tfv = TfidfVectorizer(min_df=3,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')

tfv.fit(lebeldesc_texts)
X_tfv = tfv.transform(lebeldesc_texts)
print("shape is", X_tfv.shape)

svd = TruncatedSVD(n_components=70, random_state=2434)#NMF(n_components=150, random_state=2434, shuffle=True, verbose=True)
svd.fit(X_tfv)

X_labeldesc_tfv_svd = svd.transform(X_tfv)


# In[22]:


X_train = X_labeldesc_tfv_svd[:len(train_df)]
X_test = X_labeldesc_tfv_svd[len(train_df):]
y_train = train_df["AdoptionSpeed"]
group = train_df["group"]

param = {'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread":nthread,
         "verbosity": -1}

train_oof, test_oof = make_lgb_oofs(X_train, y_train, group, X_test, param)


# In[23]:


print(rmse(y_train, train_oof))


# In[24]:


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def load_fasttext_vectors(EMBEDDING_FILE):
    vectors = dict()
    with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()[1:]):
            key, *vec = line.rstrip().split()
            vectors[key] = np.array(vec, dtype=np.float32)
    return vectors

def load_glove_vectors(EMBEDDING_FILE):
    vectors = dict()
    with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            key, *vec = line.rstrip().split()
            vectors[key] = np.array(vec, dtype=np.float32)
    return vectors

def get_swem_vectors(cleaned_texts, word_vectors, dim=300):
    dim = word_vectors["word"].shape[0]
    swem_result = np.zeros((len(cleaned_texts), dim), dtype=np.float32) + 1e-6
    exist_words = set(word_vectors.keys())

    for i, text in tqdm(enumerate(cleaned_texts)):
        vecs = []
        for word in nltk.word_tokenize(text):
            word = word.lower()
            if word in exist_words:
                vecs.append(word_vectors[word])
        if len(vecs):
            V = np.vstack(vecs)
            swem_result[i] = V[np.argmax(np.abs(V), axis=0), np.arange(dim)]
    return swem_result

def get_idf_weighted_vectors(cleaned_texts, word_vectors, idfs, dim=300):
    dim = word_vectors["word"].shape[0]
    result = np.zeros((len(cleaned_texts), dim), dtype=np.float32) + 1e-6
    exist_words = set(word_vectors.keys()) & set(idfs.keys())
    
    for i, text in tqdm(enumerate(cleaned_texts)):
        idf_sum = 0
        vec = np.zeros(dim, dtype=np.float32)
        for word in nltk.word_tokenize(text):
            word = word.lower()
            if word in exist_words:
                vec += word_vectors[word] * idfs[word]
                idf_sum += idfs[word]
        if idf_sum > 0:
            vec /= idf_sum
        result[i] = vec
    return result


# In[25]:


tfidf = TfidfVectorizer()
tfidf.fit(cleaned_texts)
idfs = {k:tfidf.idf_[v] for k, v in tfidf.vocabulary_.items()}


EMBEDDING_FILE = "../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec"
commoncrawl_vectors = load_fasttext_vectors(EMBEDDING_FILE)

commoncrawl_swems = get_swem_vectors(cleaned_texts, commoncrawl_vectors) + 1e-8
commoncrawl_idfs = get_idf_weighted_vectors(cleaned_texts, commoncrawl_vectors, idfs)


# In[26]:


del commoncrawl_vectors
collect()


# In[27]:


rescuerid_tf = rescuerid_encoder.transform(pd.concat([train_df["RescuerID"], test_df["RescuerID"]]))

cossim_res = []
vecsize = np.sqrt(np.square(commoncrawl_swems).sum(axis=1))
for i in tqdm(range(commoncrawl_swems.shape[0])):
    tmp = {}
    cossim = ((commoncrawl_swems[i].reshape(1, -1) @ commoncrawl_swems.T) / (vecsize[i] * vecsize))[0]
    cossim[i] = 0
    same_rescuer = rescuerid_tf[i] == rescuerid_tf
    different_cossim = cossim[np.where(same_rescuer^1)[0]]
    tmp["defferent_rescuer_cossim_mean"] = different_cossim.mean()
    tmp["defferent_rescuer_cossim_std"] = different_cossim.std()
    tmp["defferent_rescuer_cossim_max"] = different_cossim.max()
    if same_rescuer.sum() > 1:
        same_rescuer[i] = False
        same_cossim = cossim[np.where(same_rescuer)[0]]
        tmp["same_rescuer_cossim_mean"] = same_cossim.mean()
        tmp["same_rescuer_cossim_std"] = same_cossim.std()
        tmp["same_rescuer_cossim_max"] = same_cossim.max()
    cossim_res.append(tmp)

cossim_df = pd.DataFrame(cossim_res)
cossim_df.columns = ["commoncrawl_" + col for col in cossim_df.columns]


# In[28]:


# image


# In[29]:


from keras.applications.densenet import DenseNet121, preprocess_input
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

def make_model(BaseModel, base_shape, weights="imagenet"):
    inp = Input(base_shape)
    base_model = BaseModel(input_tensor=inp, weights=weights, include_top=False)
    x = base_model.output
    out = GlobalAveragePooling2D()(x)
    model = Model(inp, out)
    return model

weight_path = "../input/densenet-keras/DenseNet-BC-121-32-no-top.h5"

dense121 = make_model(DenseNet121, (224, 224, 3), weights=weight_path)


# In[30]:


import cv2
from skimage import feature
from imagehash import whash
from PIL import Image

def resize_image(image, resized_shape):
    h, w, c = image.shape
    if h > w:
        new_image = np.zeros((h, h, c), dtype=np.uint8)
        left = (h-w)//2
        right = left + w
        new_image[:, left:right, :] = image
    else:
        new_image = np.zeros((w, w, c), dtype=np.uint8)
        top = (w-h)//2
        bottom = top + h
        new_image[top:bottom, :, :] = image
    resized_image = cv2.resize(new_image, resized_shape, cv2.INTER_LANCZOS4)
    return resized_image

def shrink_image(image):
    h, w, c = image.shape
    if h > w:
        new_h = 224
        new_w = int((w * 224)//h)
    else:
        new_w = 224
        new_h = int((h * 224)//w)
    return cv2.resize(image, (new_w, new_h), cv2.INTER_LANCZOS4)

def padding_image(image):
    h, w, c = image.shape
    new_image = np.zeros((224, 224, 3), dtype=np.uint8)
    if h == 224:
        left = (h-w)//2
        right = left + w
        new_image[:, left:right, :] = image
    else:
        top = (w-h)//2
        bottom = top + h
        new_image[top:bottom, :, :] = image
    return new_image


# In[31]:


def image_analysis(path):
    res = {}
    
    res["PetID"], res["FileID"] = path.split("/")[-1][:-4].split("-")
    image = cv2.imread(path)[:,:,[2, 1, 0]]
    image_hight, image_width = image.shape[:2]
    image_size = image_hight * image_width
    image_aspect = image_width / image_hight
    
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_s3 = float(np.sum(feature.canny(grey_image, sigma=3))) / image_size
    blurrness = cv2.Laplacian(grey_image, cv2.CV_64F).var()
    
    whash_res = whash(Image.fromarray(image))

    dark_percent = np.all(image.reshape(-1, 3) <= 20, axis=1).mean()
    light_percent = np.all(image.reshape(-1, 3) >= 240, axis=1).mean()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
    hue, saturation, brightness = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    hue_degree = hue / 255 * 2 * np.pi
    hue_sin, hue_cos = np.sin(hue_degree), np.cos(hue_degree)

    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg, yb = red - green, (red+green)/2 - blue
    colorfulness = np.sqrt(rg.var() + yb.var()) + 0.3*np.sqrt(np.square(rg.mean()) + np.square(yb.mean()))

    grayscale_simplicty = (np.cumsum(np.sort(np.histogram(grey_image.ravel(), 255, [0, 255])[0])) > image_size * .05).mean()
    hue_simplicty = (np.cumsum(np.sort(np.histogram(hue.ravel(), 255, [0, 255])[0])) > image_size * .05).mean()

    if res["FileID"] == "1":
        resized_image = resize_image(image, (224, 224))
        #res_feature = extract_deep_feature(resized_image, Res_exter, resnet50)
        #dense_feature = extract_deep_feature(resized_image, Dense_exter, densenet)
        # deep_feature = np.hstack([vgg_feature, res_feature, dense_feature])[0]
    else:
        resized_image = None    
    
    res.update({"image_hight": image_hight,
                "image_width": image_width,
                "image_size": image_size,
                "image_aspect": image_aspect,
                "dark_percent": dark_percent,
                "light_percent": light_percent,
                "canny_s3": canny_s3,
                "blurrness": blurrness,
                "hue_sin_mean": hue_sin.mean(),
                "hue_cos_mean": hue_cos.mean(),
                "red_mean": red.mean(),
                "red_std": red.std(),
                "green_mean": green.mean(),
                "green_std": green.std(),
                "blue_mean": blue.mean(),
                "blue_srd": blue.std(),
                "saturation_mean": saturation.mean(),
                "saturarion_std": saturation.std(),
                "brightness_mean": brightness.mean(),
                "brightness_std": brightness.std(),
                "colorfulness": colorfulness,
                "greyscale_simplicity": grayscale_simplicty,
                "hue_simplicty": hue_simplicty,
                "whash": whash_res,
                "image": resized_image,
                })
    return res


# In[32]:


def split_extracter(paths, exter, preprocess_func, n_splits=10):
    splited_len = -(-len(paths)//n_splits)
    image_feat_df = pd.DataFrame()
    exter_feature = []
    all_keys = []
    for j in tqdm(range(n_splits)):
        r = Parallel(n_jobs=-1, verbose=0)([delayed(image_analysis)(image_path) for image_path in paths[splited_len*j:splited_len*(j+1)]])

        keys = []
        images = []

        mini_images = np.zeros((len(r), 96, 96, 3))
        for i in range(len(r)):
            image = r[i].pop("image")
            if not image is None:
                keys.append(r[i]["PetID"])
                images.append(image)

        tmp_image_feat = pd.DataFrame(r)
        image_feat_df = pd.concat([image_feat_df, tmp_image_feat])        
        image_array = np.zeros((len(keys), 224, 224, 3), dtype=np.float32)

        for i in range(len(keys)):
            image_array[i] = images[i]
    
        exter_feature.append(exter.predict(preprocess_func(image_array.astype(np.float32)),
                                                   batch_size=32, verbose=1))
        all_keys.extend(keys)

    exter_feature = np.vstack(exter_feature)
    exter_df = pd.DataFrame(all_keys)
    exter_df.columns = ["PetID"]
    exter_df = pd.concat([exter_df, pd.DataFrame(exter_feature)], axis=1)    
         
    return image_feat_df, exter_df


# In[33]:


image_path = [dir_ + file for dir_ in ["../input/petfinder-adoption-prediction/train_images/",
                                       "../input/petfinder-adoption-prediction/test_images/"]
                              for file in os.listdir(dir_)][:1000]


# In[34]:


# if you use full dataset, it takes
# densenet121: 4500 sec
# gloval feature: 2700 sec
# total: 7200 sec

image_feat_df, exter_df = split_extracter(image_path, dense121, preprocess_func=preprocess_input)


# In[35]:


image_feat_df.head()

