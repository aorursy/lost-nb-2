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
from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF

from collections import Counter

import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm, tqdm_notebook

np.random.seed(369)


# In[2]:


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


# In[3]:


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
    
def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))    


# In[4]:


import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook
from keras.applications.densenet import preprocess_input, DenseNet121

train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
img_size = 256
batch_size = 16


# In[5]:


pet_ids = train_df['PetID'].values
n_batches = len(pet_ids) // batch_size + 1


# In[6]:


def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path, pet_id):
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image


# In[7]:


from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, 
                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",
                       include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)


# In[8]:


features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/petfinder-adoption-prediction/train_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


# In[9]:


train_feats = pd.DataFrame.from_dict(features, orient='index')
train_feats.columns = ['pic_'+str(i) for i in range(train_feats.shape[1])]


# In[10]:


test_df = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')

pet_ids = test_df['PetID'].values
n_batches = len(pet_ids) // batch_size + 1

features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/petfinder-adoption-prediction/test_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


# In[11]:


test_feats = pd.DataFrame.from_dict(features, orient='index')
test_feats.columns = ['pic_'+str(i) for i in range(test_feats.shape[1])]


# In[12]:


test_feats = test_feats.reset_index()
test_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

train_feats = train_feats.reset_index()
train_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)

test_feats.head()


# In[13]:


print('Train')
train = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
print(train.shape)

print('Test')
test = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
print(test.shape)

print('Breeds')
breeds = pd.read_csv("../input/petfinder-adoption-prediction/breed_labels.csv")
print(breeds.shape)

print('Colors')
colors = pd.read_csv("../input/petfinder-adoption-prediction/color_labels.csv")
print(colors.shape)

print('States')
states = pd.read_csv("../input/petfinder-adoption-prediction/state_labels.csv")
print(states.shape)

target = train['AdoptionSpeed']
train_id = train['PetID']
test_id = test['PetID']

# additional some magic features :)
train['second_color'] = train['Color2']*10 + train['Color3']
train['len_desc'] = train['Description'].str.len()
train['len_name'] = train['Name'].str.len()
    # split breed
train['Breed1_Cat'] = train['Breed1'][train['Type']==1].fillna(-1)
train['Breed1_Dog'] = train['Breed1'][train['Type']==2].fillna(-1)
train['Breed2_Cat'] = train['Breed2'][train['Type']==1].fillna(-1)
train['Breed2_Dog'] = train['Breed2'][train['Type']==2].fillna(-1)
train['age_bins'] = pd.cut(train['Age'],5,labels=[1,2,3,4,5]).astype(int)
train['gender+age_bins'] = train['Gender']*10+train['age_bins']

# additional feature by lam
test['second_color'] = test['Color2']*10 + test['Color3']
test['len_desc'] = test['Description'].str.len()
test['len_name'] = test['Name'].str.len()
    # split breed
test['Breed1_Cat'] = test['Breed1'][test['Type']==1].fillna(-1)
test['Breed1_Dog'] = test['Breed1'][test['Type']==2].fillna(-1)
test['Breed2_Cat'] = test['Breed2'][test['Type']==1].fillna(-1)
test['Breed2_Dog'] = test['Breed2'][test['Type']==2].fillna(-1)
test['age_bins'] = pd.cut(test['Age'],5,labels=[1,2,3,4,5]).astype(int)
test['gender+age_bins'] = test['Gender']*10+test['age_bins']

test['AdoptionSpeed'] = np.nan
X = pd.concat([train, test], ignore_index=True, sort=False)
# additional feature
rescuer_count = X.groupby(['RescuerID'])['PetID'].count()
rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']
train['RescuerID_COUNT'] = train['RescuerID'].map(rescuer_count.to_dict())
test['RescuerID_COUNT'] = test['RescuerID'].map(rescuer_count.to_dict())

train = pd.merge(train, train_feats, how='left', on='PetID')
test = pd.merge(test, test_feats, how='left', on='PetID')

import gc
del X; gc.collect()


# In[14]:





# In[14]:


doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in train_id:
    try:
        with open('../input/petfinder-adoption-prediction/train_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

train.loc[:, 'doc_sent_mag'] = doc_sent_mag
train.loc[:, 'doc_sent_score'] = doc_sent_score

doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in test_id:
    try:
        with open('../input/petfinder-adoption-prediction/test_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

test.loc[:, 'doc_sent_mag'] = doc_sent_mag
test.loc[:, 'doc_sent_score'] = doc_sent_score


# In[15]:


## WITHOUT ERROR FIXED
train_desc = train.Description.fillna("none").values
test_desc = test.Description.fillna("none").values

tfv = TfidfVectorizer(min_df=3,  max_features=10000,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')
    
# Fit TFIDF
tfv.fit(list(train_desc))
X =  tfv.transform(train_desc)
X_test = tfv.transform(test_desc)
print("X (tfidf):", X.shape)

svd = TruncatedSVD(n_components=200)
svd.fit(X)
# print(svd.explained_variance_ratio_.sum())
# print(svd.explained_variance_ratio_)
svd_col = svd.transform(X)

svd_col = pd.DataFrame(svd_col, columns=['svd_{}'.format(i) for i in range(200)])
train = pd.concat((train, svd_col), axis=1)
svd_col_test = svd.transform(X_test)
svd_col_test = pd.DataFrame(svd_col_test, columns=['svd_{}'.format(i) for i in range(200)])
test = pd.concat((test, svd_col_test), axis=1)

nmf = NMF(n_components=200)
nmf.fit(X)
# print(svd.explained_variance_ratio_.sum())
# print(svd.explained_variance_ratio_)
nmf_col = nmf.transform(X)

nmf_col = pd.DataFrame(nmf_col, columns=['nmf_{}'.format(i) for i in range(200)])
train = pd.concat((train, nmf_col), axis=1)
nmf_col_test = svd.transform(X_test)
nmf_col_test = pd.DataFrame(nmf_col_test, columns=['nmf_{}'.format(i) for i in range(200)])
test = pd.concat((test, nmf_col_test), axis=1)

print("train:", train.shape)


# In[16]:



# del embeddings_index; gc.collect()


# In[17]:


# word embedding
import gc
import re, string
import time 
gc.collect()
def get_desc_vector(sent):
    v = np.zeros((300,))
    n_words = 0
    for w in sent.split():
        if w in embeddings_index:
            v += embeddings_index.get(w)
            n_words += 1
        v = v / n_words if n_words > 0 else np.zeros((300,))
    return v

def preprocess(x):
    x = str(x).lower()
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    x = re_tok.sub(r' \1 ', x)
    return x

train.Description = train.Description.apply(preprocess)
test.Description = test.Description.apply(preprocess)



# fasttext
EMBEDDING_FILE = '../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE))

from multiprocess import Pool
pool = Pool(2)
try: train_x_1 = pool.map(get_desc_vector, train.Description.values.tolist())
except ValueError as error: print(error)
pool.terminate()
train_x_1 = np.array(train_x_1)

pool = Pool(2)
try: test_x_1 = pool.map(get_desc_vector, test.Description.values.tolist())
except ValueError as error: print(error)
pool.terminate()
test_x_1 = np.array(test_x_1)

del embeddings_index; gc.collect()



# glove
EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE))

from multiprocess import Pool
pool = Pool(2)
try: train_x_2 = pool.map(get_desc_vector, train.Description.values.tolist())
except ValueError as error: print(error)
pool.terminate()
train_x_2 = np.array(train_x_2)

pool = Pool(2)
try: test_x_2 = pool.map(get_desc_vector, test.Description.values.tolist())
except ValueError as error: print(error)
pool.terminate()
test_x_2 = np.array(test_x_2)

del embeddings_index; gc.collect()



# concate mean
train_x_mean = np.mean([train_x_1, train_x_2], axis=0)
test_x_mean = np.mean([test_x_1, test_x_2], axis=0)

embedding_train = pd.DataFrame()
embedding_train['PetID'] = train.PetID.values
for i in range(300):
    embedding_train[f'emb_{i}'] = train_x_mean[:,i]

embedding_test = pd.DataFrame()
embedding_test['PetID'] = test.PetID.values
for i in range(300):
    embedding_test[f'emb_{i}'] = test_x_mean[:,i]
    
train = pd.merge(train, embedding_train, how='left', on='PetID')
test = pd.merge(test, embedding_test, how='left', on='PetID')

print(train.shape)
del train_x_mean, embedding_train, embedding_test;gc.collect()


# In[18]:


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in train_id:
    try:
        with open('../input/petfinder-adoption-prediction/train_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)
print(nf_count)
print(nl_count)
train.loc[:, 'vertex_x'] = vertex_xs
train.loc[:, 'vertex_y'] = vertex_ys
train.loc[:, 'bounding_confidence'] = bounding_confidences
train.loc[:, 'bounding_importance'] = bounding_importance_fracs
train.loc[:, 'dominant_blue'] = dominant_blues
train.loc[:, 'dominant_green'] = dominant_greens
train.loc[:, 'dominant_red'] = dominant_reds
train.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
train.loc[:, 'dominant_score'] = dominant_scores
train.loc[:, 'label_description'] = label_descriptions
train.loc[:, 'label_score'] = label_scores


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in test_id:
    try:
        with open('../input/petfinder-adoption-prediction/test_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
test.loc[:, 'vertex_x'] = vertex_xs
test.loc[:, 'vertex_y'] = vertex_ys
test.loc[:, 'bounding_confidence'] = bounding_confidences
test.loc[:, 'bounding_importance'] = bounding_importance_fracs
test.loc[:, 'dominant_blue'] = dominant_blues
test.loc[:, 'dominant_green'] = dominant_greens
test.loc[:, 'dominant_red'] = dominant_reds
test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
test.loc[:, 'dominant_score'] = dominant_scores
test.loc[:, 'label_description'] = label_descriptions
test.loc[:, 'label_score'] = label_scores


# In[19]:


train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)
test.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)

train.drop(['Breed1', 'Breed2', 'age_bins'], axis=1, inplace=True)
test.drop(['Breed1', 'Breed2', 'age_bins'], axis=1, inplace=True)

train.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)
test.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)


# In[20]:


numeric_cols = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 
                'doc_sent_mag', 'doc_sent_score', 'dominant_score', 'dominant_pixel_frac', 
                'dominant_red', 'dominant_green', 'dominant_blue', 'bounding_importance', 
                'bounding_confidence', 'vertex_x', 'vertex_y', 'label_score', 'len_desc', 'len_name',
               'RescuerID_COUNT', 'second_color'] +\
               [col for col in train.columns if col.startswith('pic') or col.startswith('svd') or col.startswith('nmf') or col.startswith('emb')]
cat_cols = list(set(train.columns) - set(numeric_cols))
train.loc[:, cat_cols] = train[cat_cols].astype('category')
test.loc[:, cat_cols] = test[cat_cols].astype('category')
print(train.shape)
print(test.shape)

# get the categorical features
foo = train.dtypes
cat_feature_names = foo[foo == "category"]
cat_features = [train.columns.get_loc(c) for c in train.columns if c in cat_feature_names]


# In[21]:


for i in cat_cols:
    train.loc[:, i] = pd.factorize(train.loc[:, i])[0]
    test.loc[:, i] = pd.factorize(test.loc[:, i])[0]


# In[22]:


import tensorflow as tf
tf.reset_default_graph()
N_SPLITS = 5
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)
    fold_splits = kf.split(train, target)
    cv_scores = []
    qwk_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0], N_SPLITS))
    all_coefficients = np.zeros((N_SPLITS, 4))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/' + str(N_SPLITS))
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, coefficients, qwk = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        all_coefficients[i - 1, :] = coefficients
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            qwk_scores.append(qwk)
            print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))
        i += 1
    print('{} cv RMSE scores : {}'.format(label, cv_scores))
    print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv QWK scores : {}'.format(label, qwk_scores))
    print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
    print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
    pred_full_test = pred_full_test / float(N_SPLITS)
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
               'cv': cv_scores, 'qwk': qwk_scores,
               'importance': feature_importance_df,
               'coefficients': all_coefficients}
    return results

params = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'eta': 0.0123,
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': 1,
}
# Additional parameters:
n_splits = 5
verbose_eval = 1000
num_rounds = 60000
early_stop = 500

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
    print('Prep LGB')
    # d_train = lgb.Dataset(train_X, label=train_y)
    # d_valid = lgb.Dataset(test_X, label=test_y)
    d_train = xgb.DMatrix(data=train_X, label=train_y, feature_names=train_X.columns)
    d_valid = xgb.DMatrix(data=test_X, label=test_y, feature_names=test_X.columns)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    print('Train LGB')

    model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                      early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

    #valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
    #test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)
    
    print('Predict 1/2')
    #valid_pred = model.predict(test_X, num_iteration=model.best_iteration)
    pred_test_y = model.predict(xgb.DMatrix(test_X, feature_names=test_X.columns), ntree_limit=model.best_ntree_limit)

    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients)
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(test_y, pred_test_y_k)
    print("QWK = ", qwk)
    print('Predict 2/2')
    #pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    pred_test_y2 = model.predict(xgb.DMatrix(test_X2, feature_names=test_X2.columns), ntree_limit=model.best_ntree_limit)

    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), coefficients, qwk

results = run_cv_model(train, test, target, runLGB, params, rmse, 'lgb')


# In[23]:


# imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
# imports.sort_values('importance', ascending=False)


# In[24]:


# for i in range(300):
#     print(imports[imports['feature'] == 'emb_' + str(i)])


# In[25]:


# imports.head()


# In[26]:


optR = OptimizedRounder()
coefficients_ = np.mean(results['coefficients'], axis=0)
print(coefficients_)
# manually adjust coefs
coefficients_[0] = 1.645
coefficients_[1] = 2.115
coefficients_[3] = 2.84
train_predictions = [r[0] for r in results['train']]
train_predictions = optR.predict(train_predictions, coefficients_).astype(int)
Counter(train_predictions)


# In[27]:


optR = OptimizedRounder()
coefficients_ = np.mean(results['coefficients'], axis=0)
print(coefficients_)
# manually adjust coefs
coefficients_[0] = 1.645
coefficients_[1] = 2.115
coefficients_[3] = 2.84
test_predictions = [r[0] for r in results['test']]
test_predictions = optR.predict(test_predictions, coefficients_).astype(int)
Counter(test_predictions)


# In[28]:


print("True Distribution:")
print(pd.value_counts(target, normalize=True).sort_index())
print("Test Predicted Distribution:")
print(pd.value_counts(test_predictions, normalize=True).sort_index())
print("Train Predicted Distribution:")
print(pd.value_counts(train_predictions, normalize=True).sort_index())


# In[29]:


pd.DataFrame(sk_cmatrix(target, train_predictions), index=list(range(5)), columns=list(range(5)))


# In[30]:


quadratic_weighted_kappa(target, train_predictions)
rmse(target, [r[0] for r in results['train']])
submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
submission.head()


# In[31]:


submission.to_csv('submission.csv', index=False)

