#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from mlens.ensemble import SuperLearner

data_path = '/kaggle/input/fake-news'


class feature_eng(TransformerMixin, BaseEstimator):
    """ Create features """

    def __init__(self):
        self.ps = PorterStemmer()

    def __nltk_downloads(self):
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt')

    def __clean(self, string_val):
        string_val = re.sub('[^a-zA-Z]', ' ', str(string_val))
        string_val = string_val.lower()
        string_val = word_tokenize(string_val)
        string_val = [
            self.ps.stem(word) for word in string_val if not word in stopwords.words('english')]
        string_val = ' '.join(string_val)

        return string_val

    def fit(self, X, y=None):
        try:
            stopwords.words('english')
            word_tokenize(['required downdload'])
        except:
            self.__nltk_downloads()

        return self

    def transform(self, X):

        print("==> preprocessing author")
        self.X = X.copy()
        self.X['author_miss'] = self.X['author'].isna() * 1
        self.X['author_len'] = self.X['author'].apply(lambda x: len(str(x)))
        self.X['author_w_len'] = self.X['author'].apply(
            lambda x: len(str(x).split(' ')))
        self.X['author_char_by_words'] = self.X['author_len'] /             self.X['author_w_len']
        self.X['author_clean'] = Parallel()(delayed(self.__clean)(x)
                                            for x in tqdm(self.X['author'].tolist()))

        print("==> preprocessing text")
        self.X['text_miss'] = self.X['text'].isna() * 1
        self.X['text_len'] = self.X['text'].apply(lambda x: len(str(x)))
        self.X['text_w_len'] = self.X['text'].apply(
            lambda x: len(str(x).split(' ')))
        self.X['text_char_by_words'] = self.X['text_len'] /             self.X['text_w_len']
        self.X['text_clean'] = Parallel()(delayed(self.__clean)(x)
                                          for x in tqdm(self.X['text'].tolist()))

        print("==> preprocessing title")
        self.X['title_miss'] = self.X['title'].isna() * 1
        self.X['title_len'] = self.X['title'].apply(lambda x: len(str(x)))
        self.X['title_w_len'] = self.X['title'].apply(
            lambda x: len(str(x).split(' ')))
        self.X['title_char_by_words'] = self.X['title_len'] /             self.X['title_w_len']
        self.X['title_clean'] = Parallel()(delayed(self.__clean)(x)
                                           for x in tqdm(self.X['title'].tolist()))

        self.X = self.X.drop(
            ['title', 'author', 'text', 'author_clean'], axis=1)

        return self.X


class word_vecs():
    """create word vectors """

    def __init__(self):
        self

    def fit(self, X, y=None):
        self.X = X.copy()
        titles = self.X['title_clean'].to_list()
        text = self.X['title_clean'].to_list()

        self.titles_cv = CountVectorizer(max_features=1000, ngram_range=(1, 3))
        self.titles_cv.fit(titles)

        self.text_cv = CountVectorizer(max_features=1000, ngram_range=(1, 3))
        self.text_cv.fit(text)

    def transform(self, X, y=None):
        otherX = X.drop(['text_clean', 'title_clean'], axis=1).values
        titles = X['title_clean'].to_list()
        text = X['title_clean'].to_list()
        titles_feats = self.titles_cv.transform(titles).toarray()
        texta_feats = self.text_cv.transform(text).toarray()

        feats = np.concatenate([otherX, titles_feats, texta_feats], axis=1)

        return feats

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    
    
if __name__ == '__main__':

    # read train 
    train = pd.read_csv(data_path + '/train.csv')
    train = train.set_index('id')

    # Partition train
    dev_X, val_X, dev_y, val_y = train_test_split(
        train.drop('label', axis=1), train['label'], test_size=.3, random_state=2020)

    
    def mlens_model():
        
        ensemble = SuperLearner(scorer=roc_auc_score, folds=5, array_check=1, random_state=2020, verbose=True)
        ensemble.add([('catboost',cb.CatBoostClassifier(verbose=False)), ('lgb', lgb.LGBMClassifier())], proba=True)
        ensemble.add_meta((xgb.XGBClassifier(verbose=False)), proba=True)

        return ensemble

    # make pipeline
    feats_preprocess = Pipeline(
        steps=[('feature_create', feature_eng()), ('word_vectors', word_vecs())])
    model_pipe = Pipeline(steps=[('feats_preprocess', feats_preprocess),
                                  ('model_ens', mlens_model())])

    # model training
    model_pipe.fit(dev_X, dev_y)
    print(pd.DataFrame(model_pipe['model_ens'].data)) # model report
    
    # validation
    val_pred = model_pipe.predict_proba(val_X)
    print('val auc: ', roc_auc_score(val_y, val_pred[:,1]))
    print('val classification summary:')
    print(classification_report(val_y, val_pred[:,1]>=.5))

    # test predictions
    test = pd.read_csv(data_path + '/test.csv').set_index('id')
    test_pred = model_pipe.predict_proba(test)
    test['label'] = test_pred[:,1]
    test[['label']].reset_index().to_csv('/kaggle/working/test_submit.csv', index=False)

