#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 20}

mpl.rc('font', **font)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from datetime import datetime
from timeit import timeit


from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


# In[ ]:


# import the Train and Test CSV files
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


# analyse the dataset: find the number of true'1'/false'0' values under each labels in the training dataset
valCount = (train_data.iloc[:, 2:8]).apply(pd.value_counts)
valCount.sort_values(by=1, ascending=True, axis=1, inplace=True)
valCount


# In[ ]:


# plot only the true '1' values under each labels in training dataset
valCount.loc[1].plot.barh(figsize=(15,5)).grid()


# In[ ]:


# extract comment_text column from train and test for vectorizing
trainComTxt = train_data['comment_text']
testComTxt = test_data['comment_text']
fullTxt = pd.concat([trainComTxt,testComTxt])


# In[ ]:


# create pipeline for TF-IDF vectorizer and Linear Support Vector Classifier for Grid search CV
txtClf = Pipeline([('tfidfvec', TfidfVectorizer(analyzer='word',stop_words='english',use_idf=True,smooth_idf=True)),
                    ('lsvc', LinearSVC())])


# In[ ]:


# set possible hyper parameters for TF-IDF and Linear Support Vector Classifier for Grid Search CV
params = {'tfidfvec__sublinear_tf': (True, False),
          'tfidfvec__ngram_range': [(1,1),(1,2)],
          'tfidfvec__max_features': (20000,30000),
          'lsvc__C':(1,2),
          'lsvc__loss':('hinge','squared_hinge')}


# In[ ]:


# create grid search CV object with pipeline and params
gsClf = GridSearchCV(txtClf, params, n_jobs=2)


# In[ ]:


#validate the training data with 20000 records
gstrain = train_data.iloc[:20000,1]
gstrain.head()
gstarget = train_data.iloc[:20000, 2:8]
gstarget.head(10)


# In[ ]:


# labels in list for iteration
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[ ]:


# Grid search CV for all label and find best score and best param for respective labels
for label in labels:
    target = gstarget[label]
    strTm = datetime.now()
    gsClf = gsClf.fit(gstrain, target)
    bestParam = gsClf.best_params_
    bestScore = gsClf.best_score_
    difTm = (datetime.now()-strTm)
    msg = "For %s, Best Params: %s and Best Score:%f, Time taken:%s sec." % (label,str(bestParam),bestScore,str(difTm.seconds))
    print msg


# In[ ]:


# TFidf vectorizer with best param identified from Grid search CV
tfIdf = TfidfVectorizer(smooth_idf=True, sublinear_tf=True, analyzer='word', use_idf=True, 
                        ngram_range=(1, 1), stop_words="english", max_features=30000)


# In[ ]:


# Fit and Tranform comment texts into TF-IDF vectors for train and test data
tfIdf.fit(fullTxt)
trainFeat = tfIdf.transform(trainComTxt)
testFeat = tfIdf.transform(testComTxt)


# In[ ]:


trainFeat.shape


# In[ ]:


testFeat.shape


# In[ ]:


# Cross validation of with training data with roc_auc scoring with Linear Support Vector Classifier parameter identified by Grid Search CV
for label in labels:
    trainTarg = train_data[label]
    cv = cross_val_score(LinearSVC(loss='hinge', C=1), trainFeat, train_data[label], cv=10, scoring='roc_auc')
    score = cv.mean()
    msg = "%s: %f" % (label, score)
    print msg


# In[ ]:


# Cross validation of with training data with roc_auc scoring with Linear Support Vector Classifier parameter identified by Grid Search CV
for label in labels:
    trainTarg = train_data[label]
    cv = cross_val_score(LinearSVC(loss='squared_hinge', C=1), trainFeat, train_data[label], cv=10, scoring='roc_auc')
    score = cv.mean()
    msg = "%s: %f" % (label, score)
    print msg


# In[ ]:


# create a dataframe store predicted probabilities
predResults = pd.DataFrame.from_dict({'Comment': test_data['comment_text']})


# In[ ]:


# Using Linear Support Vector Classifier model, fitting the training data for all labels & predicting probabilities for test data using Calibrated Classifier
for label in labels:
    trainTarg = train_data[label]
    lSVCcClf_h = CalibratedClassifierCV(base_estimator=LinearSVC(loss='hinge', C=1), cv=5)
    lSVCcClf_h.fit(trainFeat,trainTarg)
    predResults[label] = lSVCcClf_h.predict_proba(testFeat)[:,1]
    debugMsg = "Prediction completed for %s label" %label
    print debugMsg


# In[ ]:


# validate size of predicted results
predResults.shape


# In[ ]:


predResults.head()


# In[ ]:


# save the predicted results as csv file
predResults.to_csv('lsvc_Prediction_hinge.csv',index=False)


# In[ ]:


# create a dataframe store predicted probabilities
predResults = pd.DataFrame.from_dict({'Comment': test_data['comment_text']})


# In[ ]:


# Using Linear Support Vector Classifier model, fitting the training data for all labels & predicting probabilities, using Calibrated Classifier, for test data
for label in labels:
    trainTarg = train_data[label]
    lSVCcClf_sh = CalibratedClassifierCV(base_estimator=LinearSVC(loss='squared_hinge', C=1), cv=5)
    lSVCcClf_sh.fit(trainFeat,trainTarg)
    predResults[label] = lSVCcClf_sh.predict_proba(testFeat)[:,1]
    debugMsg = "Prediction completed for %s label" %label
    print debugMsg


# In[ ]:


# validate size of predicted results
predResults.shape


# In[ ]:


predResults.head()


# In[ ]:


# save the predicted results as csv file
predResults.to_csv('lsvc_Prediction_squared_hinge.csv',index=False)

