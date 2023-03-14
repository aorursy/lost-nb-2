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

# Any results you write to the current directory are saved as output.


# In[2]:


df_train = pd.read_csv('../input/train.csv')
df_train.head()


# In[3]:


df_train.info()


# In[4]:


dtIdx = pd.DatetimeIndex(df_train['datetime'])


# In[5]:


df_train['hour'] = dtIdx.hour
df_train['dayofweek'] = dtIdx.dayofweek
df_train['month'] = dtIdx.month
df_origin = df_train


# In[6]:


df_train = df_train.drop(['casual', 'registered', 'datetime'], axis = 1)
df_train.head()


# In[7]:


df_train_data = df_train.drop('count', axis=1)


# In[8]:


df_train_target = df_train['count']


# In[9]:


df_train_target.head()


# In[10]:


from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection


# In[11]:


ms = model_selection.ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)


# In[12]:


'''
for idx_train, idx_test in ms.split(df_train_data):
    csv = linear_model.Ridge().fit(df_train_data.iloc[idx_train], \
                                   df_train_target.iloc[idx_train])
    print('train score: {0: .3f}, test score: {1: .3f}'.format(
        csv.score(df_train_data.iloc[idx_train], df_train_target.iloc[idx_train]),
        csv.score(df_train_data.iloc[idx_test], df_train_target.iloc[idx_test])
    ))
'''


# In[13]:


'''
for idx_train, idx_test in ms.split(df_train_data):
    csv = svm.SVR(kernel='rbf', C=10, gamma=0.001).fit(df_train_data.iloc[idx_train],\
                                                       df_train_target.iloc[idx_train])
    print('train score: {0: .3f}, test score: {1: .3f}'.format(
        csv.score(df_train_data.iloc[idx_train], df_train_target.iloc[idx_train]),
        csv.score(df_train_data.iloc[idx_test], df_train_target.iloc[idx_test])
    ))
'''


# In[14]:


#df_train_data_notime = df_train_data.drop(['hour', 'dayofweek', 'month'], axis=1)


# In[15]:


df_train_data.head()


# In[16]:


df_train_target.head()


# In[17]:


# for idx_train,idx_test in ms.split(df_train_data):
#     csv = RandomForestRegressor(n_estimators=500).fit(df_train_data.iloc[idx_train],\
#                                                      df_train_target.iloc[idx_train])
#     print('train score: {0: .3f}, test score: {1: .3f}'.format(
#         csv.score(df_train_data.iloc[idx_train], df_train_target.iloc[idx_train]),
#         csv.score(df_train_data.iloc[idx_test], df_train_target.iloc[idx_test])
#     ))


# In[18]:


'''
for idx_train,idx_test in ms.split(df_train_data):
    csv = RandomForestRegressor(n_estimators=100).fit(df_train_data_notime.iloc[idx_train],\
                                                     df_train_target.iloc[idx_train])
    print('train score: {0: .3f}, test score: {1: .3f}'.format(
        csv.score(df_train_data_notime.iloc[idx_train], df_train_target.iloc[idx_train]),
        csv.score(df_train_data_notime.iloc[idx_test], df_train_target.iloc[idx_test])
    ))
'''


# In[19]:


from sklearn.model_selection import GridSearchCV


# In[20]:


X_train, X_test, y_train, y_test = model_selection.    train_test_split(df_train_data, df_train_target, test_size = 0.2, random_state=0)


# In[21]:


'''
tuned_parameters = [{'n_estimators':[50, 100, 500]}]
scores = ['r2']
#for score in scores:
clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring='r2', verbose=2)
clf.fit(X_train, y_train)
print('best:', clf.best_estimator_)
for params, mean_train, mean_test, std_train, std_test in zip(clf.cv_results_['params'],
                                                    clf.cv_results_['mean_train_score'],
                                                    clf.cv_results_['mean_test_score'],
                                                    clf.cv_results_['std_train_score'],
                                                    clf.cv_results_['std_test_score']):
    print("params:", params)
    print('mean train score: %.3f'%mean_train)
    print('mean test score: %.3f'%mean_test)
    print('std train scores: %.3f'%std_train)
    print('std test scores: %.3f'%std_test)
'''


# In[22]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
# title = "Learning Curves (RFR, n_estimators=100)"
# estimator = RandomForestRegressor(n_estimators=100)
# plot_learning_curve(estimator, title, 
#                     df_train_data, df_train_target, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
# plt.show()


# In[23]:


# estimator2 = RandomForestRegressor(n_estimators=200, max_features=0.6, max_depth=15)
# plot_learning_curve(estimator2, title, 
#                    df_train_data, df_train_target, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
# plt.show()


# In[24]:


df_origin.columns


# In[25]:


df_origin.groupby('windspeed').mean().plot(y='count', marker='o')


# In[26]:


df_origin.groupby('humidity').mean().plot(y='count', marker='o')


# In[27]:


d = df_origin.groupby('humidity')


# In[28]:


corr = df_origin[['temp','weather','windspeed','dayofweek', 'month', 'hour','count']].corr()
corr


# In[29]:


import matplotlib.pyplot as plt
plt.figure()
plt.matshow(corr)
plt.colorbar()
plt.show()


# In[30]:


df_test = pd.read_csv('../input/test.csv')
df_test.head()


# In[31]:


df_sample = pd.read_csv('../input/sampleSubmission.csv')
df_sample.head()


# In[32]:


df_test['hour'] = pd.DatetimeIndex(df_test['datetime']).hour
df_test['dayofweek'] = pd.DatetimeIndex(df_test['datetime']).dayofweek
df_test['month'] = pd.DatetimeIndex(df_test['datetime']).month
df_test_data = df_test.drop(['datetime'], axis=1)
df_test_data.head()


# In[33]:





# In[33]:


rfr = RandomForestRegressor(n_estimators=500).fit(df_train_data, df_train_target)


# In[34]:


score = rfr.score(df_train_data, df_train_target)


# In[35]:


print("score: %.3f"%score)


# In[36]:


df_sample['count'] = rfr.predict(df_test_data)


# In[37]:


df_sample.head()


# In[38]:


#df_sample['count'] = df_sample['count'].apply(lambda x: int(x + 0.5))


# In[39]:


df_sample.head()


# In[40]:


df_sample.info()


# In[41]:


df_sample.to_csv('submission.csv', index=False)


# In[42]:


df_demo = pd.read_csv('submission.csv')
df_demo.head()


# In[43]:





# In[43]:




