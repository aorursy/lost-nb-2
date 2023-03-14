#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ml_metrics import quadratic_weighted_kappa

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from joblib import Parallel, delayed
import multiprocessing

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(name, group) for name, group in dfGrouped)
    return pd.DataFrame(retLst)

###
# def get_last_event(group):
#     return group.sort_values('timestamp', ascending=False).iloc[0]
# last_events = test.groupby('installation_id').apply(get_last_event).event_id.value_counts()
# print(last_events.index) # ['7ad3efc6', '3bfd1a65', '90d848e0', '5b49460a', 'f56e0afc']
###
last_event_before_assessment = {'Cauldron Filler (Assessment)': '90d848e0',
                                'Cart Balancer (Assessment)': '7ad3efc6',
                                'Mushroom Sorter (Assessment)': '3bfd1a65',
                                'Bird Measurer (Assessment)': 'f56e0afc',
                                'Chest Sorter (Assessment)': '5b49460a'}

media_seq = pd.read_csv('../input/dsb-feats-v2/media_sequence.csv')
clips_seq = media_seq[media_seq.type=='Clip']
clip_times = dict(zip(clips_seq.title, clips_seq.duration))


def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))
    return train, test, train_labels, specs


def get_worst_score(group):
    return group.sort_values('accuracy_group').iloc[0]


def is_assessment(titles_series):
    def is_assessment_title(title):
        return "assessment" in title.lower()
    return titles_series.apply(lambda x: is_assessment_title(x))

def num_unique_days(timestamps):
    return pd.to_datetime(timestamps).apply(lambda x: x.date()).unique().size


def days_since_first_event(timestamps):
    dates = pd.to_datetime(timestamps).apply(lambda x: x.date())
    return (dates.max() - dates.min()).days


def get_events_before_game_session(events, game_session, assessment_title):
    if not (game_session or assessment_title):
        return events
    else:
        assessment_event = last_event_before_assessment.get(assessment_title)
        game_session_index = events.index[(events.game_session == game_session) &                                            (events.event_id.isin([assessment_event] if assessment_event else last_event_before_assessment.values()))]
        return events.loc[:game_session_index[-1]]


def get_clip_duration_features(events):
    clips = events[events.type=='Clip']
    if clips.empty:
        game_time = 0
        skip_rate = 0
        avg_watch_length = 0
    else:
        game_time = clips.apply(lambda x: min(x.ts_diff, clip_times.get(x.title)), axis=1).sum()
        skip_rate = clips.apply(lambda x: x.ts_diff < clip_times.get(x.title), axis=1).mean()
        avg_watch_length = clips.apply(lambda x: min(x.ts_diff / clip_times.get(x.title), 1), axis=1).mean()
    return pd.Series([game_time, skip_rate, avg_watch_length], 
                     index=['clip_game_time', 'clip_skip_rate', 'clip_avg_watch_length'],
                     dtype=float)
    
def group_by_game_session_and_sum(events, columns):
    """
    some columns are rolling counts by game session,
    take the max value of each game session then sum for totals
    """
    series = pd.Series(dtype=int)
    for c in columns:
        # set beginning values for each type to 0
        for stype in ['activity', 'game', 'assessment', 'clip']:
            series[stype+'_'+c] = 0
        series['total_'+c] = 0 
        
        # get session type and total values and add to running total
        for session_id, session in events.groupby('game_session'):
            session_type = session['type'].iloc[0].lower()
            session_value = session[c].max() / 1000.0 if c=='game_time' else session[c].max()
            series[session_type+'_'+c] += session_value
            series['total_'+c] += session_value
        if c=='game_time':
            series = series.drop(labels='clip_'+c)
            series = series.append(get_clip_duration_features(events))
    return series


def summarize_events(events):
    """
    takes a dataframe of events and returns a pd.Series with aggregate/summary values
    """
    events = events.sort_values('ts').reset_index()
    events['ts_diff'] = -events.ts.diff(-1).dt.total_seconds()
    numeric_rows = ['event_count', 'game_time']
    aggregates = group_by_game_session_and_sum(events, numeric_rows)
    aggregates['num_unique_days'] = num_unique_days(events['timestamp'])
    aggregates['elapsed_days'] = days_since_first_event(events['timestamp'])
    aggregates['last_world'] = events.tail(1)['world'].values[0]
    aggregates['last_assessment'] = events[is_assessment(events['title'])].tail(1)['title'].values[0]
    aggregates['assessments_taken'] = events['title'][events.event_id.isin(last_event_before_assessment.values())].value_counts()
    aggregates['type_counts'] = events[['game_session', 'type']].drop_duplicates()['type'].value_counts()
    aggregates['title_counts'] = events[['game_session', 'title']].drop_duplicates()['title'].value_counts()
    aggregates['event_code_counts'] = events.event_code.value_counts()
    aggregates['event_id_counts'] = events.event_id.value_counts()
    aggregates['unique_game_sessions'] = events.game_session.unique().size
    return aggregates


def summarize_events_before_game_session(name, events):
    if not isinstance(name, (list,tuple)) or len(name)==1:
        # for test data
        game_session=None
        assessment=None
        name_series = pd.Series([name], index=['installation_id'])
    else:
        installation_id, game_session, assessment = name
        name_series = pd.Series(name, index=['installation_id', 'game_session_y', 'title_y'])
    
    events = events.rename(columns={'game_session_x': 'game_session', 'title_x': 'title'}, errors='ignore')
    events_before = get_events_before_game_session(events, game_session, assessment)
    aggregates = summarize_events(events_before)
    try:
        labels = events[['num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']].iloc[0]             .append(name_series)
        row = aggregates.append(labels)
    except KeyError:
        row = aggregates.append(name_series)
        # print("no label columns, just returning features")
    return row


def expand_count_features(features):
    print('**expanding event type count features**')
    expanded_type_counts = features.type_counts.apply(pd.Series).fillna(0)
    # rename the type count columns
    expanded_type_counts.columns = [c.lower()+'_ct' for c in expanded_type_counts.columns]
    
    print('**expanding title count features**')
    expanded_title_counts = features.title_counts.apply(pd.Series).fillna(0)
    # rename the type count columns
    expanded_title_counts.columns = [c.lower().replace(' ', '_')+'_ct' for c in expanded_title_counts.columns]

    print('**expanding event code count features**')
    expanded_event_code_counts = features.event_code_counts.apply(pd.Series).fillna(0)
    # rename the event_code count columns
    expanded_event_code_counts.columns = ['event_{}_ct'.format(int(c)) for c in expanded_event_code_counts.columns]
    # non_zero_event_code_counts 
    for ec in expanded_event_code_counts.columns:
        expanded_event_code_counts['non_zero_'+ec] = (expanded_event_code_counts[ec] > 0).astype(int)
    
    print('**expanding event id count features**')
    expanded_event_id_counts = features.event_id_counts.apply(pd.Series).fillna(0)
    # rename the event_id count columns
    expanded_event_id_counts.columns = ['eid_{}_ct'.format(c) for c in expanded_event_id_counts.columns]
    
    expanded_assessments_taken = features.assessments_taken.apply(pd.Series).fillna(0)
    
    feats = pd.concat([features.drop(['type_counts', 'title_counts', 'event_code_counts', 'event_id_counts', 'assessments_taken'], axis=1), expanded_type_counts, expanded_title_counts, expanded_event_code_counts, expanded_event_id_counts, expanded_assessments_taken], axis=1)
    return feats


def split_features_and_labels(df):
    labels_df = df[['title_y', 'num_correct', 'num_incorrect',
                    'accuracy', 'accuracy_group', 'installation_id', 'game_session_y']].copy()
    feats_df = df.drop(
        ['title_y', 'num_correct', 'num_incorrect', 'game_session_y',
         'accuracy', 'accuracy_group'], axis=1)
    return feats_df, labels_df


def basic_user_features_transform(train_data, train_labels=None):
    data = train_data[['event_id', 'game_session', 'timestamp', 'installation_id', 'event_count', 'event_code',
                       'game_time', 'title', 'type', 'world']]
    data['ts'] = pd.to_datetime(data.timestamp)
    if train_labels is not None:
        train_w_labels = data.merge(train_labels, on='installation_id')
        groups = train_w_labels.groupby(['installation_id', 'game_session_y', 'title_y'])
    else:
        groups = data.groupby(['installation_id'])
    # game session y is index 1 of the group name
    # passing none to game session is for eval data, does not subset any of the data for each installation_id
    print('**getting user features before each training assessment**')
    features = applyParallel(groups,
                             lambda name, group: summarize_events_before_game_session(name, group))
    
    expanded_features = expand_count_features(features)
    

    if train_labels is not None:
        return split_features_and_labels(expanded_features)
    else:
        return expanded_features, None

def get_data_processing_pipe(feats, log_features, categorical_features):
    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = [c for c in feats.columns if c not in log_features+categorical_features+['installation_id']]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value=0, strategy='constant')),
        ('scaler', StandardScaler())])

    numeric_log_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value=0, strategy='constant')),
        ('log_scale', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        remainder='drop',
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('num_log', numeric_log_transformer, log_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor


# In[3]:


from sklearn.base import clone
import numpy as np
from collections import Counter

        
class OrdinalRegressor():    
    def __init__(self, clf, **kwargs):
        self.clf = clf(**kwargs)
#         self.dist = None
        self.threshold_optimizer = OptimizedRounder([0,1,2,3])
        
    def fit(self, X, y, **fit_params):
#         self.dist = Counter(y)
#         for k in self.dist:
#             self.dist[k] /= y.size
        self.clf.fit(X, y, **fit_params)
        self.threshold_optimizer.fit(self.predict(X), y)
    
    def predict(self, X, **predict_params):
        pred = self.clf.predict(X)
        if predict_params.get('classify'):
            return self.classify(pred)
        return pred
    
    def set_params(self, **kwargs):
        self.clf = self.clf.set_params(**kwargs)
        
    def classify(self, pred):
#         acum = 0
#         bound = {}
#         for i in range(3):
#             acum += self.dist[i]
#             bound[i] = np.percentile(pred, acum * 100)
#         # print('y_classify bounds:', bound)
        
#         def classify_example(x):
#             if x <= bound[0]:
#                 return 0
#             elif x <= bound[1]:
#                 return 1
#             elif x <= bound[2]:
#                 return 2
#             else:
#                 return 3
        
#         return list(map(classify_example, pred))
        return self.threshold_optimizer.predict(pred)
    
    def predict_and_classify(self, X):
        return self.classify(self.predict(X))
    
    def predict_proba(self, X):
        return self.predict_and_classify(X)
    
    def decision_function(self, X):
        return self.predict_and_classify(X)


# In[4]:


import pandas as pd
from functools import partial
from sklearn.metrics import cohen_kappa_score
import scipy as sp
import numpy as np

class OptimizedRounder(object):
    def __init__(self, labels):
        self.coef_ = 0
        self.labels = labels
    
    def _kappa_loss(self, coef, X, y):
#         print(coef)
        if len(set(coef)) != len(coef):
            return 0
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels)
        return -cohen_kappa_score(y, preds, weights = 'quadratic')
    
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5]
        constraints = ({'type': 'ineq', 'fun' : lambda x: x[1] - x[0] - 0.001},
                       {'type': 'ineq', 'fun' : lambda x: x[2] - x[1] - 0.001})
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'COBYLA', constraints=constraints)
    
    def predict(self, X, coef=None):
        coef = coef if coef else self.coefficients()
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels)
        return preds
    
    def coefficients(self):
        return self.coef_['x']


# In[5]:


feats = installation_features = pd.read_csv("../input/dsb-feats-v2/installation_features_v2.csv")
labels = installation_labels = pd.read_csv("../input/dsb-feats-v2/installation_labels_v2.csv")
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')


# In[6]:


print(feats.shape)
print(labels.shape)
print(test.shape)


# In[7]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import xgboost as xgb
from sklearn.utils import class_weight
from sklearn.ensemble import VotingRegressor
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape=[538], hidden_units=[64], learning_rate=0.003, dropout=0, l1=0, l2=0, epochs=20):
    model = keras.Sequential([
        layers.Input(input_shape)
    ])
    for hu in hidden_units:
        model.add(layers.Dense(hu, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2)))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

class KerasRegressor_v2(KerasRegressor):
    def __init__(self, build_fn, **kwargs):
        super().__init__(build_fn, **kwargs)
        self._estimator_type = "regressor"


# feature_pipe=features.get_data_processing_pipe(feats,log_features=['game_time', 'event_count'], categorical_features=['last_world', 'last_assessment'])
feature_pipe=get_data_processing_pipe(feats,
                                               log_features=list(filter(lambda c: c.startswith('event') or 
                                                                                  c.endswith('event_count') or
                                                                                  c.endswith('game_time') or
                                                                                  c.startswith('eid_'), 
                                                                        feats.columns)), 
                                               categorical_features=['last_world', 'last_assessment'])

xgb_params = {'colsample_bytree': 0.3, 
              'learning_rate': 0.03, 
              'max_depth': 7, 
              'n_estimators': 300, 
              'reg_alpha': 10, 
              'subsample': 0.8}
mlp_params = {
    'dropout': 0.1,
    'epochs': 20,
    'hidden_units': (128, 128),
    'l1': 0.001,
    'l2': 0.0,
    'learning_rate': 0.0001,
}  


ordinal_pipe = Pipeline(steps=[
    ('preprocess', feature_pipe),
    ('clf', OrdinalRegressor(VotingRegressor, estimators=[('xgb', xgb.XGBRegressor(**xgb_params)), 
                                                          ('mlp', KerasRegressor_v2(build_model, **mlp_params))],
                                              weights=(0.7,0.3)))])


# In[8]:


test_feats, _ = basic_user_features_transform(test)

for c in feats.columns:
    if c not in test_feats.columns:
        test_feats[c] = 0
        
installation_ids = test_feats.installation_id
test_X = test_feats


# In[9]:


ordinal_pipe.fit(feats[sorted(feats.columns)], labels.accuracy_group)


# In[10]:


test_predict = ordinal_pipe.predict(test_X[sorted(feats.columns)], **{'classify': True})


# In[11]:


pd.DataFrame({'installation_id':installation_ids, 'accuracy_group':test_predict}).to_csv('submission.csv', index=False)


# In[12]:


test_predict.value_counts()


# In[ ]:




