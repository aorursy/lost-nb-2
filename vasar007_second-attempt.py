#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import os
import re
import time
import warnings

from collections import Counter
from functools import partial

import numpy as np
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.model_selection import GroupKFold

from category_encoders.ordinal import OrdinalEncoder
from typing import Dict, Any

from tqdm import tqdm
from numba import jit


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
pd.options.display.precision = 15
pd.set_option('max_rows', 500)


# In[3]:


@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


# In[4]:


def round_values(y_pred):
    # This coefficients were calculated using emperical approach.
    lower_bound = 1.12232214
    middle_bound = 1.73925866
    upper_bound = 2.22506454

    y_pred[y_pred <= lower_bound] = 0

    y_pred[
        np.where(np.logical_and(y_pred > lower_bound, y_pred <= middle_bound))
    ] = 1

    y_pred[
        np.where(np.logical_and(y_pred > middle_bound, y_pred <= upper_bound))
    ] = 2

    y_pred[y_pred > upper_bound] = 3

    return y_pred
  


# In[5]:


def eval_qwk_lgb(y_true, y_pred):
    y_pred = round_values(y_pred)
    return 'cappa', qwk(y_true, y_pred), True


class LGBWrapper:

    def __init__(self):
        self.model = lgb.LGBMRegressor()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None,
            y_holdout=None, params=None):
        if params['objective'] == 'regression':
            eval_metric = eval_qwk_lgb
        else:
            eval_metric = 'auc'

        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        self.model = self.model.set_params(**params)

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_names.append('valid')

        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))
            eval_names.append('holdout')

        if 'cat_cols' in params.keys():
            cat_cols = [
                col for col in params['cat_cols'] if col in X_train.columns
            ]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = 'auto'
        else:
            categorical_columns = 'auto'

        self.model.fit(
            X=X_train,
            y=y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric=eval_metric,
            verbose=params['verbose'],
            early_stopping_rounds=params['early_stopping_rounds'],
            categorical_feature=categorical_columns
        )

        self.best_score_ = self.model.best_score_
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X_test):
        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)


# In[6]:


class MainTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, convert_cyclical: bool=False,
                 create_interactions: bool=False,
                 n_interactions: int=20):
        self.convert_cyclical = convert_cyclical
        self.create_interactions = create_interactions
        self.feats_for_interaction = None
        self.n_interactions = n_interactions

    def fit(self, X, y=None):
        if self.create_interactions:
            self.feats_for_interaction = [
                col for col in X.columns
                if "sum" in col or
                   "mean" in col or
                   "max" in col or
                   "std" in col or
                   "attempt" in col
            ]

            self.feats_for_interaction1 = np.random.choice(self.feats_for_interaction, self.n_interactions)
            self.feats_for_interaction2 = np.random.choice(self.feats_for_interaction, self.n_interactions)

        return self

    def transform(self, X, y=None):
        data = copy.deepcopy(X)
        if self.create_interactions:
            for col1 in self.feats_for_interaction1:
                for col2 in self.feats_for_interaction2:
                    data[f"{col1}_int_{col2}"] = data[col1] * data[col2]

        if self.convert_cyclical:
            data["timestampHour"] = np.sin(2 * np.pi * data["timestampHour"] / 23.0)
            data["timestampMonth"] = np.sin(2 * np.pi * data["timestampMonth"] / 23.0)
            data["timestampWeek"] = np.sin(2 * np.pi * data["timestampWeek"] / 23.0)
            data["timestampMinute"] = np.sin(2 * np.pi * data["timestampMinute"] / 23.0)
        return data

    def fit_transform(self, X, y=None, **fit_params):
        data = copy.deepcopy(X)
        self.fit(data)
        return self.transform(data)


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, main_cat_features: list=None, num_cols: list=None):
        self.main_cat_features = main_cat_features
        self.num_cols = num_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = copy.deepcopy(X)
        return data

    def fit_transform(self, X, y=None, **fit_params):
        data = copy.deepcopy(X)
        self.fit(data)
        return self.transform(data)


# In[7]:


class RegressorModel:

    def __init__(self, columns: list=None, model_wrapper=None):
        self.columns = columns
        self.model_wrapper = model_wrapper
        self.result_dict = {}
        self.train_one_fold = False
        self.preprocesser = None

    def fit(self,
            X: pd.DataFrame,
            y,
            X_holdout: pd.DataFrame=None,
            y_holdout=None,
            folds=None,
            params: dict=None,
            eval_metric: str='rmse',
            cols_to_drop: list=None,
            preprocesser=None,
            transformers: dict=None,
            adversarial: bool=False,
            plot: bool=True):
        if folds is None:
            folds = KFold(n_splits=3, random_state=42)
            self.train_one_fold = True

        self.columns = X.columns if self.columns is None else self.columns
        self.feature_importances = pd.DataFrame(columns=['feature', 'importance'])
        self.trained_transformers = {k: [] for k in transformers}
        self.transformers = transformers
        self.models = []
        self.folds_dict = {}
        self.eval_metric = eval_metric
        n_target = 1
        self.oof = np.zeros((len(X), n_target))
        self.n_target = n_target

        X = X[self.columns]
        if X_holdout is not None:
            X_holdout = X_holdout[self.columns]

        if preprocesser is not None:
            self.preprocesser = preprocesser
            self.preprocesser.fit(X, y)
            X = self.preprocesser.transform(X, y)
            self.columns = X.columns.tolist()
            if X_holdout is not None:
                X_holdout = self.preprocesser.transform(X_holdout)

        for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y, X['installation_id'])):

            if X_holdout is not None:
                X_hold = X_holdout.copy()
            else:
                X_hold = None
            self.folds_dict[fold_n] = {}
            if params['verbose']:
                print(f'Fold {fold_n + 1} started at {time.ctime()}')
            self.folds_dict[fold_n] = {}

            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            if self.train_one_fold:
                X_train = X[self.columns]
                y_train = y
                X_valid = None
                y_valid = None

            datasets = {'X_train': X_train, 'X_valid': X_valid, 'X_holdout': X_hold, 'y_train': y_train}
            X_train, X_valid, X_hold = self.transform_(datasets, cols_to_drop)

            self.folds_dict[fold_n]['columns'] = X_train.columns.tolist()

            model = copy.deepcopy(self.model_wrapper)

            if adversarial:
                X_new1 = X_train.copy()
                if X_valid is not None:
                    X_new2 = X_valid.copy()
                elif X_holdout is not None:
                    X_new2 = X_holdout.copy()
                X_new = pd.concat([X_new1, X_new2], axis=0)
                y_new = np.hstack((np.zeros((X_new1.shape[0])), np.ones((X_new2.shape[0]))))
                X_train, X_valid, y_train, y_valid = train_test_split(X_new, y_new)

            model.fit(X_train, y_train, X_valid, y_valid, X_hold, y_holdout, params=params)

            self.folds_dict[fold_n]['scores'] = model.best_score_
            if self.oof.shape[0] != len(X):
                self.oof = np.zeros((X.shape[0], self.oof.shape[1]))
            if not adversarial:
                self.oof[valid_index] = model.predict(X_valid).reshape(-1, n_target)

            fold_importance = pd.DataFrame(
                list(zip(X_train.columns, model.feature_importances_)),
                columns=['feature', 'importance']
            )
            self.feature_importances = self.feature_importances.append(fold_importance)
            self.models.append(model)

        self.feature_importances['importance'] = self.feature_importances['importance'].astype(int)

        self.calc_scores_()

        if plot:
            fig, ax = plt.subplots(figsize=(16, 12))
            plt.subplot(2, 2, 1)
            self.plot_feature_importance(top_n=20)
            plt.subplot(2, 2, 2)
            self.plot_metric()
            plt.subplot(2, 2, 3)
            plt.hist(y.values.reshape(-1, 1) - self.oof)
            plt.title('Distribution of errors')
            plt.subplot(2, 2, 4)
            plt.hist(self.oof)
            plt.title('Distribution of oof predictions')

    def transform_(self, datasets, cols_to_drop):
        for name, transformer in self.transformers.items():
            transformer.fit(datasets['X_train'], datasets['y_train'])
            datasets['X_train'] = transformer.transform(datasets['X_train'])
            if datasets['X_valid'] is not None:
                datasets['X_valid'] = transformer.transform(datasets['X_valid'])
            if datasets['X_holdout'] is not None:
                datasets['X_holdout'] = transformer.transform(datasets['X_holdout'])
            self.trained_transformers[name].append(transformer)
        if cols_to_drop is not None:
            cols_to_drop = [col for col in cols_to_drop if col in datasets['X_train'].columns]

            datasets['X_train'] = datasets['X_train'].drop(cols_to_drop, axis=1)
            if datasets['X_valid'] is not None:
                datasets['X_valid'] = datasets['X_valid'].drop(cols_to_drop, axis=1)
            if datasets['X_holdout'] is not None:
                datasets['X_holdout'] = datasets['X_holdout'].drop(cols_to_drop, axis=1)
        self.cols_to_drop = cols_to_drop

        return datasets['X_train'], datasets['X_valid'], datasets['X_holdout']

    def calc_scores_(self):
        print()
        datasets = [k for k, v in [v['scores'] for k, v in self.folds_dict.items()][0].items() if len(v) > 0]
        self.scores = {}
        for d in datasets:
            scores = [v['scores'][d][self.eval_metric] for k, v in self.folds_dict.items()]
            print(f"CV mean score on {d}: {np.mean(scores):.4f} +/- {np.std(scores):.4f} std.")
            self.scores[d] = np.mean(scores)

    def predict(self, X_test, averaging: str='usual'):
        full_prediction = np.zeros((X_test.shape[0], self.oof.shape[1]))
        if self.preprocesser is not None:
            X_test = self.preprocesser.transform(X_test)
        for i in range(len(self.models)):
            X_t = X_test.copy()
            for name, transformers in self.trained_transformers.items():
                X_t = transformers[i].transform(X_t)

            if self.cols_to_drop is not None:
                cols_to_drop = [col for col in self.cols_to_drop if col in X_t.columns]
                X_t = X_t.drop(cols_to_drop, axis=1)
            y_pred = self.models[i].predict(X_t[self.folds_dict[i]['columns']]).reshape(-1, full_prediction.shape[1])

            # If case transformation changes the number of the rows.
            if full_prediction.shape[0] != len(y_pred):
                full_prediction = np.zeros((y_pred.shape[0], self.oof.shape[1]))

            if averaging == 'usual':
                full_prediction += y_pred
            elif averaging == 'rank':
                full_prediction += pd.Series(y_pred).rank().values

        return full_prediction / len(self.models)

    def plot_feature_importance(self, drop_null_importance: bool=True,
                                top_n: int=10):
        top_feats = self.get_top_features(drop_null_importance, top_n)
        feature_importances = self.feature_importances.loc[self.feature_importances['feature'].isin(top_feats)]
        feature_importances['feature'] = feature_importances['feature'].astype(str)
        top_feats = [str(i) for i in top_feats]
        sns.barplot(data=feature_importances, x='importance', y='feature', orient='h', order=top_feats)
        plt.title('Feature importances')

    def get_top_features(self, drop_null_importance: bool=True, top_n: int=10):
        grouped_feats = self.feature_importances.groupby(['feature'])['importance'].mean()
        if drop_null_importance:
            grouped_feats = grouped_feats[grouped_feats != 0]
        return list(grouped_feats.sort_values(ascending=False).index)[:top_n]

    def plot_metric(self):
        full_evals_results = pd.DataFrame()
        for model in self.models:
            evals_result = pd.DataFrame()
            for k in model.model.evals_result_.keys():
                evals_result[k] = model.model.evals_result_[k][self.eval_metric]
            evals_result = evals_result.reset_index().rename(columns={'index': 'iteration'})
            full_evals_results = full_evals_results.append(evals_result)

        full_evals_results = full_evals_results            .melt(id_vars=['iteration'])            .rename(columns={'value': self.eval_metric, 'variable': 'dataset'})

        sns.lineplot(
            data=full_evals_results,
            x='iteration',
            y=self.eval_metric,
            hue='dataset'
        )
        plt.title('Training progress')


# In[8]:


class ReadResult:

    def __init__(self, train, test, train_labels, specs, sample_submission):
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.specs = specs
        self.sample_submission = sample_submission


class EncodedResult:

    def __init__(self, train, test, train_labels, win_code,
                 list_of_user_activities, list_of_event_code,
                 activities_labels, assess_titles, list_of_event_id,
                 all_title_event_code):
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.win_code = win_code
        self.list_of_user_activities = list_of_user_activities
        self.list_of_event_code = list_of_event_code
        self.activities_labels = activities_labels
        self.assess_titles = assess_titles
        self.list_of_event_id = list_of_event_id
        self.all_title_event_code = all_title_event_code


# In[9]:


def read_data(input_path: str, additional_path: str=None):
    if additional_path is None:
        additional_path = input_path

    print("Reading train.csv file....")
    train = pd.read_csv(f"{additional_path}/train.csv")
    print(f"Training.csv file have {train.shape[0]} rows and {train.shape[1]} columns")

    print("Reading test.csv file....")
    test = pd.read_csv(f"{input_path}/test.csv")
    print(f"Test.csv file have {test.shape[0]} rows and {test.shape[1]} columns")

    print("Reading train_labels.csv file....")
    train_labels = pd.read_csv(f"{input_path}/train_labels.csv")
    print(f"Train_labels.csv file have {train_labels.shape[0]} rows and {train_labels.shape[1]} columns")

    print("Reading specs.csv file....")
    specs = pd.read_csv(f"{input_path}/specs.csv")
    print(f"Specs.csv file have {specs.shape[0]} rows and {specs.shape[1]} columns")

    print("Reading sample_submission.csv file....")
    sample_submission = pd.read_csv(f"{input_path}/sample_submission.csv")
    print(f"Sample_submission.csv file have {sample_submission.shape[0]} rows and {sample_submission.shape[1]} columns.")

    return ReadResult(train, test, train_labels, specs, sample_submission)


# In[10]:


# Read data.
read_result = read_data(input_path="../input/data-science-bowl-2019")


# In[11]:


read_result.train.head(n=10)


# In[12]:


def encode_title(train, test, train_labels):
    # Encode title.
    train["title_event_code"] = list(map(lambda x, y: str(x) + "_" + str(y), train["title"], train["event_code"]))
    test["title_event_code"] = list(map(lambda x, y: str(x) + "_" + str(y), test["title"], test["event_code"]))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # Make a list with all the unique "titles" from the train and test set.
    list_of_user_activities = list(set(train["title"].unique()).union(set(test["title"].unique())))

    # Make a list with all the unique "event_code" from the train and test set.
    list_of_event_code = list(set(train["event_code"].unique()).union(set(test["event_code"].unique())))
    list_of_event_id = list(set(train["event_id"].unique()).union(set(test["event_id"].unique())))

    # Make a list with all the unique worlds from the train and test set.
    list_of_worlds = list(set(train["world"].unique()).union(set(test["world"].unique())))

    # Create a dictionary numerating the titles.
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train["type"] == "Assessment"]["title"].value_counts().index).union(set(test[test["type"] == "Assessment"]["title"].value_counts().index)))

    # Replace the text titles with the number titles from the dict.
    train["title"] = train["title"].map(activities_map)
    test["title"] = test["title"].map(activities_map)
    train["world"] = train["world"].map(activities_world)
    test["world"] = test["world"].map(activities_world)
    train_labels["title"] = train_labels["title"].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100 * np.ones(len(activities_map))).astype("int")))

    # Then, it set one element, the "Bird Measurer (Assessment)" as 4110, 10 more than the rest
    win_code[activities_map["Bird Measurer (Assessment)"]] = 4110

    # Convert text into datetime.
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])

    return EncodedResult(
        train, test, train_labels, win_code, list_of_user_activities,
        list_of_event_code, activities_labels, assess_titles, list_of_event_id,
        all_title_event_code
    )


# In[13]:


# Get usefull dict with maping encode.
encoded_result = encode_title(
    read_result.train, read_result.test, read_result.train_labels
)


# In[14]:


encoded_result.train.head(n=10)


# In[15]:


def get_data(user_sample, encoded_result: EncodedResult, test_set=False):
    """
    The user_sample is a DataFrame from train or test where the only one
    installation_id is filtered.
    And the test_set parameter is related with the labels processing, that is
    only requered if test_set=False.
    """
    # Constants and parameters declaration.
    last_activity = 0

    user_activities_count = {"Clip": 0, "Activity": 0, "Assessment": 0, "Game": 0}

    # New features: time spent in each activity.
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    durations = []
    last_accuracy_title = {"acc_" + title: -1 for title in encoded_result.assess_titles}

    # These 4 variables are Dict[str, int].
    event_code_count = {ev: 0 for ev in encoded_result.list_of_event_code}
    event_id_count = {eve: 0 for eve in encoded_result.list_of_event_id}
    title_count = {eve: 0 for eve in encoded_result.activities_labels.values()}
    title_event_code_count = {t_eve: 0 for t_eve in encoded_result.all_title_event_code}

    # Itarates through each session of one instalation_id.
    for i, session in user_sample.groupby("game_session", sort=False):
        # session is a DataFrame that contain only one game_session.

        # Get some sessions information.
        session_type = session["type"].iloc[0]
        session_title = session["title"].iloc[0]
        session_title_text = encoded_result.activities_labels[session_title]

        # For each assessment, and only this kind off session, the features
        # below are processed and a register are generated.
        if (session_type == "Assessment") & (test_set or len(session) > 1):
            # Search for event_code 4100, that represents the assessments
            # trial.
            all_attempts = session.query(f"event_code == {encoded_result.win_code[session_title]}")

            # Then, check the numbers of wins and the number of losses.
            true_attempts = all_attempts["event_data"].str.contains("true").sum()
            false_attempts = all_attempts["event_data"].str.contains("false").sum()

            # Copy a dict to use as feature template,
            # it's initialized with some itens:
            # {"Clip": 0, "Activity": 0, "Assessment": 0, "Game": 0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())

            # Get installation_id for aggregated features.
            features["installation_id"] = session["installation_id"].iloc[-1]

            # Add title as feature, remembering that title represents the name
            # of the game.
            features["session_title"] = session["title"].iloc[0]

            # The 4 lines below add the feature of the history of the trials of
            # this player.
            # It is based on the all time attempts so far, at the moment of
            # this assessment.
            features["accumulated_correct_attempts"] = accumulated_correct_attempts
            features["accumulated_uncorrect_attempts"] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far
            if durations == []:
                features["duration_mean"] = 0
            else:
                features["duration_mean"] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)

            # The accurace is the all time wins divided by the all time
            # attempts.
            features["accumulated_accuracy"] = accumulated_accuracy / counter if counter > 0 else 0
            accuracy = true_attempts / (true_attempts+false_attempts) if (true_attempts + false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title["acc_" + session_title_text] = accuracy

            # A feature of the current accuracy categorized
            # It is a counter of how many times this player was in each
            # accuracy group
            if accuracy == 0:
                features["accuracy_group"] = 0
            elif accuracy == 1:
                features["accuracy_group"] = 3
            elif accuracy == 0.5:
                features["accuracy_group"] = 2
            else:
                features["accuracy_group"] = 1
            features.update(accuracy_groups)
            accuracy_groups[features["accuracy_group"]] += 1

            # Mean of the all accuracy groups of this player.
            features["accumulated_accuracy_group"] = accumulated_accuracy_group / counter if counter > 0 else 0
            accumulated_accuracy_group += features["accuracy_group"]

            # How many actions the player has done so far, it is initialized
            # as 0 and updated some lines below.
            features["accumulated_actions"] = accumulated_actions

            # There are some conditions to allow this features to be inserted
            # in the datasets.
            # If it's a test set, all sessions belong to the final dataset.
            # It it's a train, needs to be passed throught this clausule:
            # session.query(f"event_code == {win_code[session_title]}").
            # That means, must exist an event_code 4100 or 4110.
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # This piece counts how many actions was made in each event_code so
        # far.
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == "title":
                        x = encoded_result.activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter

        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, "title")
        title_event_code_count = update_counters(title_event_code_count, "title_event_code")

        # Counts how many actions the player has done so far, used in the
        # feature of the same name.
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1

    # If it's the test_set, only the last assessment must be predicted,
    # the previous are scraped.
    if test_set:
        return all_assessments[-1]

    # In the train_set, all assessments goes to the dataset.
    return all_assessments


def get_train_and_test(train, test, encoded_result: EncodedResult):
    compiled_train = []
    compiled_test = []

    range_train = tqdm(
        train.groupby("installation_id", sort=False), total=17000
    )
    for (ins_id, user_sample) in range_train:
        compiled_train += get_data(user_sample, encoded_result)

    range_test = tqdm(
        test.groupby("installation_id", sort=False), total=1000
    )
    for ins_id, user_sample in range_test:
        test_data = get_data(user_sample, encoded_result, test_set=True)
        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    categoricals = ["session_title"]

    return reduce_train, reduce_test, categoricals


# In[16]:


# Tranform function to get the train and test set.
reduce_train, reduce_test, categoricals = get_train_and_test(
   encoded_result.train, encoded_result.test, encoded_result
)


# In[17]:


reduce_train.head(n=10)


# In[18]:


def preprocess(reduce_train, reduce_test, assess_titles):
    for df in [reduce_train, reduce_test]:
        df["installation_session_count"] = df.groupby(["installation_id"])["Clip"].transform("count")
        df["installation_duration_mean"] = df.groupby(["installation_id"])["duration_mean"].transform("mean")
        df["installation_title_nunique"] = df.groupby(["installation_id"])["session_title"].transform("nunique")

        df["sum_event_code_count"] = df[
            [2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080,
             2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 4022, 4025, 4030,
             4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020,
             4070, 2025, 2030, 4080, 2035, 2040, 4090, 4220, 4095]
            ].sum(axis=1)

        df["installation_event_code_count_mean"] = df.groupby(["installation_id"])["sum_event_code_count"].transform("mean")

        # Remove invalid characters from titles for json serialization.
        df.columns = [
            "".join(c if c.isalnum() else "_" for c in str(x))
            for x in df.columns
        ]

    # Delete useless columns.
    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns
    features = [
        x for x in features
        if x not in ["accuracy_group", "installation_id"]
    ] + ["acc_" + title for title in assess_titles]

    return reduce_train, reduce_test, features


# In[19]:


# Call feature engineering function.
reduce_train, reduce_test, features = preprocess(
    reduce_train, reduce_test, encoded_result.assess_titles
)
reduce_train.columns = reduce_train.columns.str.replace(",", "")


# In[20]:


reduce_train.head(n=10)


# In[21]:


# Define parameters.
params = {
    "n_estimators": 2000,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "subsample": 0.75,
    "subsample_freq": 1,
    "learning_rate": 0.04,
    "feature_fraction": 0.9,
    "max_depth": 15,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "verbose": 100,
    "early_stopping_rounds": 100,
    "eval_metric": "cappa"
}


# In[22]:


# Define training target.
y = reduce_train["accuracy_group"]


# In[23]:


# Split data on test and train set.
n_fold = 5
folds = GroupKFold(n_splits=n_fold)


# In[24]:


# Define colums to drop.
cols_to_drop = [
    "game_session", "installation_id", "timestamp",
    "accuracy_group", "timestampDate"
]


# In[25]:


# Train model.
mt = MainTransformer()
ft = FeatureTransformer()
transformers = {
    "ft": ft
}

regressor_model = RegressorModel(
    model_wrapper=LGBWrapper()
)

regressor_model.fit(
    X=reduce_train,
    y=y,
    folds=folds,
    params=params,
    preprocesser=mt,
    transformers=transformers,
    eval_metric="cappa",
    cols_to_drop=cols_to_drop
)


# In[26]:


class OptimizedRounder:

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])
        return -qwk(y, X_p)

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

    def coefficients(self):
        return self.coef_["x"]


# In[27]:


get_ipython().run_cell_magic('time', '', '\n# Make predictions on for marked data.\npr = regressor_model.predict(reduce_train)\n\noptR = OptimizedRounder()\noptR.fit(pr.reshape(-1,), y)\ncoefficients = optR.coefficients()')


# In[28]:


# Check metric value.
opt_preds = optR.predict(pr.reshape(-1, ), coefficients)
qwk(y, opt_preds)


# In[29]:


# Make predictions for test data without answers.
pr = regressor_model.predict(reduce_test)
pr = round_values(pr)


# In[30]:


# Save predictions to file with appropriate format.
sample_submission = read_result.sample_submission

sample_submission["accuracy_group"] = pr.astype(int)
sample_submission.to_csv("submission.csv", index=False)


# In[31]:


sample_submission.head(n=10)


# In[32]:


sample_submission['accuracy_group'].value_counts(normalize=True)

