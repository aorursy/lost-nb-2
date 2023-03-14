#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
test_df = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
bureau_df = pd.read_csv('../input/home-credit-default-risk/bureau.csv')

primary_key = 'SK_ID_CURR'
target = 'TARGET'

def preprocessing(df, prefix, primary_key, drops):
    numbers = df.select_dtypes(include=np.number).columns
    objects = df.select_dtypes(include=np.object).columns
    ohes = objects
    les = [c for c in objects if c not in ohes]
    for c in numbers:
        if c == primary_key or c in drops:
            continue
        df[c] = df[c].fillna(df[c].median())
    for c in ohes:
        l = df[c].unique().tolist()
        if l.__contains__(np.nan):
            l.remove(np.nan)
        else:
            if len(l) == 2:
                df[c] = df[c].apply(lambda s: 1 if s == l[0] else 0)
                continue
        for v in l:
            df[str(c) + '_' + str(v)] = df[c].apply(lambda s: 1 if s == v else 0)
        df = df.drop([c], axis=1)
    for c in les:
        if c in drops:
            continue
        df[c] = df[c].fillna('NaN')
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
    df = df.drop(drops, axis=1)
    df = df.groupby(primary_key, as_index = False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    columns = [primary_key]
    for l1 in df.columns.levels[0]:
        if l1 ==primary_key or l1 in drops:
            continue
        for l2 in df.columns.levels[1][:-1]:
            columns.append(prefix + '_' + l1 + '_' + l2)
    df.columns = columns
    return df

bureau_df = preprocessing(bureau_df, 'breau', primary_key, ['SK_ID_BUREAU'])

bureau_df.head(5)


# In[2]:


train_df = train_df.merge(bureau_df, on = primary_key, how = 'left')
test_df = test_df.merge(bureau_df, on = primary_key, how = 'left')

numbers = train_df.select_dtypes(include=np.number).columns
objects = train_df.select_dtypes(include=np.object).columns
ohes = objects
les = [c for c in objects if c not in ohes]
drops = []

for c in numbers:
    if c in [primary_key, target] or c in drops:
        continue
    for df in [train_df, test_df]:
        df[c] = df[c].fillna(df[c].median())
        stdsc = StandardScaler()
        df[[c]] = stdsc.fit_transform(df[[c]])

for c in ohes:
    l = train_df[c].unique().tolist()
    if l.__contains__(np.nan):
        l.remove(np.nan)
    else:
        if len(l) == 2:
            for df in [train_df, test_df]:
                df[c] = df[c].apply(lambda s: 1 if s == l[0] else 0)
            continue
    for v in l:
        for df in [train_df, test_df]:
            df[str(c) + '_' + str(v)] = df[c].apply(lambda s: 1 if s == v else 0)
    [train_df, test_df] = [df.drop([c], axis=1) for df in [train_df, test_df]]

for c in les:
    if c in drops:
        continue
    for df in [train_df, test_df]:
        df[c] = df[c].fillna('NaN')
    le = LabelEncoder()
    le.fit(pd.concat([train_df[c], test_df[c]]))
    for df in [train_df, test_df]:
        df[c] = le.transform(df[c])
        stdsc = StandardScaler()
        df[[c]] = stdsc.fit_transform(df[[c]])

[train_df, test_df] = [df.drop(drops, axis=1) for df in [train_df, test_df]]

train_df.head(5)


# In[3]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

models = [
    # RandomForestClassifier(),
    # LogisticRegression(),
    # Perceptron(),
    # SGDClassifier(),
    # GaussianNB(),
    # KNeighborsClassifier(),
    # LinearSVC(),
    # SVC(),
    # DecisionTreeClassifier(),
    XGBClassifier(),
]

X, y = train_df.drop([primary_key, target], axis=1).values, train_df[target].values
X_train, X_test, y_train, y_test =     train_test_split(X, y,
                     test_size=0.3,
                     random_state=0,
                     stratify=y)

results = []

for m in models:
    scores = cross_val_score(estimator=m,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc',
                             n_jobs=1)
    m.fit(X_train, y_train)
    validation_score = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])

results.append({
        'Model': m.__class__.__name__,
        'Training CV ROC AUC': '%.4f +/- %.4f' % (np.mean(scores), np.std(scores)),
        'Validation ROC AUC': validation_score
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('Validation ROC AUC', ascending=False))

y_pred = m.predict_proba(test_df.drop(primary_key, axis=1).values)[:,1]

submission = pd.DataFrame({
        primary_key: test_df[primary_key],
        target: y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[4]:


name_list = train_df.drop([primary_key, target], axis=1).columns.tolist()
fi_list = m.feature_importances_.tolist()
fi_df = pd.DataFrame(list(zip(name_list, fi_list)), columns=['Column', 'FeatureImportance'])    .sort_values('FeatureImportance', ascending=False)

fi_df.head(20)

