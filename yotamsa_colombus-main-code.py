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


import pandas as pd
import numpy as np
import os
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
seed = 123


# In[3]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/SampleSubmission.csv")
label = "NESHER"
bad_features = ["DAPAR","KABA","TZADAK","TZIYUN_HATAMA_MM","IND_MATIM_MM","IND_OVER_MINIMAL_REQUIREMENTS_MM",
                "TZIYUN_HATAMA_MP","IND_MATIM_MP","IND_OVER_MINIMAL_REQUIREMENTS_MP","MADAD_EITANUT"]
app = pd.read_csv("../input/applications.csv")


# In[4]:


print("Train shape: "+str(train_data.shape))
print("Test shape: "+str(test_data.shape))


# In[5]:


def add_wills_features(df_train):
    ##adds wills features, returns a dataframe that contains the new columns
    ##need to join with the original dataframe and keep the original values 
    dests_wills_cols = [col for col in df_train.columns if "_WILL" in col and "DESTINATION" not in col and "LOHEM" not in col]
    df_train['dests_wills_max'] = df_train[dests_wills_cols].max(axis = 1)
    df_train['dests_wills_sum'] = df_train[dests_wills_cols].sum(axis = 1)
    df_train['dests_wills_all_nan'] = df_train['dests_wills_max'].apply(lambda x: int(x==0))
    df_train['dests_wills_non_zero_count'] = df_train[dests_wills_cols].astype(bool).sum(axis=1)
    df_train['dests_wills_mean_without0'] = df_train[['dests_wills_sum','dests_wills_non_zero_count']].apply(lambda x: x[0]/x[1], axis = 1)
    df_train['dests_wills_mean_with0'] = df_train[dests_wills_cols].mean(axis = 1)
add_wills_features(train_data)
add_wills_features(test_data)


# In[6]:


cols = ["DESTINATION", "KAHAS_TWO_MONTHS_BEFORE_GIYUS", 'LOHEM_WILL', 'DESTINATION_WILL']
dests_details = pd.concat([train_data[cols],test_data[cols]])    .groupby(by = "DESTINATION", sort = False).agg({"KAHAS_TWO_MONTHS_BEFORE_GIYUS":"mean",
                                                    "LOHEM_WILL":"mean",
                                                    "DESTINATION_WILL":"mean"})
dests_details.columns = ["dest_details_dest_will", "dest_details_lohem_will", "dest_details_kahas"]
train_data = train_data.join(dests_details, on = 'DESTINATION')
test_data = test_data.join(dests_details, on = 'DESTINATION')


# In[7]:


app.Begin_date = pd.to_datetime(app.Begin_date.apply(lambda x: '2'+x[1:]), format='%Y-%m-%d %H:%M:%S')
app.End_date = pd.to_datetime(app.End_date.apply(lambda x: '2'+x[1:]), format='%Y-%m-%d %H:%M:%S')
app['date_diff'] = app.End_date - app.Begin_date

def f(x):
    d = {}
    d['num_apps'] = x.shape[0]
    d['num_in'] = sum(x.Incident_direct == 'נכנס')
    d['num_out'] = sum(x.Incident_direct == 'יוצא')
    d['in_percentage'] = d['num_in'] / d['num_apps']
    d['arotz_phone_percentage'] = sum(x.ArotzPnia == 'טלפון') / d['num_apps']
    d['arotz_fax_percentage'] = sum(x.ArotzPnia == 'פקס') / d['num_apps']
    d['arotz_mail_percentage'] = sum(x.ArotzPnia == 'מייל') / d['num_apps']
    d['arotz_internet_percentage'] = sum(x.ArotzPnia == 'אינטרנט') / d['num_apps']
    d['arotz_personal_percentage'] = sum(x.ArotzPnia == 'מסירה אישית') / d['num_apps']
    d['arotz_other_percentage'] = 1 - (d['arotz_phone_percentage']+d['arotz_fax_percentage']+d['arotz_mail_percentage']+
                                       d['arotz_internet_percentage']+d['arotz_personal_percentage'])
    d['mean_date_diff'] = x.date_diff.mean().seconds
    d['max_date_diff'] = x.date_diff.max().seconds
    return pd.Series(d, index=d.keys())
app_features = app.groupby("CustomerID").apply(f)

train_data = train_data.set_index("TZ").join(app_features)
test_data = test_data.set_index("TZ").join(app_features)
train_data = train_data.reset_index()
test_data = test_data.reset_index()


# In[8]:


def kaba_as_feature(df, quality_col_name, other_col_name):
#     df[other_col_name] = df[other_col_name] + 1 
    new_col_name = quality_col_name + '_' + other_col_name
    df[new_col_name] = df[quality_col_name] / df[other_col_name] + 1
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df[new_col_name].max())
    return df


# In[9]:


train_data = kaba_as_feature(train_data, "KABA", "LOHEM_WILL")
test_data = kaba_as_feature(test_data, "KABA", "LOHEM_WILL")


# In[10]:


print("Train shape: "+str(train_data.shape))
print("Test shape: "+str(test_data.shape))


# In[11]:


features = list(set(train_data.columns) -                set(["MISPAR_ISHI", "TZ", "DESTINATION", "MAHZOR_ACHARON", "NESHER"]) -                set(bad_features))


# In[12]:


n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
auc = []

param = {'max_depth': 5,
         'eta': 0.2,
         'silent': 1,
         'objective': 'binary:logistic', 
         'nthread':8,
         "eval_metric":'auc',
         "seed": seed}
num_round = 100

for i,(train,val) in enumerate(kf.split(train_data)):
    print("Fold {} out of {}".format(i+1, n_splits))
    dtrain = xgb.DMatrix(train_data.loc[train, features], label=train_data.loc[train, label])
    dval = xgb.DMatrix(train_data.loc[val, features], label=train_data.loc[val, label])
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(param, dtrain, num_round, evallist)
    auc.append(float(model.eval(dval).split(':')[1]))
print("{}-fold cross validation AUC: {:.4f}".format(n_splits, np.mean(auc)))


# In[13]:


from sklearn.model_selection import GridSearchCV
params = {
         'gamma': [0, 0.5, 1],
         'learning_rate': [0.1, 0.2, 0.5],
         'colsample_bytree': [0.6, 0.8, 1.0],
         'max_depth': [3, 4, 5],
         'n_estimators': [100, 200]
        }

model = XGBClassifier(n_jobs = 8)
gscv = GridSearchCV(model, params, cv=5, scoring='roc_auc')
gscv.fit(train_data[features], train_data[label])
gs_best = gscv.best_estimator_


# In[14]:


bst = gscv.best_estimator_
bst.fit(train_data[features], train_data[label])
sample[label] = [x[1] for x in bst.predict_proba(test_data[features])]
sample.to_csv("../data/submission.csv", index=False)


# In[15]:


sorted(list(model.get_fscore().items()), key=lambda x: x[1], reverse=True)


# In[16]:


features = list(set([col for col in train_data.columns if not col.endswith("_y")]) - set(["MISPAR_ISHI", "TZ", "NESHER"]) - set(bad_features))
cat_features = ["MAHZOR_ACHARON", "DESTINATION_x"]
cat_features = [i for (i,x) in  enumerate(features) if x in cat_features]


# In[17]:


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
auc = []

params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth':10,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': seed,
    'use_best_model': False,
    'verbose':1
}



for i,(train,val) in enumerate(kf.split(train_data)):
    print("Fold {} out of {}".format(i+1, n_splits))
    train_pool = cb.Pool(train_data.loc[train, features], train_data.loc[train, label],
                      cat_features=cat_features)
    validate_pool = cb.Pool(train_data.loc[val, features], train_data.loc[val, label],
                         cat_features=cat_features)

    model = cb.CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=validate_pool)
    
    auc.append(roc_auc_score(validate_pool.get_label(), [x[1] for x in model.predict_proba(validate_pool)]))
print("{}-fold cross validation AUC: {:.2f}".format(n_splits, np.mean(auc)))


# In[18]:


n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
p_scores = []
params = {
    'iterations': [400, 500, 700],
    'learning_rate': [0.03 ,0.05, 0.07],
    'depth':[4,5,6],
    'rsm': [0.7, 0.8, 0.9],
    'loss_function': ['Logloss'],
    'eval_metric': ['AUC'],
    'random_seed': [seed],
    'use_best_model': [False]
}


for i,p in enumerate(ParameterGrid(params)):
    print("Model number {}".format(i))
    auc = []
    for i,(train,val) in enumerate(kf.split(train_data)):
        print("Fold {} out of {}".format(i+1, n_splits))
        train_pool = cb.Pool(train_data.loc[train, features], train_data.loc[train, label],
                          cat_features=cat_features)
        validate_pool = cb.Pool(train_data.loc[val, features], train_data.loc[val, label],
                             cat_features=cat_features)

        model = cb.CatBoostClassifier(**p)
        model.fit(train_pool, eval_set=validate_pool)

        auc.append((roc_auc_score(validate_pool.get_label(), [x[1] for x in model.predict_proba(validate_pool)])))
    p_scores.append([np.mean(auc), p])


# In[19]:


sorted(auc, key=lambda x: x[0], reverse=True)[0]


# In[20]:


params = {'depth': 5,
 'eval_metric': 'AUC',
 'iterations': 500,
 'learning_rate': 0.05,
 'loss_function': 'Logloss',
 'random_seed': 123,
 'rsm': 0.8,
 'use_best_model': False}


# In[21]:


train_pool = cb.Pool(train_data[features], train_data[label],
                      cat_features=cat_features)
test_pool = cb.Pool(test_data[features],
                      cat_features=cat_features)
model = cb.CatBoostClassifier(**params)
model = model.fit(train_pool)
                     
sample[label] = [x[1] for x in model.predict_proba(test_pool)]
sample.to_csv("../data/submission1000.csv", index=False)


# In[22]:


sorted(zip(features,model.feature_importances_), key=lambda x: x[1], reverse=True)

