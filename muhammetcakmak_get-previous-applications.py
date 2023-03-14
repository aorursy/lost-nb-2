#!/usr/bin/env python
# coding: utf-8

# In[1]:


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook as tqdm
# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import re


# In[3]:


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# In[4]:


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:100].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(15, 20))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


# In[5]:


def application_train():

    df = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
    test_df = pd.read_csv('../input/home-credit-default-risk/application_test.csv')

    df = df.append(test_df).reset_index()
    df = df[df['CODE_GENDER'] != 'XNA']

    lbe = LabelEncoder()

    for col in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[col] = lbe.fit_transform(df[col])

    df = pd.get_dummies(df, dummy_na = True)

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
    df['NEW_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['NEW_ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['NEW_PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    df.drop("index", axis = 1, inplace =  True)

    df.columns = pd.Index(["APP_" + col for col in df.columns.tolist()])

    df.rename(columns={"APP_SK_ID_CURR":"SK_ID_CURR"}, inplace = True)

    df.rename(columns={"APP_TARGET":"TARGET"}, inplace = True)
    
    return df


# In[6]:


def previous_application():


    df_prev = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')
    
    # Features that has outliers
    feat_outlier = ["AMT_ANNUITY","AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE", "SELLERPLACE_AREA"]
    
    # Replacing the outliers of the features with their own upper values
    for var in feat_outlier:
        
        Q1 = df_prev[var].quantile(0.01)
        Q3 = df_prev[var].quantile(0.99)
        IQR = Q3-Q1
        lower = Q1- 1.5*IQR
        upper = Q3 + 1.5*IQR
        
        df_prev[var][(df_prev[var] > (upper))] = upper
    
    # 365243 value will be replaced by NaN in the following features
    feature_replace = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for var in feature_replace:
        df_prev[var].replace(365243, np.nan, inplace= True)
        
        
    # One hot encoding
    categorical_columns = [col for col in df_prev.columns if df_prev[col].dtype == 'object']
    df_prev = pd.get_dummies(df_prev, columns = categorical_columns, dummy_na = True)
    
    
    # Creating new features
    
    df_prev['APP_CREDIT_PERC'] = df_prev['AMT_APPLICATION'] / df_prev['AMT_CREDIT']
    df_prev['NEW_CREDIT_TO_ANNUITY_RATIO'] = df_prev['AMT_CREDIT']/df_prev['AMT_ANNUITY']
    df_prev['NEW_DOWN_PAYMENT_TO_CREDIT'] = df_prev['AMT_DOWN_PAYMENT'] / df_prev['AMT_CREDIT']
    df_prev['NEW_TOTAL_PAYMENT'] = df_prev['AMT_ANNUITY'] * df_prev['CNT_PAYMENT']
    df_prev['NEW_TOTAL_PAYMENT_TO_AMT_CREDIT'] = df_prev['NEW_TOTAL_PAYMENT'] / df_prev['AMT_CREDIT']
    # Innterest ratio previous application (simplified)

    df_prev['SIMPLE_INTERESTS'] = (df_prev['NEW_TOTAL_PAYMENT']/df_prev['AMT_CREDIT'] - 1)/df_prev['CNT_PAYMENT']
    
    # Previous applications numeric features
    num_aggregations = {}
    num_cols = df_prev.select_dtypes(exclude=['object']) 
    num_cols.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis= 1,inplace = True)
    for num in num_cols:
        num_aggregations[num] = ['min', 'max', 'mean', 'var','sum']

    
        # Previous applications categoric features
    cat_aggregations = {}
    for i in df_prev.columns: 
        if df_prev[i].dtypes == "O":
            cat_aggregations[i] = ['mean']

    
    
    prev_agg = df_prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    
    
    # Dropping features with small variance
    features_with_small_variance = prev_agg.columns[(prev_agg.std(axis = 0) < .1).values]
    prev_agg.drop(features_with_small_variance, axis = 1, inplace = True)
    
    
    

    
    return prev_agg


# In[7]:


def pre_processing_and_combine():
   with timer("Process application train"):
       df = application_train()
       print("application train & test shape:", df.shape)
   
   with timer("previous_application"):
           df_prev = previous_application()
           print("previous_application:", df_prev.shape) 
           
   # merging prev and train application table
   df_prev_and_train = df.merge(df_prev, how = 'left',on = 'SK_ID_CURR')
   
   print("the shape of prev and train data:", df_prev_and_train.shape) 

   
   
   return df_prev_and_train


# In[8]:


def modeling(df_prev_and_train):
    all_data = df_prev_and_train.copy()
    all_data = all_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    train_df = all_data[all_data['TARGET'].notnull()]
    test_df = all_data[all_data['TARGET'].isnull()]

    folds = KFold(n_splits = 10, shuffle = True, random_state = 1001)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf = LGBMClassifier(
                n_jobs = -1,
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=34,
                colsample_bytree=0.9497036,
                subsample=0.8715623,
                max_depth=8,
                reg_alpha=0.041545473,
                reg_lambda=0.0735294,
                min_split_gain=0.0222415,
                min_child_weight=39.3259775,
                silent=-1,
                verbose=-1)

        clf.fit(train_x, train_y, eval_set = [(train_x, train_y), (valid_x, valid_y)], 
                eval_metric = 'auc', verbose = 200, early_stopping_rounds = 200)

        #y_pred_valid
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))) 


    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds)) #y_pred_valid   

    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv("submission_lightgbm.csv", index= False)

    display_importances(feature_importance_df)
    
    return feature_importance_df


# In[9]:


def main():
    
    with timer("Preprocessing Time"):
        all_data = pre_processing_and_combine()
    
    with timer("Modeling"):
        feat_importance = modeling(all_data)
    return feat_importance


# In[10]:


if __name__ == "__main__":
    with timer("Full model run"):
        main()


# In[ ]:





# In[ ]:




