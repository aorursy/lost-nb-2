#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

from pydicom import dcmread
import cv2

path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"


# In[2]:


ls /kaggle/input/osic-pulmonary-fibrosis-progression/


# In[3]:


ls /kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430


# In[4]:


train_df  = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
subm = pd.read_csv(path + "sample_submission.csv")


# In[5]:


train_df


# In[6]:


train_df.Patient.nunique()


# In[7]:


train_df.Weeks.max()


# In[8]:


train_df.Weeks.min()


# In[9]:


fig, ax = plt.subplots(1,1)

sns.distplot(train_df[train_df["Weeks"].notna()]["Weeks"], ax=ax, color="#2222EE")
ax.set_title("distribution of weeks in train");


# In[10]:


fig, ax = plt.subplots(1,1)

sns.distplot(train_df[train_df["FVC"].notna()]["FVC"], ax=ax, color="#22EE22")
ax.set_title("distribution of FVC in train");


# In[11]:


fig, ax = plt.subplots(1,1)

sns.distplot(train_df[train_df["Percent"].notna()]["Percent"], ax=ax, color="#EE2222")
ax.set_title("distribution of Percent in train");


# In[12]:


fig, ax = plt.subplots(1,1)

sns.distplot(train_df[train_df["Age"].notna()]["Age"], ax=ax, color="#992299")
ax.set_title("distribution of Age in train");


# In[13]:


train_df.Sex.value_counts()


# In[14]:


train_df.Sex.value_counts(normalize=True)


# In[15]:


train_df.groupby("Patient")["Sex"].first().value_counts(normalize=True)


# In[16]:


train_df["SmokingStatus"].value_counts()


# In[17]:


train_df["SmokingStatus"].value_counts(normalize=True)


# In[18]:


test_df


# In[19]:


def merge_subm_test(subm,test_df):
    
    a = subm['Patient_Week'].str.split("_", expand=True)
    a.columns=["Patient","Week"]
        
    test_df = test_df.merge(a, on="Patient")

    return test_df

test_df = merge_subm_test(subm,test_df)


# In[20]:


test_df


# In[21]:


test_df.groupby(["Patient"])["Weeks"].count()


# In[22]:


test_df.groupby(["Patient"])["Week"].first()


# In[23]:


test_df.groupby(["Patient"])["Week"].last()


# In[24]:


ls /kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/


# In[25]:


fig, axs = plt.subplots(5, 6,figsize=(20,20))
for n in range(0,30):
    #print(int(n/6),np.mod(n,6))
    image = dcmread("/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/" + str(n+1) + ".dcm")
    axs[int(n/6),np.mod(n,6)].imshow(image.pixel_array);


# In[26]:


ls /kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/


# In[27]:


fig, axs = plt.subplots(5, 6,figsize=(20,20))
for n in range(0,28):
    #print(int(n/6),np.mod(n,6))
    image = dcmread("/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/" + str(n+1) + ".dcm")
    axs[int(n/6),np.mod(n,6)].imshow(image.pixel_array);


# In[28]:


train_df  = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
subm = pd.read_csv(path + "sample_submission.csv")


# In[29]:


def proc_df(df):
    
    df = pd.concat([df, pd.get_dummies(df["SmokingStatus"], dtype=int)], axis=1)
    df.drop(["SmokingStatus"],axis=1,inplace=True)

    df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})

    return df

train_df = proc_df(train_df)


# In[30]:


def proc_train(df):

    df_final = pd.DataFrame()

    for patient, df2 in df.groupby('Patient'):
        
        df11 = df2[["Patient","Weeks","FVC"]]


        df2 = df2.rename(columns={
            "FVC": "base_FVC", 
            "Percent": "base_Percent", 
            "Weeks": "base_Week"
        }, errors="raise")
        
        df3 = pd.merge(df11, df2, how='outer', on='Patient')
        df3 = df3.query('Weeks!=base_Week')
        df3['week_diff'] = df3['base_Week'] - df3['Weeks']

        df_final = pd.concat([df_final, df3])
        
    return df_final.reset_index(drop=True)


# In[31]:


train_df = proc_train(train_df)


# In[32]:


a = subm['Patient_Week'].str.split("_", expand=True)
a.columns=["Patient","Weeks"]
a["Weeks"] = a["Weeks"].astype("int")
    
test_df.rename(
    columns={
        'Weeks': 'base_Week',
        'FVC': 'base_FVC',
        'Percent': 'base_Percent',
        'Age': 'Age'
    },
    inplace=True
)

test_df = proc_df(test_df)

test_df = pd.merge(a, test_df, how='left', on=['Patient'])
test_df['week_diff'] = test_df['base_Week'] - test_df['Weeks']

test_df


# In[33]:


train_df = train_df.drop(set(train_df.columns)-set(test_df.columns)-{'FVC', 'Percent'}, axis=1)
test_df = test_df.drop(set(test_df.columns)-set(train_df.columns), axis=1)

X = train_df.drop(["Patient","FVC"], axis=1)
y = train_df["FVC"]
test = test_df.drop(["Patient"], axis=1)


# In[34]:


X


# In[35]:


test


# In[36]:


num_fold = 5

def get_lgbm_model(X_train, y_train, X_val, y_val, fold, param_choice, columns):
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    if param_choice == "normal":
        params = {
            "metric":"rmse"
        }
    elif param_choice == "quantile1":
        params = {
            "objective":"quantile",
            "alpha":0.2,
            "metric":"quantile"
        }    
    elif param_choice == "quantile2":
        params = {
            "objective":"quantile",
            "alpha":0.8,
            "metric":"quantile"
        }    
    
    model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)], 
        verbose=1000, 
        early_stopping_rounds=100
    )

    fold_importance = pd.DataFrame()
    print(columns)
    print(model.feature_importances_)
    fold_importance["feature"] = columns
    fold_importance["importance"] = model.feature_importances_
    #plot_save_feat_imp(fold_importance, fold)    
    fold_importance = fold_importance.sort_values(by=['importance'])
    fold_importance.to_csv('feature_importances_'+ y_train.name + "_" + param_choice + str(fold) + '.csv')
    
    return model

def get_lgbm_pred(X, y, test, param_choice):
    print("get_lgbm_pred ", param_choice)

    pred = []
    pred_val = np.zeros((len(X)))
            
    #X_train, X_val, y_train, y_val = train_test_split(X_scaled, y.fillna(0).values, test_size=0.2, shuffle=True, random_state=42)
    kf = KFold(n_splits=num_fold, random_state=None, shuffle=False)
    fold = 0
    score = 0
    for train_index, test_index in kf.split(X, y):
        fold += 1
        print("fold ", fold)
    
        X_train = X.iloc[train_index, :]
        X_val = X.iloc[test_index, :]
        y_train = y[train_index]
        y_val = y[test_index]
        
        model = get_lgbm_model(X_train, y_train, X_val, y_val, fold, param_choice, X_train.columns)


        if fold ==1:
            pred.append(model.predict(test))
        else:
            pred += model.predict(test)

        
        pred_val[test_index] = model.predict(X_val)            
        score = score + np.sqrt(mean_squared_error(y_val, pred_val[test_index]))
        print("score ", str(score/fold))
            
                    
    print("\n\n\n")

    f = open("score","a+")
    f.write(str(score/num_fold)+", ")
    f.close()
    
    return pred[0]/num_fold, pred_val


# In[37]:


pred_FVC_te, pred_FVC_tr = get_lgbm_pred(X, y, test, "normal")

pred_FVC_te_q1, pred_FVC_tr_q1 = get_lgbm_pred(X, y, test, "quantile1")
pred_FVC_te_q2, pred_FVC_tr_q2 = get_lgbm_pred(X, y, test, "quantile2")

pred_conf_tr = pred_FVC_tr_q2 - pred_FVC_tr_q1
pred_conf_te = pred_FVC_te_q2 - pred_FVC_te_q1


# In[38]:


def metric(confidence, fvc, pred_fvc):

    confidence = max(confidence, 70)
    delta = min(abs(fvc-pred_fvc), 1000)
    score = -(math.sqrt(2)*(delta/confidence)) - np.log(math.sqrt(2)*confidence)
        
    return score

def calc_score(confidence, fvc, pred_fvc):
    
    score = 0
    for n in range(len(confidence)):
        score += (metric(confidence[n], fvc[n], pred_fvc[n]))
        
    return score/len(train_df)

score = calc_score(pred_conf_tr, train_df.FVC.values, pred_FVC_tr)
print(score)


# In[39]:


subm["FVC"] = pred_FVC_te
subm["Confidence"] = pred_conf_te
subm.to_csv("submission.csv", index=False)


# In[40]:


subm

