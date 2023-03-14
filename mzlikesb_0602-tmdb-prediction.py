#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import json
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,GroupKFold
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

import os
#print(os.listdir("../input"))


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.index = train['id']
test.index = test['id']
print(train.shape, test.shape)
train.head()


# In[3]:


print(train.shape)
print(test.shape)
print(train.info())
print(train.isna().sum())


# In[4]:


# 수입 살펴보기
# log : x=0 -> y=-Inf -> 경고발생 vs log1p : x=x+1 를 해줌. x=0+1 -> y=0
_ = sns.distplot(np.log1p(train['revenue']), bins=100, kde=True)


# In[5]:


# 예산 살펴보기
print('train null or 0 :',train[train.budget == 0 | train.budget.isnull()].shape[0])
print('test null or 0:', test[test.budget == 0| test.budget.isnull()].shape[0])
print('중앙값 :', train['budget'].mean())
_ = sns.distplot(np.log1p(train['budget']), bins=100, kde=True)


# In[6]:


# runtime
print('null or 0 :', train['runtime'][(train['runtime'].isnull()) | (train['runtime']==0)].count())
print('null or 0 :', test['runtime'][(test['runtime'].isnull()) | (test['runtime']==0)].count())
print('중앙값 : ',train['runtime'].mean())
_ = sns.distplot(np.log1p(train['runtime']), bins=100, kde=True)


# In[7]:


f, axes = plt.subplots(4, 1, figsize=(10, 10))
_ = sns.regplot(x='budget', y='revenue', data=train,ax=axes[0])
_ = sns.regplot(x='popularity', y='revenue', data=train, ax=axes[1])
_ = sns.regplot(x='runtime', y='revenue', data=train, ax=axes[2])


# In[8]:


# hp, tagline, collection 유무(null 갯수가 많은 컬럼들) vs revenue
f, axes = plt.subplots(3, 1, figsize=(10, 10))
_ = sns.kdeplot(train['revenue'][(train['homepage'].notnull())],label = 'has homepage', ax = axes[0])
_ = sns.kdeplot(train['revenue'][(train['homepage'].isnull())], label = 'has not homepage',ax = axes[0])
_ = sns.kdeplot(train['revenue'][(train['tagline'].notnull())], label = 'has tagline',ax = axes[1])
_ = sns.kdeplot(train['revenue'][(train['tagline'].isnull())], label = 'has not tagline',ax = axes[1])
_ = sns.kdeplot(train['revenue'][(train['belongs_to_collection'].notnull())], label = 'has Collection',ax = axes[2])
_ = sns.kdeplot(train['revenue'][(train['belongs_to_collection'].isnull())], label = 'has not Collection',ax = axes[2])


# In[9]:


# 단어 길이 vs revenue
f, axes = plt.subplots(4, 1, figsize=(10, 10))
_ = sns.kdeplot(np.log1p(train['revenue'][train['tagline'].str.len()]), ax=axes[0]).set_title('tagline length')
_ = sns.kdeplot(np.log1p(train['revenue'][train['title'].str.len()]), ax=axes[1]).set_title('title length')
_ = sns.kdeplot(np.log1p(train['revenue'][train['overview'].str.len()]), ax=axes[2]).set_title('overview length')
_ = sns.kdeplot(np.log1p(train['revenue'][train['original_title'].str.len()]), ax=axes[3]).set_title('original_title length')


# In[10]:


# 개봉 날짜
train['release_date'] = pd.to_datetime(train['release_date'], dayfirst=True)
train['release_year'] = train['release_date'].dt.year
train['release_month'] = train['release_date'].dt.month
train['release_day'] = train['release_date'].dt.day
train['release_dayofweek'] = train['release_date'].dt.dayofweek
train['release_quarter'] = train['release_date'].dt.quarter


# In[11]:


# 개봉 년도 이상값 확인
train['release_year'].groupby(train['release_year'][train['release_year']>2010]).count()


# In[12]:


# 개봉 연도, 월, 날짜, 요일, 분기 vs revenue
f, axes = plt.subplots(5, 1, figsize=(10, 20))
_ =sns.barplot(x='release_year', y='revenue',data=train,ax = axes[0])
_ =sns.barplot(x='release_month', y='revenue',data=train,ax = axes[1])
_ =sns.barplot(x='release_day', y='revenue',data=train,ax = axes[2])
_ =sns.barplot(x='release_dayofweek', y='revenue',data=train,ax = axes[3])
_ =sns.barplot(x='release_quarter', y='revenue',data=train,ax = axes[4])


# In[13]:


# original language
f, axes = plt.subplots(2, 1, figsize=(10, 8))
_= sns.countplot(x='original_language', data=train, ax=axes[0])
_= sns.barplot(x='original_language', y='revenue',data=train, ax=axes[1])


# In[14]:


# json을 dictionary 형태로 변환 
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else eval(x) ) # eval?
    return df


# In[15]:


import copy
temp = copy.deepcopy(train)
temp = text_to_dict(temp)


# In[16]:


# 시리즈별 갯수
# x[0] = train.iloc[0]['belongs_to_collection'][0]
temp['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0).value_counts()[:10]


# In[17]:


# 장르별 갯수
temp['genres'].apply(lambda x: x[0]['name'] if x != {} else 0).value_counts()[:10]


# In[18]:


# Feature engineering
runtime_mean = train['runtime'].mean()
budget_mean =train['budget'].mean()

def prepare(df):    
    # 0 or null 값 채우기
    df['runtime'][(df['runtime'].isnull()) | (df['runtime'] == 0)] = runtime_mean
    df['budget'][df['budget']==0] = budget_mean

    # 바이너리(유무) 컬럼 치환
    df['hasHP']=0
    df.loc[df['homepage'].isnull() == False, 'hasHP'] = 1
    df['hasTL']=0
    df.loc[df['tagline'].isnull() == False, 'hasTL'] = 1
    df['hasCollection']=0
    df.loc[df['belongs_to_collection'].isnull() == False, 'hasCollection'] = 1
    df['isEng']=0
    df.loc[df['original_language'] == 'en', 'isEng'] = 1    
    df['isReleased'] = 1
    df.loc[ df['status'] != "Released" ,"isReleased"] = 0 

    # 글자 길이 치환
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['original_title_word_count'] = df['original_title'].str.split().str.len()

    # 날짜
    df['release_date'] = pd.to_datetime(df['release_date'], dayfirst=True)
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day    
    df['release_year'][df['release_year']>2018] = df['release_year']-100

    df['release_dayofweek'] = df['release_date'].dt.dayofweek
    df['release_quarter'] = df['release_date'].dt.quarter
    
    # 스케일링
    df['budget'] = np.log1p(df['budget'])
    
    # 범주형 데이터
    df = pd.get_dummies(df, columns=['original_language'], prefix='original_language')
    
    # json 컬럼 변환     
    df = text_to_dict(df)
    
    # 시리즈
    df['_collection_name'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
    le = LabelEncoder()
    le.fit(list(df['_collection_name'].fillna('')))
    df['_collection_name'] = le.transform(df['_collection_name'].fillna('').astype(str))      
    
    # 장르 : 2개 이상인 경우도 있지만 일단 심플하게 첫번째 장르만 집계
    df['_genres_name'] = df['genres'].apply(lambda x: x[0]['name'] if x != {} else 0)
    le = LabelEncoder()
    le.fit(list(df['_genres_name'].fillna('')))
    df['_genres_name'] = le.transform(df['_genres_name'].fillna('').astype(str))  
    
    # 키워드 수, 캐스팅 수
    df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)
    
    # 제작국가 수, 제작사 수
    df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))
    df['cast_count'] = df['cast'].apply(lambda x : len(x))
    df['crew_count'] = df['crew'].apply(lambda x : len(x))
    
    df = df.drop(['imdb_id','poster_path','status','homepage','tagline','belongs_to_collection',                          'genres', 'production_companies','production_countries', 'spoken_languages',                          'Keywords', 'cast', 'crew','overview','title','original_title','release_date'],axis=1)
    
    return df


# In[19]:


all_data = pd.concat([train, test]).reset_index(drop = True)
all_data = prepare(all_data)
train = all_data.loc[:train.shape[0] - 1,:]
test = all_data.loc[train.shape[0]:,:] 

print(train.shape)
train.head()


# In[20]:


features = list(train.columns)
features =  [i for i in features if i != 'id' and i != 'revenue']


# In[21]:


# for mse validation 
def score(data, y):
    validation_res = pd.DataFrame({"id": data["id"].values,
                                   "transactionrevenue": data["revenue"].values,
                                   "predictedrevenue": np.expm1(y)})
    validation_res = validation_res.groupby("id")["transactionrevenue", "predictedrevenue"].sum().reset_index()
    return np.sqrt(mean_squared_error(np.log1p(validation_res["transactionrevenue"].values), 
                                     np.log1p(validation_res["predictedrevenue"].values)))


# In[22]:


class KFoldValidation():
    def __init__(self, data, n_splits=5):
        unique_vis = np.array(sorted(data['id'].astype(str).unique()))
        folds = GroupKFold(n_splits)
        ids = np.arange(data.shape[0])
        
        self.fold_ids = []
        for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
            self.fold_ids.append([
                    ids[data['id'].astype(str).isin(unique_vis[trn_vis])],
                    ids[data['id'].astype(str).isin(unique_vis[val_vis])]
                ])
            
    def validate(self, train, test, features, model, name="", prepare_stacking=False, 
                 fit_params={"early_stopping_rounds": 500, "verbose": 100, "eval_metric": "rmse"}):
        model.FI = pd.DataFrame(index=features)
        full_score = 0
        
        if prepare_stacking:
            test[name] = 0
            train[name] = np.NaN
        
        for fold_id, (trn, val) in enumerate(self.fold_ids):
            devel = train[features].iloc[trn]
            y_devel = np.log1p(train["revenue"].iloc[trn])
            valid = train[features].iloc[val]
            y_valid = np.log1p(train["revenue"].iloc[val])
                       
            print("Fold ", fold_id, ":")
            model.fit(devel, y_devel, eval_set=[(valid, y_valid)], **fit_params)
            
            if len(model.feature_importances_) == len(features):  
                model.FI['fold' + str(fold_id)] = model.feature_importances_ / model.feature_importances_.sum()

            predictions = model.predict(valid)
            predictions[predictions < 0] = 0
            print("Fold ", fold_id, " error: ", mean_squared_error(y_valid, predictions)**0.5)
            
            fold_score = score(train.iloc[val], predictions)
            full_score += fold_score / len(self.fold_ids)
            print("Fold ", fold_id, " score: ", fold_score)
            if prepare_stacking:
                train[name].iloc[val] = predictions
                
                test_predictions = model.predict(test[features])
                test_predictions[test_predictions < 0] = 0
                test[name] += test_predictions / len(self.fold_ids)
                
        print("Final score: ", full_score)
        return full_score


# In[23]:


Kfolder = KFoldValidation(train)


# In[24]:


lgbmodel = lgb.LGBMRegressor(n_estimators=10000, 
                             objective='regression', 
                             metric='rmse',
                             max_depth = 5,
                             num_leaves=30, 
                             min_child_samples=100,
                             learning_rate=0.01,
                             boosting = 'gbdt',
                             min_data_in_leaf= 10,
                             feature_fraction = 0.9,
                             bagging_freq = 1,
                             bagging_fraction = 0.9,
                             importance_type='gain',
                             lambda_l1 = 0.2,
                             bagging_seed=42, 
                             subsample=.8, 
                             colsample_bytree=.9,
                             use_best_model=True)
Kfolder.validate(train, test, features , lgbmodel, name="lgbfinal", prepare_stacking=True) 


# In[25]:


xgbmodel = xgb.XGBRegressor(max_depth=5, 
                            learning_rate=0.01, 
                            n_estimators=10000, 
                            objective='reg:linear', 
                            gamma=1.45, 
                            seed=42, 
                            silent=True,
                            subsample=0.8, 
                            colsample_bytree=0.7, 
                            colsample_bylevel=0.5)
Kfolder.validate(train, test, features, xgbmodel, name="xgbfinal", prepare_stacking=True)


# In[26]:


catmodel = cat.CatBoostRegressor(iterations=10000, 
                                 learning_rate=0.01, 
                                 depth=5, 
                                 eval_metric='RMSE',
                                 colsample_bylevel=0.8,
                                 bagging_temperature = 0.2,
                                 metric_period = None,
                                 early_stopping_rounds=200,
                                 random_seed=42)
Kfolder.validate(train, test, features , catmodel, name="catfinal", prepare_stacking=True,
               fit_params={"use_best_model": True, "verbose": 100})


# In[27]:


train['Revenue_lgb'] = train["lgbfinal"]
print("RMSE model lgb :" ,score(train, train.Revenue_lgb),)

train['Revenue_xgb'] = train["xgbfinal"]
print("RMSE model xgb :" ,score(train, train.Revenue_xgb))

train['Revenue_cat'] = train["catfinal"]
print("RMSE model cat :" ,score(train, train.Revenue_cat))

# 모델 합성
train['Revenue_Dragon'] = 0.4 * train["lgbfinal"] + 0.2 * train["xgbfinal"] + 0.4 * train["catfinal"]
print("RMSE model Dragon :" ,score(train, train.Revenue_Dragon))


# In[28]:


test['revenue'] =  np.expm1(test["lgbfinal"])
test[['id','revenue']].to_csv('submission_lgb.csv', index=False)
print(test[['id','revenue']].head())

test['revenue'] =  np.expm1(test["xgbfinal"])
test[['id','revenue']].to_csv('submission_xgb.csv', index=False)
print(test[['id','revenue']].head())

test['revenue'] =  np.expm1(test["catfinal"])
test[['id','revenue']].to_csv('submission_cat.csv', index=False)
print(test[['id','revenue']].head())

test['revenue'] =  np.expm1(0.4 * test["lgbfinal"]+ 0.4 * test["catfinal"] + 0.2 * test["xgbfinal"])
test[['id','revenue']].to_csv('submission_Dragon1.csv', index=False)
print(test[['id','revenue']].head())

