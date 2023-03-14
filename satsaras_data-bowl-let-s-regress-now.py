#!/usr/bin/env python
# coding: utf-8

# In[1]:




import os
import pandas as pd
import seaborn as sns
import numpy as np
import datetime
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from pprint import pprint
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import StratifiedKFold, GroupKFold,GroupShuffleSplit,StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale 
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hashlib
from ml_metrics import quadratic_weighted_kappa
from catboost import CatBoostClassifier,Pool,CatBoostRegressor
import shap
import statistics
from functools import partial
import scipy as sp
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from mlxtend.regressor import StackingCVRegressor
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


# In[2]:


df_train=pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')


# In[3]:


spec=pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')


# In[4]:


df_test=pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')


# In[5]:


#spec=None
spec['info']=spec['info'].str.upper()
spec['hashed_info']=spec['info'].transform(hash)
spec_unique=pd.DataFrame(spec[['hashed_info']].drop_duplicates())
spec_unique['deduped_event_id']=np.arange(len(spec_unique))
spec=pd.merge(spec,spec_unique,on='hashed_info',how='left')
z=dict(zip(spec.event_id,spec.deduped_event_id))
df_train['event_id']=df_train['event_id'].map(z)
df_test['event_id']=df_test['event_id'].map(z)
    #df_train=df_train[df_train['event_id'].isin(df_test['event_id'])]
df_train=df_train[df_train['event_id']!=137]  # this particular event id only has 2 records in train and none in test....
df_event_id_train=pd.pivot_table(df_train.loc[:,['installation_id','game_session','event_id']],aggfunc=len,columns=['event_id'],index=['installation_id','game_session']).add_prefix('event_id_').rename_axis(None,axis=1).reset_index()
df_event_id_test=pd.pivot_table(df_test.loc[:,['installation_id','game_session','event_id']],aggfunc=len,columns=['event_id'],index=['installation_id','game_session']).add_prefix('event_id_').rename_axis(None,axis=1).reset_index()
df_event_id_train=df_event_id_train.fillna(0)
df_event_id_train=df_event_id_train.fillna(0)
df_event_id_test=df_event_id_test.fillna(0)


# In[6]:


def create_features(df):
    df['timestamp']=pd.to_datetime(df['timestamp'])
    df['Incorrect_Game_Attempt']=np.where((df['event_data'].str.contains('"correct":false')&(df['type']=='Game')),1,0)
    df['Correct_Game_Attempt']=np.where((df['event_data'].str.contains('"correct":true')&(df['type']=='Game')),1,0)
    df['Is_Weekend']=np.where(((df['timestamp'].dt.day_name()=='Sunday')|(df['timestamp'].dt.day_name()=='Saturday')),1,0)
    df['Phase_Of_Day']=np.where(df['timestamp'].dt.hour.isin(range(6,12)),'Morning',np.where(df['timestamp'].dt.hour.isin(range(13,19)),'Evening','Night'))
    #train_Assessed_ids=set(df.loc[df['type']=='Assessment']
    #.loc[((df['title'] != "Bird Measurer (Assessment)")&(df['event_code']==4100))|((df['title'] == "Bird Measurer (Assessment)")&(df['event_code']==4110))]
    #.loc[:,'installation_id'].unique())
    df_world=pd.pivot_table(df.loc[df['world']!='NONE',['installation_id','game_session','world']].drop_duplicates(),index=['installation_id','game_session'],columns=['world'],aggfunc=len).add_prefix('rolling_').rename_axis(None, axis=1).reset_index()
    
    df_type_world=pd.merge(df_world,pd.pivot_table(df.loc[:,['installation_id','game_session','type']].drop_duplicates(),index=['installation_id','game_session'],columns=['type'],fill_value=0,aggfunc=len).rename_axis(None, axis=1).reset_index(),on=['installation_id','game_session'],how='right')
    df_type_world_title=pd.merge(df_type_world,pd.pivot_table(df.loc[:,['installation_id','game_session','title']].drop_duplicates(),index=['installation_id','game_session'],columns=['title'],fill_value=0,aggfunc=len).add_prefix('rolling_').rename_axis(None, axis=1).reset_index(),on=['installation_id','game_session'],how='right')

    df_activity_weekend=pd.merge(df_type_world_title,pd.DataFrame(pd.pivot_table(df.loc[:,['installation_id','game_session','Is_Weekend']].drop_duplicates(),index=['installation_id','game_session'],columns=['Is_Weekend'],fill_value=0,aggfunc=len)).add_prefix('Weekend_').rename_axis(None, axis=1).reset_index(),on=['installation_id','game_session'],how='right')
    df_activity_weekend_phase_of_day=pd.merge(pd.DataFrame(pd.pivot_table(df.loc[:,['installation_id','game_session','Phase_Of_Day']].drop_duplicates(),index=['installation_id','game_session'],columns=['Phase_Of_Day'],fill_value=0,aggfunc=len)).rename_axis(None, axis=1).reset_index(),df_activity_weekend,on=['installation_id','game_session'],how='left')
    df_train_Assessed=df.copy()
    df_train_Assessed['Incorrect_Attempt']=np.where((df['event_data'].str.contains('"correct":false'))&(((df['title'] != "Bird Measurer (Assessment)")&(df['event_code']==4100))|((df['title'] == "Bird Measurer (Assessment)")&(df['event_code']==4110))),1,0)
    df_train_Assessed['Correct_Attempt']=np.where((df['event_data'].str.contains('"correct":true'))&(((df['title'] != "Bird Measurer (Assessment)")&(df['event_code']==4100))|((df['title'] == "Bird Measurer (Assessment)")&(df['event_code']==4110))),1,0)
    df_train_acc=df_train_Assessed[df_train_Assessed['title'].isin(['Bird Measurer (Assessment)','Mushroom Sorter (Assessment)','Cauldron Filler (Assessment)','Chest Sorter (Assessment)','Cart Balancer (Assessment)'])].groupby(['installation_id','title','game_session'])['Incorrect_Attempt','Correct_Attempt'].sum().rename_axis(None, axis=1).reset_index()
    df_train_acc['Total_Attempts']=df_train_acc.apply(lambda x: x['Incorrect_Attempt'] + x['Correct_Attempt'], axis=1)
    #df_train_acc=df_train_acc[df_train_acc['Total_Attempts']!=0]
    #
    #df_train_acc['accuracy']=[x['Correct_Attempt']/ x['Total_Attempts'] for x in df_train_acc if x['Total_Attempts']>0]
    df_train_acc['accuracy']=np.where(df_train_acc['Total_Attempts']>0,df_train_acc['Correct_Attempt']/ df_train_acc['Total_Attempts'],0)
    df_train_acc['accuracy_group']=np.where(df_train_acc['accuracy']==1,3,np.where(df_train_acc['accuracy']==.5,2,np.where(df_train_acc['accuracy']==0,0,1)))
    df_game_attempt=df.groupby(['installation_id','game_session'])['Incorrect_Game_Attempt','Correct_Game_Attempt'].sum().rename_axis(None, axis=1).reset_index()

    df_event_codes=pd.pivot_table(df_train_Assessed.loc[:,['installation_id','game_session','event_code']],index=['installation_id','game_session'],columns=['event_code'],fill_value=0,aggfunc=len).add_prefix('event_code_').rename_axis(None, axis=1).reset_index()
    df_final=pd.merge(pd.merge(df_train_acc,df_activity_weekend_phase_of_day,on=['installation_id','game_session'],how='right'),df_event_codes,on=['installation_id','game_session'],how='right')
    df_gametime=df.groupby(['installation_id','game_session'])['game_time','timestamp','event_count'].max().reset_index()
    df_final=pd.merge(df_final,df_gametime,on=['installation_id','game_session'],how='left')
    df_final=df_final.fillna(value=0)
    #df_final_title=pd.pivot_table(df_final.loc[df_final['title']!=0,['installation_id','game_session','title','Incorrect_Attempt','Correct_Attempt']].drop_duplicates(),index=['installation_id','game_session'],columns=['title'],values=['Incorrect_Attempt','Correct_Attempt'],aggfunc=sum)
    #df_final_title=pd.pivot_table(df_final.loc[df_final['title']!=0,['installation_id','game_session','title','Incorrect_Attempt','Correct_Attempt']].drop_duplicates(),index=['installation_id','game_session'],columns=['title'],values=['Incorrect_Attempt','Correct_Attempt'],aggfunc=sum)
    #df_final_title.columns=['_'.join(col) for col in df_final_title.columns.values]
    #df_final_title=df_final_title.reset_index()
    #df_final=pd.merge(df_final.drop(['Incorrect_Attempt','Correct_Attempt','accuracy'],axis=1),df_final_title,on=['installation_id','game_session'],how='left')
    df_final=pd.merge(df_final,df.loc[df['world']!="NONE",['installation_id','game_session','world']].drop_duplicates(),on=['installation_id','game_session'],how='left')
    df_final=pd.merge(df_final,df_game_attempt,on=['installation_id','game_session'],how='left')
    df_final=df_final.fillna(value=0)
    return(df_final)


# In[7]:


def rolling_exponential_average(df):
    col=list(df.select_dtypes(include=['float64','int64']).columns)
    #print(df.head())
    df=df.sort_values(['installation_id','timestamp'])
    df_rolling_avg=df.groupby(['installation_id'])[col].apply(lambda x:x.ewm(alpha=0.1,min_periods=1).mean())
    df_rolling_avg=df_rolling_avg.rename(columns={'accuracy_group':'rolling_accuracy_group','CRYSTALCAVES':'rolling_CRYSTALCAVES','MAGMAPEAK':'rolling_MAGMAPEAK', 'TREETOPCITY':'rolling_TREETOPCITY'})
    #print(df_rolling_avg.index)
    #df_rolling_avg.index.names=['installation_id', 'level_1']
    df_rolling_avg.index.names=['level_1']
    return(df_rolling_avg)


# In[8]:


def create_final_dataset(df,Is_Test=0):
    f=create_features(df)
    df_assessed=set(df.loc[df['type']=='Assessment']
                    .loc[((df['title'] != "Bird Measurer (Assessment)")&(df['event_code']==4100))|
                         ((df['title'] == "Bird Measurer (Assessment)")&(df['event_code']==4110))]
                    .loc[:,'game_session'].unique())
    #if (Is_Test==0):
    #    f=pd.merge(f,df_event_id_train,right_index=True,left_index=True)
    #else:
    #    f=pd.merge(f,df_event_id_test,right_index=True,left_index=True)
    if (Is_Test==0):
        f=pd.merge(f,df_event_id_train,on=['installation_id','game_session'],how='left')
    else:
        f=pd.merge(f,df_event_id_test,on=['installation_id','game_session'],how='left')
    
    df_rolling_avg=rolling_exponential_average(f)
    #print(df_rolling_avg.columns)
    df_rolling_avg=df_rolling_avg.fillna(0)
    f.index.names=['level_1']
    
    t=pd.merge(f[['installation_id','game_session','timestamp','world','title']],df_rolling_avg,left_index=True,right_index=True,how='left')
    #print(t.loc[t['game_session'].isin(df_assessed),'title'].unique())
    t=t.reset_index(drop=True).sort_values(['installation_id','timestamp'])
    tm=pd.merge(t[['installation_id','game_session','title','world']],t.groupby(['installation_id']).shift(1).drop(['game_session','title','world'],axis=1),left_index=True,right_index=True,how='left')
    tm=tm.dropna()
    tm=tm[tm['game_session'].isin(df_assessed)]
    #print(tm.loc[tm['game_session'].isin(df_assessed),'title'].unique())
    #print(tm.columns)
    if (Is_Test==0):
        tm['last_attempt_time']=tm.groupby(['installation_id']).shift(1)['timestamp']
        tm['last_world']=tm.groupby(['installation_id']).shift(1)['world']
        tm['last_title']=tm.groupby(['installation_id']).shift(1)['title']
        tm['time_since_last_attempt']=np.where(tm['last_attempt_time'].isna(),0,(tm['timestamp']-tm['last_attempt_time']).dt.seconds)
    else:
        df_latest_Attempt=t.sort_values(['installation_id','timestamp']).groupby(['installation_id'],as_index=False).last()
        #df_latest_Attempt=t.groupby(['installation_id','timestamp'],as_index=False).max()
        #print(df_latest_Attempt.shape)
        #print(tm[['timestamp']].head())
        tm=tm.sort_values(['installation_id','timestamp','title','world']).groupby(['installation_id'],as_index=False).last().rename(columns={'timestamp':'last_attempt_time','world':'last_world','title':'last_title'}).drop_duplicates()
        
        df_latest_Attempt=pd.merge(tm[['installation_id','last_attempt_time','last_world','last_title']],df_latest_Attempt,on=['installation_id'],how='right')
        #print(df_latest_Attempt.head())
        df_latest_Attempt['time_since_last_attempt']=np.where(df_latest_Attempt['last_attempt_time'].isna(),0,(pd.to_datetime(df_latest_Attempt['timestamp'].dt.tz_localize(None))-pd.to_datetime(df_latest_Attempt['last_attempt_time'])).dt.seconds)
        tm=df_latest_Attempt.copy()
        #print(tm.shape)
    #tm1=pd.merge(tm,tm.groupby(['installation_id'])['timestamp'].shift(1),left_index=True,right_index=True,how='left')
    tm1=pd.merge(tm,f[['accuracy_group','game_session','installation_id']],on=['game_session','installation_id'],how='left')
    tm1=tm1.fillna(0)
    return(tm1)


# In[9]:


f_train=create_final_dataset(df_train,0)


f_test=create_final_dataset(df_test,1)

f_train = f_train.reindex(sorted(f_train.columns), axis=1)
f_test = f_test.reindex(sorted(f_test.columns), axis=1)


# In[10]:


f_train_1=shuffle(f_train)


# In[11]:


f_train_1.reset_index(inplace=True, drop=True) 


# In[12]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])
        return -cohen_kappa_score(y, preds, weights = 'quadratic')
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')
    
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])
        return preds
    
    def coefficients(self):
        return self.coef_['x']


# In[13]:


def model_classification(train_X,train_y,val_X,val_y,cat_feat):
    
    CatBoost=CatBoostClassifier(
        learning_rate=0.2,
        max_depth=6,
        bootstrap_type='Bernoulli',
        l2_leaf_reg= 0.2,
        iterations=400,
        subsample=.7,
        loss_function='MultiClass',
        eval_metric="WKappa",
        random_seed=4,cat_features=cat_feat,class_weights=[1,2,2,.5],
        classes_count=4)
    clf = CatBoost.fit(train_X,train_y,eval_set=(val_X,val_y), verbose_eval=False,use_best_model=True,plot=True,early_stopping_rounds=100)
    return(clf)

    


# In[14]:


def model_regressor(train_X,train_y,val_X,val_y,fit):
    lgb_model=LGBMRegressor(objective='regression',
                         max_depth=13,num_leaves=6,
                          n_estimators=1200,
                         bagging_fraction=0.8,
                         random_state=42,verbose=0,reg_alpha=.3,reg_lambda=.3,callbacks=[lgb.reset_parameter(learning_rate=[0.2] * 200 + [0.15] * 250+[.1]*250+[.05]*250+[.01]*250)])
    if fit==0:
        reg=lgb_model.fit(train_X,train_y,eval_set=(val_X,val_y), verbose=False)
    else:
        reg=lgb_model.fit(train_X,train_y,verbose=False)
    
    

        
    return(reg)


# In[15]:


def fit(X_train,y_train,classifier):
    train_X = X_train
    train_y=y_train
    
    train_X=train_X.drop('installation_id',axis=1)
    
    
    if(classifier==1):
            clf=model_classification(train_X,train_y,0,0,cat_feat)
            pred_val=clf.predict(val_X)
            print(classification_report(val_y,pred_val))
        #print(evals_result)
            score=cohen_kappa_score(pred_val,val_y,weights='quadratic')
            scores.append(score)
            print('choen_kappa_score :',score)
            coeff=0
    else:
            clf=model_regressor(train_X,train_y,0,0,1)
            optR = OptimizedRounder()
            optR.fit(clf.predict(train_X), train_y)
            coeff = optR.coefficients()         
            
            
    return(clf,coeff)  


# In[16]:



def cv(X_train,y_train,classifier=1,n_splits=5):
    scores=[]

    #kf = StratifiedShuffleSplit(n_splits=n_splits, random_state=42,)
    Gf=GroupKFold(n_splits=n_splits)
    #y_pre=np.zeros((len(validation_set),4),dtype=float)
    #final_test=xgb.DMatrix(validation_set)
    
    i=0
    
    
    for train_index, val_index in Gf.split(X_train, y_train,X_train['installation_id']): 
        #print(len(train_index),len(val_index))
        train_X = X_train.iloc[train_index]
        train_X=train_X.drop('installation_id',axis=1)
        val=X_train.iloc[val_index].join(y_train[val_index])
        #val_X = X_train.iloc[val_index]
        #val_X=val_X.drop('installation_id',axis=1)
        train_y = y_train[train_index]
        
        #print(train_y.columns)
        #val_y = y_train[val_index]
        val=val.groupby('installation_id',as_index=False).apply(lambda x :x.iloc[np.random.randint(0,len(x))])
        val_X=val.drop(['accuracy_group','installation_id'],axis=1)
        val_y=val['accuracy_group']
        #class_weights=class_weight.compute_class_weight('balanced',np.unique(train_y),train_y)
        #print(class_weights)
        #print(np.unique(w_array))   
        #xgb_train = xgb.DMatrix(train_X, train_y,weight=None)
        #xgb_eval = xgb.DMatrix(val_X, val_y)
        if(classifier==1):
            clf=model_classification(train_X,train_y,val_X,val_y,0)
            pred_val=clf.predict(val_X)
            print(classification_report(val_y,pred_val))
        #print(evals_result)
            score=cohen_kappa_score(pred_val,val_y,weights='quadratic')
            scores.append(score)
            print('choen_kappa_score :',score)
        else:
            clf=model_regressor(train_X,train_y,val_X,val_y,0)
            print(clf.predict(train_X))
            optR = OptimizedRounder()
            optR.fit(clf.predict(train_X), train_y)
            coeff = optR.coefficients()
            print(coeff)
            
            val_pred=clf.predict(val_X)
            pred_val=[regression_class(val_pred_i,coeff) for val_pred_i in val_pred]

            print(classification_report(val_y,pred_val))
            score=cohen_kappa_score(pred_val,val_y,weights='quadratic')
            scores.append(score)
            print('choen_kappa_score :',score)           
    print(statistics.mean(scores),statistics.stdev(scores))      
    return(clf)
            

        

   


# In[17]:


def regression_class(pred,coeff):
    if (pred<=coeff[0]):
        class_=0
    elif (coeff[0]<pred<=coeff[1]):
        class_=1
    elif (coeff[1]<pred<=coeff[2]):
        class_=2
    else:
        class_=3
    return(class_)


# In[18]:


f_train_1['accuracy_group']=f_train_1['accuracy_group'].astype('int')


# In[19]:


X_train=f_train_1.drop(['accuracy_group','game_session','last_attempt_time','timestamp'],axis=1)
y_train=f_train_1['accuracy_group']
X_test=f_test.drop(['accuracy_group','game_session','last_attempt_time','timestamp'],axis=1)
y_test=f_test['accuracy_group']


# In[20]:


c=['last_world','last_title','world','title']
cat_feat=[X_train.columns.get_loc(col) for col in c]
X_train=X_train.join(pd.get_dummies(X_train[c])).drop(c,axis=1)


# In[21]:


X_test=X_test.join(pd.get_dummies(X_test[c])).drop(c,axis=1)


# In[22]:


reg_cv=cv(X_train,y_train,0,5)


# In[23]:


reg_model,coeff=fit(X_train,y_train,0)


# In[24]:


f_test['accuracy_group']=[regression_class(pred_i,coeff) for pred_i in reg_model.predict(X_test.drop('installation_id',axis=1))] 


# In[25]:


Submission_file=f_test[['installation_id','accuracy_group']]


# In[26]:


Submission_file['accuracy_group'].value_counts()


# In[27]:


Submission_file.to_csv('submission.csv',index=False)


# In[ ]:




