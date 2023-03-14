#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import os
import re
import json
from pandas.io.json import json_normalize
from tqdm import tqdm
from tqdm import tqdm_notebook
from collections import defaultdict
from collections import Counter
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 4
import lightgbm as lgb
import time
import datetime
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import minmax_scale
from sklearn import linear_model
import gc
import seaborn as sns
from numba import jit
import warnings
warnings.filterwarnings("ignore")
from collections import deque


# In[3]:


from sklearn.base import BaseEstimator, TransformerMixin
@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
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


def get_group(accuracy):
    acc_group = np.nan

    if accuracy == 0:
        acc_group = 0
    elif accuracy == 1:
        acc_group = 3
    elif accuracy == 0.5:
        acc_group = 2
    else:
        acc_group = 1

    return acc_group


# In[5]:


def process_raw_data(n1,n2):
    
    print('Start reading train data')
    train101 = pd.read_csv('/kaggle/input/data-science-bowl-2019//train.csv', nrows=n1)
   
    print('Start reading test data')
    test101 = pd.read_csv('/kaggle/input/data-science-bowl-2019//test.csv', nrows=n2)
   
    print('Start reading train lables data')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    
    
    print('Start reading specs data')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    
    print('Start reading sample_submission data')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    
    print( "Raw size...")
    print(train101.shape)
    print(test101.shape)
    
    train_ids_with_subms = train101[train101['type'] == "Assessment"]['installation_id'].drop_duplicates().tolist()
    train101= train101[train101['installation_id'].isin(train_ids_with_subms)]
    
    
    test_ids_with_subms = test101[test101['type'] == "Assessment"]['installation_id'].drop_duplicates().tolist()
    test101= test101[test101['installation_id'].isin(test_ids_with_subms)]
   
    print( "keep ..only with assessments..")
    
    print(train101.shape)
    print(test101.shape)
 
    # convert text into datetime
    train101['timestamp'] = pd.to_datetime(train101['timestamp'])
    test101['timestamp'] = pd.to_datetime(test101['timestamp'])  
    
    
    train101.rename(columns={'type': 'sesion_type'}, inplace=True)
    test101.rename(columns={'type': 'sesion_type'}, inplace=True)
    
    
    train101['title'] = train101['title'].str.slice(0,7).str.replace(' ','_') + "_" + train101['sesion_type'].str.slice(0,2)
    test101['title'] = test101['title'].str.slice(0,7).str.replace(' ','_') + "_" + test101['sesion_type'].str.slice(0,2)
    
   
    print("raw files reading completed...")
    return train101, test101, train_labels, specs, sample_submission


# In[6]:


def test_last_row(test):
    test.sort_values(['installation_id','timestamp'], inplace=True)
    test.loc[test.groupby('installation_id')['event_code'].tail(1).index, 'event_code'] = 4100
    
    return test


# In[7]:


def pre_process(input_df):
    
    input_df.loc[((input_df['event_code']==4100) &  (input_df['title']=='Bird_Me_As')),'event_code']=1001
    x= input_df.copy()
    
    x.loc[((x['event_code']==4110) &  (x['title']=='Bird_Me_As')),'event_code']=4100
   
    del input_df
    gc.collect()
    
    #x = x[x.installation_id=='051794c4']
    
    return x


# In[8]:


def parse_json(input_df):
    
    variables_array = []
    event_data_dict = {}
    row_counter = 0
    for r in input_df.itertuples(name='Row', index=False):
        
        correct_cnt = np.nan
        wrong_cnt =np.nan
        #print(r.sesion_type)

        json_string = json.loads(r.event_data)

        if json_string.get('correct') is True:
            correct_cnt =1
            wrong_cnt = 0
        if json_string.get('correct') is False:
            wrong_cnt = 1
            correct_cnt = 0

        variables_array.append([r.installation_id, r.game_session, r.title, r.sesion_type ,
                                r.event_code , r.event_id,  r.game_time ,  r.timestamp, json_string.get('duration') , 
                                correct_cnt, wrong_cnt,
                                json_string.get('dwell_time')]) 
        row_counter +=1
    
    del input_df
    gc.collect()
    
    print(row_counter)
    print(len(variables_array))
    return pd.DataFrame(data=variables_array, columns=['installation_id','game_session','title', 'sesion_type',                                                       'event_code','event_id', 'game_time' ,  'timestamp', 'duration','correct_cnt', 'wrong_cnt','dwell_time' ])


# In[9]:


def cal_accuracy(input_df, session_type):
    
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
    
    if session_type=='Assessment':
        
        act_assm_gm =input_df.loc[ ((input_df['event_code']==4100) & (input_df['sesion_type']==session_type)) ,['installation_id','game_session','event_code','sesion_type', 'title','correct_cnt','wrong_cnt']]        .groupby(['installation_id','game_session','sesion_type','title']).agg({'correct_cnt':'sum', 'wrong_cnt':'sum'}).reset_index()
    else:
        act_assm_gm =input_df.loc[ (input_df['sesion_type']==session_type) ,['installation_id','game_session','event_code','sesion_type', 'title','correct_cnt','wrong_cnt']]        .groupby(['installation_id','game_session','sesion_type','title']).agg({'correct_cnt':'sum', 'wrong_cnt':'sum'}).reset_index()
        
    act_assm_gm['accuracy'] = act_assm_gm['correct_cnt']/(act_assm_gm['correct_cnt']+ act_assm_gm['wrong_cnt']) 
    act_assm_gm.head()

    act_assm_tm = input_df[['installation_id','game_session','event_code','sesion_type', 'title','duration']]    .groupby(['installation_id','game_session','sesion_type','title']).agg({'duration':'sum'}).reset_index()
    
    act_assm_tm.head()
    output_df = input_df[['installation_id','game_session','title','sesion_type','timestamp','game_time']]        .groupby(['installation_id','game_session','title','sesion_type']).agg({'game_time':'max', 'timestamp':'min'}).reset_index()

    output_df = pd.merge(output_df,act_assm_tm[['installation_id','game_session', 'duration']], on=['installation_id','game_session'], how='inner')
    
    #output_df['tm']=output_df['duration'].mask(pd.isnull, output_df['game_time'])
    output_df.loc[(output_df['game_time'] >= 5000000) ,'game_time'] = 5000000


    # bring in accuracy count

    output_df = pd.merge(output_df,act_assm_gm[['installation_id','game_session', 'correct_cnt','wrong_cnt','accuracy']],
                         on=['installation_id','game_session'], how='inner')
    output_df['acc_group'] = output_df['accuracy'].map(get_group)
    #output_df['tm_std'] = output_df.groupby('title')['tm'].transform(lambda x: minmax_scale(x.astype(float)))
    output_df['tm'] = output_df.groupby('title')['game_time'].transform(lambda x: minmax_scale(x.astype(float)))
    output_df['total_attemps'] = output_df['correct_cnt']+ output_df['wrong_cnt']
    output_df['mod_accuracy']=output_df.apply(lambda x: x['accuracy'] + (x['accuracy']*(1- x['tm'])) , axis=1)
    
    del input_df
    gc.collect()
    
    print(output_df.shape)
    
    return output_df


# In[10]:


def accuracy_pivot(pivot_df):
    
    
    # select all assessments except the 4100 of Bird_Measu_Assessment
    acc2 = pivot_df.pivot_table( index=['installation_id','game_session'],                                 columns='title', values=['correct_cnt','wrong_cnt','accuracy','acc_group','tm','mod_accuracy'],                                   aggfunc={ 'tm': np.sum, 
                                             'correct_cnt':np.sum,
                                            'wrong_cnt': np.sum , 
                                           'accuracy':np.sum ,
                                           'acc_group':np.sum,
                                            'mod_accuracy':np.sum
                                           }, fill_value=np.nan)


    acc2.columns.tolist()
    acc2.columns = ['_'.join(map(str,i)) for i in acc2.columns.tolist()]
    acc2 = acc2.reset_index()
    print(acc2.shape)

    # bring timestamp for correct accumulation of all numbers
    print(acc2.shape)
   
    acc_cumul = pd.merge(pivot_df[['installation_id','game_session','title','sesion_type']], 
                         acc2, on= ['installation_id','game_session'], how= 'inner')
    
    # this should be same as training lables file..
    acc_cumul.shape
    
    del pivot_df
    gc.collect()
    
    return acc_cumul


# In[11]:


def get_event_pivot(input_df, pivot_column):
    
    if pivot_column=='event_id':
        code_list= [3110,4070,3010,4020,2020,2030,3020,3021,3120,3121]
    else:
        code_list= [4010,2030,4025,4030,3020,3010,4100,3110,4020,4070,4220,2020,4110,
                    4090,4031,3121,2010,3120,3021,2060,4035,2050,4045,4040,4022,2083,
                    2070,2075,2080,5010,2035,4235,4095,4021,5000,4230,2025,2081,2040]

    
    
    x= input_df.loc[input_df['event_code'].isin(code_list), ['installation_id','game_session','event_code','event_id']]
    t1 = x.pivot_table( index=['installation_id','game_session'],
                                                        columns=[pivot_column],
                                                        values=[pivot_column], 
                                                        aggfunc= { pivot_column: 'size' }, 
                                                        fill_value=np.nan)
    t1.columns.tolist()
    t1.columns = ['_cnt_'.join(map(str,i)) for i in t1.columns.tolist()]
    t1 = t1.reset_index()
    print(t1.shape)
    
    del input_df
    gc.collect()

    return t1
    


# In[12]:


def get_other_kpi(input_df, n_sessions):


    input_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in input_df.columns]
    input_df.sort_values(['installation_id', 'timestamp'], inplace=True)
    variables_array = []
    fixed_cols =  ['installation_id', 'timestamp', 'game_session', 'title', 'sesion_type']
    drop_cols =  ['game_time', 'duration']
    num_col_names = (set(input_df.columns.tolist())- set(fixed_cols) -set(drop_cols) )


    

    for inst_id, g_session in input_df.groupby('installation_id'):
        kpi_deque = {name:deque([np.nan],n_sessions) for name in num_col_names}
        
        for r in g_session.itertuples(name='Row', index=False):
            #print(inst_id)
            if r.sesion_type == 'Assessment':

                accuracy_dict = {name:0 for name in num_col_names}
                fixed_cols_dict = {name:'' for name in fixed_cols } 
                # update accuracy_dict 
                for i in num_col_names:
                    accuracy_dict[i]= np.nanmean (kpi_deque[i])
                    kpi_deque[i].append(getattr(r, i))
                for i in fixed_cols:
                    fixed_cols_dict[i]=getattr(r, i) 

                variables_array.insert(len(variables_array), {**fixed_cols_dict,**accuracy_dict})      
            else:
                for i in num_col_names:
                    kpi_deque[i].append(getattr(r, i))
    del input_df
    gc.collect()
                    
    return pd.DataFrame(variables_array)


# In[13]:


def get_agg(input_df, lable_df,df_type):

    tr_cols = input_df.columns.tolist()
    
    fixed_cols= [ 'installation_id', 'game_session','title','timestamp','sesion_type']
    event_code_columns = [s for s in tr_cols if "event_code" in s ]
    event_id_columns = [s for s in tr_cols if "event_id" in s ]
    accu_columns =  [s for s in tr_cols if "_As" in s ]
    game_columns =  [s for s in tr_cols if "_Ga" in s ]
    first_df_col = fixed_cols + accu_columns
    second_df_col = fixed_cols + event_id_columns
    third_df_col =  fixed_cols + event_code_columns
    fourth_df_col = fixed_cols + game_columns

    processed_df1= get_other_kpi(input_df[first_df_col],40)
    processed_df2= get_other_kpi(input_df[second_df_col],40)
    processed_df3= get_other_kpi(input_df[third_df_col],40)
    processed_df4= get_other_kpi(input_df[fourth_df_col],40)

    
    
    processed_df = pd.merge(processed_df1,processed_df2, on =['installation_id','timestamp','game_session','title','sesion_type'], how='inner')
    processed_df = pd.merge(processed_df,processed_df3, on =['installation_id','timestamp','game_session','title','sesion_type'], how='inner')
    processed_df = pd.merge(processed_df,processed_df4, on =['installation_id','timestamp','game_session','title','sesion_type'], how='inner')
    
    if  df_type=='train':
        processed_df = pd.merge(processed_df,lable_df, on =['installation_id','game_session'], how='inner')
        
   
    del input_df
    del processed_df1
    del processed_df2
    del processed_df3
    del processed_df4
    
    gc.collect()
    
    processed_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in processed_df.columns]
    
    title_dict = {'Cart_Ba_As':0, 'Cauldro_As':1, 'Chest_S_As':2, 'Mushroo_As':3, 'Bird_Me_As':4}
    processed_df['title'] = processed_df['title'].map(title_dict)
    

    all_columns  = processed_df.columns.to_list()
    acc_columns= ['accuracy_Bird_Me_As', 'accuracy_Mushroo_As', 'accuracy_Cauldro_As', 'accuracy_Chest_S_As', 'accuracy_Cart_Ba_As']
    acc_grp_columns =  ['acc_group_Cauldro_As', 'acc_group_Bird_Me_As','acc_group_Chest_S_As','acc_group_Cart_Ba_As', 'acc_group_Mushroo_As']
    acc_mod_columns = ['mod_accuracy_Cauldro_As','mod_accuracy_Bird_Me_As','mod_accuracy_Chest_S_As','mod_accuracy_Mushroo_As','mod_accuracy_Cart_Ba_As']

    acc_cls = [elem for elem in acc_columns if  any(re.search('(^|\s){}(\s|$)'.format(c), elem) for c in all_columns)]
    acc_grp_cls = [elem for elem in acc_grp_columns if  any(re.search('(^|\s){}(\s|$)'.format(c), elem) for c in all_columns)]
    acc_mod_cls = [elem for elem in acc_mod_columns if  any(re.search('(^|\s){}(\s|$)'.format(c), elem) for c in all_columns)]
    
    
      
    gm_columns= ['accuracy_Crystal_Ga','accuracy_Pan_Bal_Ga','accuracy_Leaf_Le_Ga','accuracy_Chow_Ti_Ga',
                 'accuracy_Dino_Di_Ga','accuracy_Dino_Dr_Ga','accuracy_Happy_C_Ga','accuracy_Bubble__Ga',
                 'accuracy_Air_Sho_Ga','accuracy_Scrub_A_Ga','accuracy_All_Sta_Ga']
    
    gm_grp_columns= ['acc_group_Crystal_Ga','acc_group_Pan_Bal_Ga','acc_group_Leaf_Le_Ga','acc_group_Chow_Ti_Ga',
                 'acc_group_Dino_Di_Ga','acc_group_Dino_Dr_Ga','acc_group_Happy_C_Ga','acc_group_Bubble__Ga',
                 'acc_group_Air_Sho_Ga','acc_group_Scrub_A_Ga','acc_group_All_Sta_Ga']
    
    
        
    gm_cls =  [elem for elem in gm_columns if  any(re.search('(^|\s){}(\s|$)'.format(c), elem) for c in all_columns)]
    gm_grp_cls =  [elem for elem in gm_grp_columns if  any(re.search('(^|\s){}(\s|$)'.format(c), elem) for c in all_columns)]
    
    processed_df['gm_group_col_ax1']=  processed_df[gm_cls].sum(axis=1)
    processed_df['gm_accuracy_col_ax1']=  processed_df[gm_grp_cls].sum(axis=1)

    processed_df['gm_group_avg'] = processed_df.groupby(['installation_id'])['gm_group_col_ax1'].transform('mean')
    processed_df['gm_accuracy_avg'] = processed_df.groupby(['installation_id'])['gm_accuracy_col_ax1'].transform('mean')

   
    processed_df['acc_group_col_ax1']=  processed_df[acc_cls].sum(axis=1)
    processed_df['accuracy_col_ax1']=  processed_df[acc_grp_cls].sum(axis=1)
    processed_df['mod_acc_group_col_ax1']=  processed_df[acc_mod_cls].sum(axis=1)



    processed_df['acc_group_avg'] = processed_df.groupby(['installation_id'])['acc_group_col_ax1'].transform('mean')
    processed_df['accuracy_avg'] = processed_df.groupby(['installation_id'])['accuracy_col_ax1'].transform('mean')
    processed_df['mod_acc_group_avg'] = processed_df.groupby(['installation_id'])['mod_acc_group_col_ax1'].transform('mean')
    
    

    return processed_df


# In[14]:


train, test, train_labels, specs, sample_submission = process_raw_data(n1=3000000000,n2=1000000000)


# In[15]:


train = pre_process(train)
print(train.shape)
test = pre_process(test)
print(test.shape)


# In[16]:


test = test_last_row(test)
print(test.shape)


# In[17]:


json_df_train = parse_json(train)
print(json_df_train.shape)
gc.collect()
json_df_test = parse_json(test)
print(json_df_test.shape)
gc.collect()


# In[18]:


del train
del test
gc.collect()


# In[19]:


acc_df_train = cal_accuracy(json_df_train,'Assessment')
print(acc_df_train.shape)
acc_df_test =  cal_accuracy(json_df_test,'Assessment')
print(acc_df_test.shape)

#only for traindataset
lable_df = acc_df_train.loc[( acc_df_train.sesion_type=='Assessment'),['installation_id','game_session','mod_accuracy', 'acc_group'] ] 
#lable_df[lable_df.installation_id=='051794c4']
lable_df.head()


# In[20]:


acc_df_train[['acc_group', 'mod_accuracy']].groupby('acc_group').describe()


# In[21]:


gm_df_train = cal_accuracy(json_df_train,'Game' )
print(gm_df_train.shape)
gm_df_test =  cal_accuracy(json_df_test,'Game')
print(gm_df_test.shape)


# In[22]:


acc_dpivot_df_train = accuracy_pivot(acc_df_train)
print(acc_dpivot_df_train.shape)
acc_dpivot_df_test = accuracy_pivot(acc_df_test)
print(acc_dpivot_df_test.shape)


# In[23]:


gm_pivot_df_train = accuracy_pivot(gm_df_train)
print(gm_pivot_df_train.shape)
gm_pivot_df_test = accuracy_pivot(gm_df_test)
print(gm_pivot_df_train.shape)


# In[24]:


master_df_train = json_df_train[['installation_id','game_session','title', 'sesion_type','timestamp']]        .groupby(['installation_id','game_session','title','sesion_type']).agg({'timestamp':'min'}).reset_index()
master_df_train.shape
master_df_test = json_df_test[['installation_id','game_session','title', 'sesion_type','timestamp']]        .groupby(['installation_id','game_session','title','sesion_type']).agg({'timestamp':'min'}).reset_index()
print(master_df_train.shape)
print(master_df_test.shape)


# In[25]:


event_count_df_train = get_event_pivot(json_df_train, 'event_code')
event_count_df_test = get_event_pivot(json_df_test, 'event_code')


# In[26]:


event_id_count_df_train = get_event_pivot(json_df_train, 'event_id')
event_id_count_df_test = get_event_pivot(json_df_test, 'event_id')


# In[27]:


del json_df_train
del json_df_test
gc.collect()


# In[28]:


train_df_11 = pd.merge(master_df_train , acc_dpivot_df_train.drop(columns=['sesion_type']) , on =['installation_id','game_session'], how='left', suffixes=("", "_y"))
train_df_11 = pd.merge(train_df_11 , gm_pivot_df_train.drop(columns=['sesion_type']) , on =['installation_id','game_session'], how='left', suffixes=("", "_y"))
train_df_11 = pd.merge(train_df_11 , event_count_df_train , on =['installation_id','game_session'], how='left')
train_df_11 = pd.merge(train_df_11, event_id_count_df_train , on =['installation_id','game_session'], how='left')
train_df_11.shape


# In[29]:


test_df_11 = pd.merge(master_df_test , acc_dpivot_df_test.drop(columns=['sesion_type']) ,  on =['installation_id','game_session'], how='left', suffixes=("", "_y"))
test_df_11 = pd.merge(test_df_11 , gm_pivot_df_test.drop(columns=['sesion_type']) , on =['installation_id','game_session'], how='left', suffixes=("", "_y"))
test_df_11 = pd.merge(test_df_11 , event_count_df_test , on =['installation_id','game_session'], how='left')
test_df_11 = pd.merge(test_df_11, event_id_count_df_test , on =['installation_id','game_session'], how='left')
test_df_11.shape


# In[30]:


train_all = get_agg(train_df_11,lable_df,'train')
print( train_all.shape)
test_all = get_agg(test_df_11, 'NA','test')
print( test_all.shape)


# In[31]:


del train_df_11
del test_df_11
gc.collect()


# In[32]:


train_all[['mod_accuracy']].head()


# In[33]:


test_all = test_all.sort_values(['installation_id','timestamp']).groupby('installation_id', sort=False).tail(1)
test_all[test_all['installation_id'].isin(['048e7427'])] # sholud give last row


# In[34]:


print(train_all.shape)
print(test_all.shape)
train_all[['mod_accuracy']].head()


# In[35]:


cols_to_drop = [ 'installation_id', 'game_session','timestamp','sesion_type', 
                'tm_Bird_Me_As', 'tm_Cart_Ba_As', 'tm_Mushroo_As', 'tm_Cauldro_As', 
                'tm_Chest_S_As', 'tm_Pan_Bal_Ga', 'tm_Air_Sho_Ga']


train = train_all.drop(columns= cols_to_drop + ['mod_accuracy'] + ['acc_group'])
test = test_all.drop(columns=cols_to_drop)
acc = train_all['mod_accuracy'] # store training labels
feature_names = list(train.columns) # store feature names


# In[36]:


n_folds = 5
k_fold = KFold(n_splits = n_folds, shuffle=True,random_state=101)


# In[37]:


def reg_model(train,n_folds):
    
    validation_scores = [] 
    training_scrores = [] 
    imp_features = np.zeros(len(feature_names)) 
    oof = np.zeros(train.shape[0]) 
     
   
    

    for tr_idx, vld_idx in k_fold.split(train):
        print(k_fold.n_splits)
        x_train, x_train_acc = train.iloc[tr_idx], acc.iloc[tr_idx] 
        x_valid, x_valid_acc = train.iloc[vld_idx], acc.iloc[vld_idx] 

        model = lgb.LGBMRegressor( n_estimators=2000, 
                                    objective='regression',
                                    boosting_type='gbdt',
                                    metric= 'rmse',
                                    subsample= 0.75,
                                    subsample_freq= 1,
                                    learning_rate= 0.04,
                                    feature_fraction= 0.8,
                                    max_depth= 10,
                                    lambda_l1= 0.5,  
                                    lambda_l2= 0.5,
                                    verbose=100,
                                    early_stopping_rounds=100,
                                    random_state=50)

        model.fit(x_train, x_train_acc, 
                   eval_metric='rmse', 
                   eval_set = [(x_valid, x_valid_acc), (x_train, x_train_acc)],
                   eval_names = ['x_valid','x_train'], 
                   early_stopping_rounds = 100, 
                   verbose = 100,
                   categorical_feature = ['title'])
 
        best_iter = model.best_iteration_ 
        imp_features += model.feature_importances_/k_fold.n_splits 
        oof[vld_idx] = model.predict(x_valid, num_iteration = best_iter).reshape(-1,)/k_fold.n_splits

 
        valid_score = model.best_score_['x_valid']['rmse']
        train_score = model.best_score_['x_train']['rmse']

        validation_scores.append(valid_score)
        training_scrores.append(train_score)
        
    del x_train, x_valid
    gc.collect()
        
    return model, imp_features , oof, validation_scores, training_scrores


# In[38]:


model, feat_imp_vals , oof, valid_scores, train_scores =  reg_model(train,5)


# In[39]:


y_pred = model.predict(test)
y_pred.size


# In[40]:


y_pred[y_pred <= 0.7] = 0
y_pred[np.where(np.logical_and(y_pred > 0.7, y_pred <= 1.3))] = 1
y_pred[np.where(np.logical_and(y_pred > 1.3, y_pred <=1.85))] = 2
y_pred[y_pred > 1.85] = 3


# In[41]:


x11= test_all.copy()
x11['accuracy_group']= y_pred
sample_submission =pd.merge(sample_submission[['installation_id']], x11[['installation_id','accuracy_group']], 
                            on='installation_id' , how='inner')
sample_submission['accuracy_group'] = sample_submission['accuracy_group'].astype(int)
print(sample_submission.shape)
sample_submission.head()


# In[42]:


sample_submission.to_csv('submission.csv', index=False)

