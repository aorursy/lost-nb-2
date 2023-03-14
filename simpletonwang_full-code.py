#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from catboost import CatBoostRegressor
from matplotlib import pyplot
import shap
import random
random.seed(42)
import os


# Any results you write to the current directory are saved as output.
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import json
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 5000)


# In[2]:


os.listdir('../input/data-science-bowl-2019/')


# In[3]:


def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission


# In[4]:


train, test, train_labels, specs, sample_submission = read_data()


# In[5]:


def time_feature(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    return df


# In[6]:


train = time_feature(train)
test = time_feature(test)


# In[7]:


train.head()


# In[8]:


title_list = list(set(train['title'].unique()).union(set(test['title'].unique())))
event_id_list = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
event_code_list = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
world_list = list(set(train['world'].unique()).union(set(test['world'].unique())))
type_list = list(set(train['type'].unique()).union(set(test['type'].unique())))
hour_list = list(set(train['hour'].unique()).union(set(test['hour'].unique())))
assessment_list = ['Bird Measurer (Assessment)', 'Cart Balancer (Assessment)', 'Cauldron Filler (Assessment)',
                   'Chest Sorter (Assessment)', 'Mushroom Sorter (Assessment)']


# In[9]:


mini_train = train.iloc[:50000]


# In[10]:


# this is the function that convert the raw data into processed features
def tianqi_get_data(user_sample, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    
    type_count: Dict[str, int] = {str(tp)+'_cnt': 0 for tp in type_list}
    event_code_count: Dict[str, int] = {str(event_code)+'_cnt': 0 for event_code in event_code_list}
    event_id_count: Dict[str, int] = {str(event_id)+'_cnt': 0 for event_id in event_id_list}
    title_count: Dict[str, int] = {str(title)+'_cnt': 0 for title in title_list}
    world_count: Dict[str, int] = {str(world)+'_cnt': 0 for world in world_list}
    hour_count: Dict[str, int] = {str(hour)+'_cnt': 0 for hour in hour_list}
            
               
    all_assessments = []
    since_last_assessment = []
    last_assessment_incorret = None
    last_title_assessment_accuracy = {'last_'+str(assess)+'_acc': None for assess in assessment_list}
    accumulated_assessment_correct = {'accumulated_'+str(assess)+'_correct': 0 for assess in assessment_list}
    accumulated_assessment_incorrect = {'accumulated_'+str(assess)+'_incorrect': 0 for assess in assessment_list}
    
    all_attemp_correct = 0
    all_attemp_incorrect = 0
    
    
    recent_attempts = [None, None, None, None, None]
    
    
    accumulated_incorrect = 0
    accumulated_correct = 0
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        session_title = session['title'].iloc[0]
        session_type = session['type'].iloc[0]
        win_code = 4110 if session_title == 'Bird Measurer (Assessment)' else 4100
        session_hour = session['hour'].iloc[0]
        hour_count[str(session_hour)+'_cnt'] += 1
        
        if (session_type == 'Clip'):
            type_count[session_type+'_cnt'] += 1
            title_count[session_title+'_cnt'] += 1
            
        if (session_type == 'Activity'):
            type_count[session_type+'_cnt'] += 1
            title_count[session_title+'_cnt'] += 1
            event_code_map = Counter(session['event_code'])
            event_id_map = Counter(session['event_id'])
            
            for event_code in event_code_map:
                event_code_count[str(event_code)+'_cnt'] += event_code_map[event_code]
                
            event_id_map = dict(session['event_id'].value_counts())
            for event_id in event_id_map:
                event_id_count[str(event_id)+'_cnt'] += event_id_map[event_id]
                
        if (session_type == 'Game'):
            type_count[session_type+'_cnt'] += 1
            title_count[session_title+'_cnt'] += 1
            
            event_code_map = Counter(session['event_code'])
            event_id_map = Counter(session['event_id'])
            
            true_attempts = len(session[session['event_code']==3021])
            false_attempts = len(session[session['event_code']==3020])
            
            all_attemp_correct += true_attempts
            all_attemp_incorrect += false_attempts
            
            if true_attempts+false_attempts>0:
                if true_attempts==0: accuracy=-1
                else: accuracy = true_attempts/(true_attempts+false_attempts)
                _ = recent_attempts.pop(0)
                recent_attempts.append(accuracy)
            
            
            for event_code in event_code_map:
                event_code_count[str(event_code)+'_cnt'] += event_code_map[event_code]
                
            event_id_map = dict(session['event_id'].value_counts())
            for event_id in event_id_map:
                event_id_count[str(event_id)+'_cnt'] += event_id_map[event_id]
                
            
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session[session['event_code']==win_code]

            
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            
#             start_time = session['timestamp'].iloc[0]
#             end_time = session['timestamp'].iloc[-1]

            features = {'correct_num': true_attempts,
                        'incorrect_num': false_attempts}
            
            features['installation_id'] = session['installation_id'].iloc[-1]
            features['assess_hour_precnt'] = hour_count[str(session['hour'].iloc[-1])+'_cnt']
            features['game_session'] = i
            features['world']  = session['world'].iloc[-1]
            
            features['assess_weekday']  = session['timestamp'].dt.weekday.iloc[-1]
            features['title'] = session_title
            features['accumulated_incorrect'] = accumulated_incorrect
            features['accumulated_correct'] = accumulated_correct
            features['accumulated_accuracy'] = None if (accumulated_incorrect+accumulated_correct)==0 else                                                accumulated_correct/(accumulated_incorrect+accumulated_correct)
            
            features['accumulated_title_correct'] = accumulated_assessment_correct['accumulated_'+str(session_title)+'_correct']
            features['accumulated_title_incorrect'] = accumulated_assessment_incorrect['accumulated_'+str(session_title)+'_incorrect']
            features['accumulated_title_accuracy'] = None if (features['accumulated_title_incorrect']+features['accumulated_title_correct'])==0 else                                                features['accumulated_title_correct']/(features['accumulated_title_incorrect']+features['accumulated_title_correct'])
            features['last_title_acc'] = last_title_assessment_accuracy['last_'+str(session_title)+'_acc']
            
            recent_acc = {
                'pre_five_attempt_acc': recent_attempts[0],
                'pre_four_attempt_acc': recent_attempts[1],
                'pre_three_attempt_acc': recent_attempts[2],
                'pre_two_attempt_acc': recent_attempts[3],
                'pre_one_attempt_acc': recent_attempts[4],
                          
            }
            
            features['all_attemp_correct'] = all_attemp_correct
            features['all_attemp_incorrect'] = all_attemp_incorrect
            features['all_attemp_accuracy'] = None if (features['all_attemp_correct']+features['all_attemp_incorrect'])==0 else                                                features['all_attemp_correct']/(features['all_attemp_correct']+features['all_attemp_incorrect'])
            features.update(recent_acc)
            features.update(type_count)
            features.update(title_count)
            features.update(event_code_count)
            features.update(event_id_count)
            
#             features['assess_hour']  = session['timestamp'].dt.hour.iloc[-1]
#             features.update(hour_count)
            
            if test_set or (true_attempts+false_attempts > 0):
                all_assessments.append(features)
                
                last_title_assessment_accuracy['last_'+str(session_title)+'_acc'] = true_attempts/(true_attempts+false_attempts)
                
            
            
            type_count[session_type+'_cnt'] += 1
            title_count[session_title+'_cnt'] += 1
            
            
            accumulated_correct += true_attempts
            accumulated_incorrect += false_attempts
            accumulated_assessment_correct['accumulated_'+str(session_title)+'_correct'] += true_attempts
            accumulated_assessment_incorrect['accumulated_'+str(session_title)+'_incorrect'] += false_attempts
            
            event_code_map = Counter(session['event_code'])
            event_id_map = Counter(session['event_id'])
            
            for event_code in event_code_map:
                event_code_count[str(event_code)+'_cnt'] += event_code_map[event_code]
                
            event_id_map = dict(session['event_id'].value_counts())
            for event_id in event_id_map:
                event_id_count[str(event_id)+'_cnt'] += event_id_map[event_id]

                
            true_attempts = len(session[session['event_code']==3021])
            false_attempts = len(session[session['event_code']==3020])
            
            all_attemp_correct += true_attempts
            all_attemp_incorrect += false_attempts
            
            if true_attempts+false_attempts>0:
                if true_attempts==0: accuracy=-1
                else: accuracy = true_attempts/(true_attempts+false_attempts)
                _ = recent_attempts.pop(0)
                recent_attempts.append(accuracy)
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments


# In[11]:


def get_train_and_test(train, test=None):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += tianqi_get_data(user_sample)
    reduce_train = pd.DataFrame(compiled_train)
    
    if test is not None:
        for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
            test_data = tianqi_get_data(user_sample, test_set = True)
            compiled_test.append(test_data)
        reduce_test = pd.DataFrame(compiled_test)
        return reduce_train, reduce_test
    return reduce_train


# In[12]:


tt = get_train_and_test(mini_train)


# In[13]:


tt.head(5)


# In[14]:


reduce_train, reduce_test = get_train_and_test(train, test)


# In[15]:


def eval_qwk_lgb_regr(mark, y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """
    dist = Counter(mark)
    for k in dist:
        dist[k] /= len(mark)
#     mark.hist()
    
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)

    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# In[16]:


final_train = reduce_train[reduce_train['correct_num']+reduce_train['incorrect_num']>0]
final_test = reduce_test.copy()


# In[17]:


final_train.shape


# In[18]:


final_train['accuracy_group'] = 3
final_train.loc[final_train['incorrect_num']==1,'accuracy_group'] = 2
final_train.loc[final_train['incorrect_num']>1,'accuracy_group'] = 1
final_train.loc[final_train['correct_num']==0,'accuracy_group'] = 0


# In[19]:


remove_features = ['installation_id', 'game_session', 'correct_num', 'incorrect_num']
TARGET = 'accuracy_group'
features = [col for col in final_train.columns if col not in remove_features and col!=TARGET]
cat_features = [col for col in features if final_train[col].dtype == 'O' or final_test[col].dtype == 'O']
for col in cat_features:
    final_train[col] = final_train[col].astype('category')
    final_test[col] = final_test[col].astype('category')


# In[20]:


final_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in final_train.columns]
final_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in final_test.columns]


# In[21]:


features = [col for col in final_train.columns if col not in remove_features and col!=TARGET]


# In[22]:


import random
random.seed(42)


# In[23]:



installation_id_list = list(set(train_labels.installation_id.unique()))
train_install_id = set(random.sample(installation_id_list, 2800))
valid_install_id = set(installation_id_list) - train_install_id


# In[24]:


train_data = final_train[final_train['installation_id'].isin(train_install_id)]


# In[25]:


valid_data = pd.DataFrame()
valid_session_id = final_train[final_train['installation_id'].                           isin(valid_install_id)].groupby('installation_id').                            apply(lambda x: random.choice(x['game_session'].values))
valid_data = final_train[final_train['game_session'].isin(valid_session_id)]


# In[26]:


print(train_data[features].shape)
print(valid_data[features].shape)


# In[27]:


params = {'n_estimators':5000,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'subsample': 0.75,
            'subsample_freq': 1,
            'learning_rate': 0.01,
            'feature_fraction': 0.6,
            'max_depth': 10,
            'lambda_l1': 1,  
            'lambda_l2': 1,
            'early_stopping_rounds': 100,
            'seed': 42,
            'n_jobs': 8
            }


# In[28]:


tr_data = lgb.Dataset(train_data[features], train_data[TARGET], categorical_feature=cat_features)
vl_data = lgb.Dataset(valid_data[features], valid_data[TARGET], categorical_feature=cat_features)


# In[29]:


cat_features


# In[30]:


model = lgb.train(
                params,
                tr_data,
                valid_sets = [tr_data,vl_data],
                verbose_eval = 100,
            )


# In[31]:


y_pred = model.predict(valid_data[features])
y_true = valid_data[TARGET]
kappa = eval_qwk_lgb_regr(train_data[TARGET], y_true, y_pred)


# In[32]:


kappa


# In[33]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=28)
installation_id_list = final_train.installation_id.unique()


# In[34]:


# put numerical value to one of bins
def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)

class OptimizedRounder3(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y, idx):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        ll = -cohen_kappa_score(y, X_p, weights='quadratic')
        return ll

    def fit(self, X, y):
        coef = [0.15, 0.3, 0.50]
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(0.075, 0.225), (0.2, 0.4), (0.4, 0.6)]
        for it1 in range(10):
            for idx in range(3):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                coef[idx] = a
                la = self._loss(coef, X, y, idx)
                coef[idx] = b
                lb = self._loss(coef, X, y, idx)
                for it in range(20):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        coef[idx] = a
                        la = self._loss(coef, X, y, idx)
                    else:
                        b = b - (b - a) * golden2
                        coef[idx] = b
                        lb = self._loss(coef, X, y, idx)
        self.coef_ = {'x': coef}

    def predict(self, X, coef):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[35]:


def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000 
    x =  preds-labels    
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess


# In[36]:


from scipy.stats import rankdata


# In[37]:


# test_predict = []
pred = np.zeros(shape = (len(sample_submission), 5))
AVG_NUM = 50
all_round_kappa = 0
all_round_truncated_kappa = 0
overall_truncated_kappa = 0
overall_kappa = 0
rmse = 0
for fold, (train_idx, valid_idx) in enumerate(kf.split(installation_id_list)):
    print(f'Fold: {fold}')
    print(f'Train installation: {len(train_idx)} Valid installation: {len(valid_idx)}')
    train_install_id = installation_id_list[train_idx]
    valid_install_id = installation_id_list[valid_idx]

    random.seed(23)
    
    train_data = final_train[final_train['installation_id'].isin(train_install_id)]
    valid_data = final_train[final_train['installation_id'].isin(valid_install_id)].reset_index()
    
    print(f'Train data size: {len(train_data)}, Valid data size: {len(valid_data)}')
    tr_data = lgb.Dataset(train_data[features], train_data[TARGET], categorical_feature=cat_features)
    vl_data = lgb.Dataset(valid_data[features], valid_data[TARGET], categorical_feature=cat_features)
    
    
    
    model = lgb.train(
                params,
                tr_data,
                valid_sets = [tr_data,vl_data],
                verbose_eval = 100,
                fobj=cauchyobj
            )
    y_pred = model.predict(valid_data[features])
    y_pred_rank = rankdata(y_pred)/len(y_pred)
    y_true = valid_data[TARGET]
    rmse += np.sqrt(np.mean(np.square(y_true - y_pred)))/5
#     print(rmse)
    kappa = eval_qwk_lgb_regr(train_data[TARGET], y_true, y_pred)
    print(f'Kappa on all valid: {kappa:.3f}')
    
    y_pred_train = model.predict(train_data[features])
    y_pred_train_rank = (rankdata(y_pred_train)/len(y_pred_train))
    optR = OptimizedRounder3()
    optR.fit(y_pred_train_rank, train_data[TARGET].values)
    round_kappa = cohen_kappa_score(optR.predict(y_pred_rank, coef=optR.coefficients()),y_true,weights='quadratic')
    print(f'Round Kappa on all valid: {round_kappa:.3f}')

    truncated_rmse = 0
    truncated_kappa = 0
    truncated_round_kappa = 0

    for i in range(AVG_NUM):
        random.seed(28+i)
        eval_idx = valid_data.groupby('installation_id')['game_session'].                                        apply(lambda x: random.choice(x.index.values))
        y_eval_pred = model.predict(valid_data.loc[eval_idx, features])
        y_eval_pred_rank = (rankdata(y_eval_pred)/len(y_eval_pred))
        y_eval_true = valid_data.loc[eval_idx, TARGET]
        truncated_kappa += eval_qwk_lgb_regr(train_data[TARGET], y_eval_true, y_eval_pred)/AVG_NUM
        truncated_rmse += np.sqrt(np.mean(np.square(y_eval_true - y_eval_pred)))/AVG_NUM
        truncated_round_kappa += cohen_kappa_score(optR.predict(y_eval_pred_rank, coef=optR.coefficients()),y_eval_true,weights='quadratic')/AVG_NUM
    
    print(f'Truncated Kappa: {truncated_kappa:.3f}')
    print(f'Truncated Round Kappa: {truncated_round_kappa:.3f}')
    print(f'Truncated RMSE: {truncated_rmse:.3f}')
    
    all_round_kappa += round_kappa/5
    overall_kappa += kappa/5
    overall_truncated_kappa += truncated_kappa/5
    all_round_truncated_kappa += truncated_round_kappa/5
    
    pred[:, fold] = rankdata(model.predict(final_test[features]))/len(final_test)
print(f'CV Kappa: {overall_kappa}, RMSE: {rmse}')
print(f'CV Round Kappa: {all_round_kappa}')
print(f'Truncated CV Kappa: {overall_truncated_kappa}')
print(f'Truncated Round CV Kappa: {all_round_truncated_kappa}')


# In[38]:


# tr_data = lgb.Dataset(final_train[features], final_train[TARGET], categorical_feature=cat_features)
# params['n_estimators']=1000
# model = lgb.train(
#             params,
#             tr_data,
#             num_boost_round=1000,
#             valid_sets = [tr_data],
#             verbose_eval = 100,
#             fobj=cauchyobj,
#         )
# pred = model.predict(final_test[features])


# In[39]:


pred_train = model.predict(final_train[features])
pred_train_rank = rankdata(pred_train)/len(pred_train)


# In[40]:


optR = OptimizedRounder3()

optR.fit(pred_train_rank, final_train[TARGET].values)
final_pred = optR.predict(pred.mean(axis=1), coef=optR.coefficients())
final_pred[final_pred==4]=3


# In[41]:


def calibrate(y_pred, train_t):
    """
    Fast cappa eval function for lgb.
    """
#     dist = Counter(train_t['accuracy_group'])
#     for k in dist:
#         dist[k] /= len(train_t)
    dist = [0.25740936499993927, 0.1254762196691288, 0.12295055710383329]
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, y_pred)))
    
    return y_pred

def predict(sample_submission, y_pred):
    sample_submission['accuracy_group'] = y_pred
    sample_submission['accuracy_group'] = sample_submission['accuracy_group'].astype(int)
    sample_submission.to_csv('submission.csv', index = False)
    print(sample_submission['accuracy_group'].value_counts(normalize = True))


# In[42]:


# final_pred = calibrate(pred, train_labels)


# In[43]:


predict(sample_submission,final_pred)


# In[44]:


final_test.shape


# In[45]:


final_test.head()


# In[46]:


sample_submission.head()

