#!/usr/bin/env python
# coding: utf-8

# In[1]:


print(wrong)
import os
import time
os.system('pip install catboost==0.15.2')
os.system('pip install xlrd')
os.system('pip install lightgbm')
os.system('pip install tqdm')
os.system('pip install gensim')

import pandas as pd
import numpy as np

from tqdm import tqdm_notebook
from sklearn.metrics import roc_auc_score
import gc

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import datetime
def tiem_deta_date(data,new_data_col):
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    data[new_data_col] = startdate + data['TransactionDT'].map(lambda x:datetime.timedelta(seconds = x))
    return data
    
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


import os
os.chdir('/cos_person/IEEE/data/')





folder_path = '../data/'

print('Loading data...')

train_identity = pd.read_csv('{}train_identity.csv'.format(folder_path))
print('\tSuccessfully loaded train_identity!')

train_transaction = pd.read_csv('{}train_transaction.csv'.format(folder_path))
print('\tSuccessfully loaded train_transaction!')

test_identity = pd.read_csv('{}test_identity.csv'.format(folder_path))
print('\tSuccessfully loaded test_identity!')

test_transaction = pd.read_csv('{}test_transaction.csv'.format(folder_path))
print('\tSuccessfully loaded test_transaction!')

sub = pd.read_csv('{}sample_submission.csv'.format(folder_path))
print('\tSuccessfully loaded sample_submission!')

print('Data was successfully loaded!\n')    



def id_split(dataframe):
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]

    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]

    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]

    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]

    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1
    gc.collect()
    
    return dataframe


train_identity = id_split(train_identity)
test_identity = id_split(test_identity)
# had_id、
some_feat_ = []
for i in range(4):
    for df_data in [train_identity,test_identity]:
        df_data['id_31_back_' + str(i)] = df_data['id_31'].str.split(' ').str[-(i + 1)]
        some_feat_.append('id_31_back_' + str(i))
# some_feat_ = list(set(some_feat_))

print('Merging data...')
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

print('Data was successfully merged!\n')

# del train_identity, train_transaction, test_identity, test_transaction
print('Train dataset has {} rows and {} columns.'.format(train.shape[0],train.shape[1]))
print('Test dataset has {} rows and {} columns.\n'.format(test.shape[0],test.shape[1]))


gc.collect()


###########public kernels************************ 

# 'V_NULL','V_NULL_count',
useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',
                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2','C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',
                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',
                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',
                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',
                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',
                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',
                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'device_name', 'device_version', 'OS_id_30', 'version_id_30',
                   'browser_id_31', 'version_id_31', 'screen_width', 'screen_height', 'had_id']



cols_to_drop = [col for col in train.columns if col not in useful_features]
cols_to_drop.remove('isFraud')
cols_to_drop.remove('TransactionDT')
cols_to_drop.remove('TransactionID')



train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

i_cols = [c for c in train.columns if c.startswith('M')]
for df in [train, test]:
    df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)
    df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)


a = np.zeros(train.shape[0])
train["lastest_browser"] = a
a = np.zeros(test.shape[0])
test["lastest_browser"] = a
def setbrowser(df):
    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1
    return df
train=setbrowser(train)
test=setbrowser(test)                               




train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')




# New feature - log of transaction amount. ()
train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])

# New feature - decimal part of the transaction amount.
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

# New feature - day of week in which a transaction happened.
train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

# New feature - hour of the day in which a transaction happened.
train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24
test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24

for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 
                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:

    f1, f2 = feature.split('__')
    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    le = LabelEncoder()
    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
    train[feature] = le.transform(list(train[feature].astype(str).values))
    test[feature] = le.transform(list(test[feature].astype(str).values))



sep_count = ['id_01', 'id_31', 'id_33', 'id_36']
cateCOLS = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6','id_30','id_31','addr1','addr2','P_emaildomain','R_emaildomain','ProductCD','DeviceType', 'DeviceInfo', 'device_name', 'device_version']
_count_full = [c for c in cateCOLS if c not in sep_count]
# for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36','addr1','addr2']:
# testing adding more full count
for feature in _count_full:    
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

# Encoding - count encoding separately for train and test
for feature in ['id_01', 'id_31', 'id_33', 'id_36']:
    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))

###########public kernels************************




df_data = pd.concat([train,test])

def agg_mode(dataDetails, gpID, modeCol, modeReCol ):
    df = dataDetails.groupby( [gpID, modeCol] )[ gpID ].agg({ 'cnt':'count' }).reset_index()
    df = df.sort_values( 'cnt' )
    df = df.groupby( gpID ).tail(1)
    df.drop( 'cnt',axis = 1 ,inplace = True)
    df.columns = [ gpID,modeReCol]
    return df
    
#############boost 0.5k feats   fill card2~5 by mod of card1
for car in ['card{}'.format(i) for i in range(2,7)]:
    gp = agg_mode(df_data,'card1',car,'tmpcard')
    gp = gp.groupby(['card1'])['tmpcard'].first()
    df_data['tmpcard'] = df_data['card1'].map(gp)
    df_data[car] = df_data[car].fillna(df_data['tmpcard'])
    del df_data['tmpcard']
    

train = df_data[df_data.isFraud.isnull()==False]
test = df_data[df_data.isFraud.isnull()==True]
del df_data



emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']

for c in ['P_emaildomain', 'R_emaildomain']:
    train[c + '_bin'] = train[c].map(emails)
    test[c + '_bin'] = test[c].map(emails)
    
    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])
    
    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')    


#############boost 1k+ feats  making uid
train['uid'] = train['card1'].astype(str)+'_'+train['card2'].astype(str)+'_'+train['card3'].astype(str)+'_'+train['card4'].astype(str)
test['uid'] = test['card1'].astype(str)+'_'+test['card2'].astype(str)+'_'+test['card3'].astype(str)+'_'+test['card4'].astype(str)

train['uid2'] = train['uid'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['addr2'].astype(str)
test['uid2'] = test['uid'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['addr2'].astype(str)

train['uid3'] = train['card1'].astype(str)+'_'+train['addr1'].astype(str)
test['uid3'] = test['card1'].astype(str)+'_'+test['addr1'].astype(str)



# summary on total data 
#########################################################################
df_data = pd.concat([train,test])
cate_cols = ['ProductCD','card4','card6']
for col in cate_cols:
    agg_data = train.groupby([col])['isFraud'].agg(['mean','sum']).reset_index()
    agg_data.columns = [col,col+'_label_mean',col+'_label_sum']
    df_data = pd.merge(df_data,agg_data,'left',col)
    

df_data = tiem_deta_date(df_data,'Transaction_dt')
df_data['TransDateStr'] = df_data['Transaction_dt'].map(lambda x:str(x)[:10])
alldays = sorted(df_data['TransDateStr'].unique().tolist())
allDay_dic = dict( zip( alldays,range( len(alldays) ) ) )
df_data['TransDateStr'] = df_data['TransDateStr'].map(allDay_dic )
df_data['TransDateStr'] = df_data['TransDateStr']//120


# 0.5k rank of count by 120days
cards = ['card1','card2','card3','card5']
for c in cards:
    col = '{}_120days'.format(c)
    print('adding card ranking col {}'.format(col))
    df_data['{}_120days'.format(c)] = df_data[c].map(str) + '_days_'+df_data['TransDateStr'].map(str)
    df_data[col+'_rank_count'] = df_data[col].map( df_data[col].value_counts() )
    df_data[col+'_rank_count'] = df_data.groupby( [ 'TransDateStr'] )[ col+'_rank_count' ].rank( ascending = False)
    del df_data[col]




#############boost 2k+ feats  making uid
#**************so called magic here*********
## *** features :first by make uid like V307 which is cumsum of user payment
# than do some fe base on it
# Magic one 
def _V307_series(AMT_V307):
    AMT = AMT_V307[0]
    V307 = AMT_V307[1]   
    a =  pd.Series(V307)
    zero_index = np.where(a == 0)[0]
    _len = len(AMT)
    series = np.zeros(_len)
    if _len==1:
        return series
    flag = 0 
    for idx in zero_index:
        flag += 1
        series[idx] = flag
        cursor = AMT[idx ]
        for j in range( idx+1,_len ):
            if abs(V307[j]-cursor)<=0.01:
                cursor = AMT[j] + cursor
                series[j] = flag
    return series
def _join(AMT,V307):
    result = [AMT,V307]
    return result






cards = ['card1','card2','card3','card4','card5','card6','addr1']
cards =['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1']
df_data['cards'] = ''
for ca in cards:
    df_data['cards'] = df_data['cards'] + '-' + df_data[ca].map(str)
    
df_data = df_data.sort_values(['card1','TransactionDT']).reset_index(drop=True)
# 2 k
D2    = df_data.groupby(['card1']).apply(lambda x: _join( x['TransactionAmt'].values.tolist(),x['V307'].values.tolist() )).reset_index()
D2.columns = ['cards','joinlists']
D2['V307_series'] = D2['joinlists'].map(_V307_series)
D2_series = []
for val in tqdm(D2['V307_series'].values):
    D2_series.extend(val)
df_data['V307_series'] = D2_series

df_data['card1_V307_series'] =df_data['card1'].map(str) +'_V307_'+ df_data['V307_series'].map(str)

df_data['cards_V307_series'] =df_data['cards'].map(str) +'_V307_'+ df_data['V307_series'].map(str)

df_data['card1_addr1_V307_series'] =df_data['card1'].map(str) +df_data['addr1'].map(str) +'_V307_'+ df_data['V307_series'].map(str)

del df_data['V307_series']

from tqdm import tqdm


#############boost 2k+ feats  making uid
#**************so called magic here*********
## *** features : first by make uid like D1/D2/D4  which is reference day for
# some events ,thanks to @tuttifrutti; https://www.kaggle.com/tuttifrutti/creating-features-from-d-columns-guessing-userid
# Magic two


cards = ['card1','card2','card3','card4','card5','card6','addr1']
df_data['cards'] = ''
for ca in cards:
    df_data['cards'] = df_data['cards'] + '-' + df_data[ca].map(str)


df_data['DaysFromStart'] = np.round(df_data['TransactionDT'] / (60 * 60 * 24), 0)



cards = ['card1','card2','card3','card4','card5','card6','addr1']
Dids = []
for d in ['D1','D4','D15']:
    df_data['{}_resDay'.format(d)] = df_data['DaysFromStart'] - df_data[d]
    df_data['card1_{}_resDay'.format(d)] =df_data['card1'].map(str) +'_V307_'+ df_data['{}_resDay'.format(d)].map(str)
    df_data['cards_{}_resDay'.format(d)] =df_data['cards'].map(str) +'_V307_'+ df_data['{}_resDay'.format(d)].map(str)
    # df_data['card1_P_emaildomain_{}_resDay'.format(d)] =df_data['card1'].map(str) +df_data['P_emaildomain'].map(str) +'_V307_'+ df_data['{}_resDay'.format(d)].map(str)
    df_data['card1_addr1_{}_resDay'.format(d)] =df_data['card1'].map(str) +df_data['addr1'].map(str) +'_V307_'+ df_data['{}_resDay'.format(d)].map(str)
    # df_data['card1_addr1_P_emaildomain_{}_resDay'.format(d)] =df_data['card1'].map(str) +df_data['addr1'].map(str) +df_data['P_emaildomain'].map(str) +'_V307_'+ df_data['{}_resDay'.format(d)].map(str)
    #,'card1_P_emaildomain_{}_resDay'.format(d),'card1_addr1_P_emaildomain_{}_resDay'.format(d) 
    del df_data['{}_resDay'.format(d)]
    _add = ['card1_{}_resDay'.format(d),'cards_{}_resDay'.format(d),'card1_addr1_{}_resDay'.format(d)]
    Dids.extend(_add)



del df_data['DaysFromStart']
del df_data['cards']
# del df_data['D1_resDay']
# del df_data['D4_resDay']
# del df_data['D3_cumsum']

#*********************************
# add make uid
#*********************************





list_count = []
def Count_engineering(all_data, name):

    all_name = 'count_' + name
    all_data[all_name] = all_data.groupby([name])[name].transform('count')
    list_count.append(all_name)

    return all_data

x_list = ['TransactionAmt']

for name in tqdm(x_list):
    df_data = Count_engineering(df_data, name)

list_num = []
def Num_engineering(all_data, x_name, y_name):

    x_name = [x_name]

    all_name = ''

    for name in x_name:
        
        if all_data[name].nunique() >= all_data[y_name].nunique():
            return all_data
        
        all_name += (name + '_')

    all_name += ('num_' + y_name)
    if y_name=='D1':
        all_data[all_name + '_mean'] = all_data.groupby(x_name)[y_name].transform('mean')
    else:
        all_data[all_name + '_max'] = all_data.groupby(x_name)[y_name].transform('max')
        all_data[all_name + '_min'] = all_data.groupby(x_name)[y_name].transform('min')
        all_data[all_name + '_mean'] = all_data.groupby(x_name)[y_name].transform('mean')
        all_data[all_name + '_std'] = all_data.groupby(x_name)[y_name].transform('std')

    list_num.append(all_name)
    
    return all_data


# 'card1__D8', 'card1__id_02', 'addr1__D8', 'addr1__id_02', 'card1__dist1', 'card1__dist2', 'card1__V332', 'card1__V323'
xy_list = ['card1__TransactionAmt', 'card1__D8',
'uid__TransactionAmt','uid2__TransactionAmt','uid3__TransactionAmt',
'card2__TransactionAmt','card3__TransactionAmt','card5__TransactionAmt', 'card1__D8',
'card3__D8','addr1__id_02', 'card1__dist1', 'card1__dist2','card1__V332', 'addr1__id_02','card1__V99', 'addr1__V99',
# 'make_uid_first__TransactionAmt','make_uid_first__dist1',
'cards_D1_resDay__TransactionAmt',
# 'cards_D2_resDay__TransactionAmt',
'cards_D4_resDay__TransactionAmt',
# 'card1_V307_series__TransactionAmt',
'cards_V307_series__TransactionAmt',
# 'card1_addr1_V307_series__TransactionAmt',
'card1__D1','card2__D1','card3__D1','card5__D1','addr1__D1',
'card1__D3','card2__D3','card3__D3','card5__D3','addr1__D3',
'card1__V307','card2__V307','card3__V307','card5__V307','addr1__V307',

'card1__V130','card2__V130','card3__V130','card5__V130','addr1__V130',
'card1__V127','card2__V127','card3__V127','card5__V127','addr1__V127'
# 'cards_D3CUM_MinsDate__TransactionAmt'
# 'D1_PART_ID__TransactionAmt','D1_PART_ID__dist1',
# 'card1__D1','card2__D1','card3__D1','card5__D1','addr1__D1'

] ### new adding pair
# 'card3__D8','addr1__id_02', 'card1__dist1', 'card1__dist2','card1__V332', 'addr1__id_02','card1__V99', 'addr1__V99',



# 'card1__V99', 'addr1__V99', 'card1__V100', 'addr1__V100', 'card1__V139', 'addr1__V139'
#  dels




for xy_name in tqdm(xy_list):
    if 'dist2' in xy_name:
        continue
    x_name, y_name = xy_name.split('__')
    if y_name not in df_data.columns:
        continue
    df_data = Num_engineering(df_data, x_name, y_name)
    

uids = Dids + ['card1_V307_series','cards_V307_series','card1_addr1_V307_series']
for col in uids:
    ct = df_data[col].value_counts()
    df_data[col+'_cnt'] = df_data[col].map(ct)

cats = ['id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',
                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',
                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'device_name', 'device_version', 'OS_id_30', 'version_id_30',
                   'browser_id_31', 'version_id_31', ]

cats = cats+uids
cats = cats + ['uid','uid2','uid3']

df_data['tmpid'] = df_data['card1'].map(str) + '-' + df_data['addr1'].map(str)
for col in ['C1', 'C5', 'C6','V53', 'V138', 'C9']:
    new_id = 'card1_addr1_' + col
    new_id_count = 'card1_addr1_' + col +'_cnt'
    df_data[ new_id ] = df_data['tmpid'] + '-' + df_data[col].map(str)
    cats.append(new_id)
    df_data[ new_id_count ] = df_data[ new_id ].map(df_data[ new_id ].value_counts())
del df_data['tmpid']


train = df_data[df_data.isFraud.isnull()==False]
test = df_data[df_data.isFraud.isnull()==True]
del df_data



# combinations count of cates
cateCOLS = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6','id_30','id_31','addr1','addr2','P_emaildomain','R_emaildomain','ProductCD','DeviceType', 'DeviceInfo', 'device_name', 'device_version']
cats = cats+cateCOLS

use_join_nnq = []
for k,v in dict(train[cateCOLS].nunique()).items():
    if v>50:
        use_join_nnq.append(k)
        print(k)
for from_col in use_join_nnq:
    for to_col in use_join_nnq:
        if from_col == to_col:
            continue
        gp = pd.concat([ train[ [from_col,to_col] ],test[ [from_col,to_col] ] ]).groupby([from_col])[to_col].nunique()
        lennnq = len(set(gp.values.tolist()))
        if lennnq <10:
            print('from_col:{}, to_col : {} creating new col only {} values less than 10 ,will not use'.format(from_col,to_col,lennnq  ))
        else:
            train[ '{}_nnqof_{}'.format( from_col,to_col )] = train[from_col].map(gp)
            test[ '{}_nnqof_{}'.format( from_col,to_col )] = test[from_col].map(gp)

#to avoid overfit
tonan_theadhold_dict = {'C1':2000,'C2':2000,'C4':1000,'C6':1000,'C8':800,'C9':240,
                       'C14':240,'C10':520,'C11':520,'C12':520,'C13':1000}

for col,thod in tonan_theadhold_dict.items():
    train.loc[train[col]>= thod,col] = np.nan
    test.loc[test[col]>= thod,col] = np.nan

import datetime
def tiem_deta_date(data,new_data_col):
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    data[new_data_col] = startdate + data['TransactionDT'].map(lambda x:datetime.timedelta(seconds = x))
    return data


train = tiem_deta_date(train,'Transaction_dt')
test = tiem_deta_date(test,'Transaction_dt')

train['TransDateStr'] = train['Transaction_dt'].map(lambda x:str(x)[:10])
test['TransDateStr'] = test['Transaction_dt'].map(lambda x:str(x)[:10])
del train['Transaction_dt']
del test['Transaction_dt']


# for col in [c for c in data.columns if '_days_nunique' in c]:
#     del train[col]
#     del test[col]


    #*************************************
    #to avoid overfit
    #*************************************

for col in ['id_01','id_06','id_11','id_13','id_21','id_25','addr1__card1','card1']:
    if col in train.columns:
        print('deal with col {}'.format(col))
        thr = 7
        if 'card1' in col:
            thr = 1
        for data in [train,test]:
            data['value_happend_days'] = data[col].map(data.groupby([col])['TransDateStr'].nunique() )
            ## adding day nunique here
#             data[col +'_days_nunique'] = data[col].map(  (pd.concat( [train[[col,'TransDateStr' ]],test[[col,'TransDateStr' ]]] ) ).groupby([col])['TransDateStr'].nunique()  )
            data.loc[ data['value_happend_days']<=thr,col ]= np.nan
            
            print('deleted {} of {} of data to nan '.format(len(data[ data['value_happend_days']<=thr]),col))  
            del data['value_happend_days']            
    else:
        print('!!! not exists,col {} had been deleted '.format(col))
#to avoid overfit        
for col in ['addr1__card1','card1']:
    train_col = train[col]
    test_col = test[col]
    print('train: deleting un overlap data :{}'.format( len(train.loc[train[col].isin( test_col ) == False,col  ]) ))
    print('test: deleting un overlap data :{}'.format( len(test.loc[test[col].isin( train_col ) == False,col  ] ) ))
    train.loc[train[col].isin( test_col ) == False,col  ] = np.nan
    test.loc[test[col].isin( train_col ) == False,col  ] = np.nan




# dealing with time based cols

test = tiem_deta_date(test,'Transaction_dt')
test['TransDateStr'] = test['Transaction_dt'].map(lambda x:str(x)[:10])
train = tiem_deta_date(train,'Transaction_dt')
train['TransDateStr'] = train['Transaction_dt'].map(lambda x:str(x)[:10])
del train['Transaction_dt']
del test['Transaction_dt']



train['weekofyear'] = pd.to_datetime(train['TransDateStr']).dt.weekofyear
train['Year'] = pd.to_datetime(train['TransDateStr']).dt.year
train['Year-weekofyear'] = train['Year'].map(str) + "|" + train['weekofyear'].map(str)


test['weekofyear'] = pd.to_datetime(test['TransDateStr']).dt.weekofyear
test['Year'] = pd.to_datetime(test['TransDateStr']).dt.year
test['Year-weekofyear'] = test['Year'].map(str) + "|" + test['weekofyear'].map(str)

del train['weekofyear']
del train['Year']
del test['weekofyear']
del test['Year']
del train['TransDateStr']
del test['TransDateStr']


# D7
Ds = ['D3','D4','D5','D6','D8','D10','D11','D12','D13','D14','D15']



def out_lier_remove(df,col):
    gp =  df.groupby(['Year-weekofyear'])[col].quantile(.995)
    df['qt'] = df['Year-weekofyear'].map(gp)
    df['new_' + col] = df[col].copy()
    df.loc[ df[col] >=df['qt'],'new_' + col]=np.nan
    del df['qt']
    return df

for col in Ds:
    train = out_lier_remove(train,col)
    test = out_lier_remove(test,col)

base_on = 'Year-weekofyear'    
new_DS = ['new_' +col for col in Ds]    
df = pd.concat([train[[base_on] + new_DS],test[[base_on] + new_DS]])
aggs = {}
for d in new_DS:
    aggs[d] = ['mean']
    
gp = df.groupby([ base_on ]).agg(aggs)
gp.columns = pd.Index([ e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
gp = gp.sort_values([base_on]).reset_index(drop = True)


    

train = train.merge(gp[[base_on]+[c for c in gp.columns if 'mean' in c]],on=[base_on],how='left')
test = test.merge(gp[[base_on]+[c for c in gp.columns if 'mean' in c]],on=[base_on],how='left')
del gp
del df
for col in Ds:
    train[col ] = (train[col]-train['new_'+col + '_mean'])/train['new_'+col + '_mean']
    test[col] = (test[col]-test['new_'+col + '_mean'])/test['new_'+col + '_mean']
    del train['new_'+col + '_mean']
    del test['new_'+col + '_mean']
    

    #*************************************
    #to avoid overfit
    #*************************************



# del train[base_on]    
# del test[base_on]
le = LabelEncoder()
le.fit(list(train[base_on].astype(str).values) + list(test[base_on].astype(str).values))
train[base_on] = le.transform(list(train[base_on].astype(str).values))
test[base_on] = le.transform(list(test[base_on].astype(str).values))    

#########################################
# bacily data cleaning
#########################################
#*******************************end*******************************

for col in train.columns:
    if train[col].dtype == 'object':
        if col == 'TransactionID':
            continue
            
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))





# drop corr counters

gp = train[[c for c in train.columns if 'count' in c]].corr()
gp[gp>0.99]
cor_dic = {}
for col in gp.columns:
    cells = gp[gp[col]>=0.999].index.values
    cor_dic[col] = cells
drop_cols = []

for k,v in cor_dic.items():
    if k in drop_cols:
        continue
    dropEmp = []
    if len(v)==1:
        continue
    for c in v:
        if c==k:
            pass
        else:
            dropEmp.append( c )
    drop_cols = drop_cols+dropEmp    
    
# catboost model..............

from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
import catboost as cbt



# w2v_exception = [c for c in train.columns if 'domain_Sep_w2v' in c]

# 'addr1__card1']+time_base_col
feats = [c for c in train.columns if c not in ['isFraud', 'TransactionDT','TransactionID','TransDateStr','Year-weekofyear'] 
        and 'new' not  in c and c not in drop_cols]

cat_list = [c for c in feats if c in cats]
        # 

for cat in cat_list:
    train[cat] = train[cat].map(str)
    test[cat] = test[cat].map(str)



X_train = train[feats]
y = train['isFraud']
X_test = test[feats]
id_test = test['TransactionID'].values

print(X_train.shape,X_test.shape)
oof = np.zeros(X_train.shape[0])



prediction = np.zeros(X_test.shape[0])
seeds = [ 2048, 1024]
num_model_seed = 2
for model_seed in range(num_model_seed):
    oof_cat = np.zeros(X_train.shape[0])
    prediction_cat=np.zeros(X_test.shape[0])
    skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        print(index)
        print('on folder tracking ........')
        # print('*********[{}]*********'.format(folder_track_log_dir))
        print('on folder tracking ........')
        
        
        # fold_pred_file_ = folder_track_log_dir + '/model_seed_{}_Fold_{}_Done.npy'.format( model_seed,index )
        # if os.path.exists( fold_pred_file_ ):
        #     continue
        # # boosting_type='Plain', 
        train_x, test_x, train_y, test_y = X_train[feats].iloc[train_index], X_train[feats].iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        cbt_model = cbt.CatBoostClassifier(iterations=10000,learning_rate=0.1,max_depth=7,verbose=100,
                                      early_stopping_rounds=500,task_type='GPU',eval_metric='AUC',
                                      cat_features=cat_list)
        cbt_model.fit(train_x[feats], train_y,eval_set=(test_x[feats],test_y))
        
        oof_cat[test_index] += cbt_model.predict_proba(test_x)[:,1]
        prediction_cat += cbt_model.predict_proba(X_test[feats])[:,1]/5
        del cbt_model
        del train_x, test_x, train_y, test_y

        
    print('AUC',roc_auc_score(y,oof_cat))    
    oof += oof_cat / num_model_seed
    prediction += prediction_cat / num_model_seed
print('score',roc_auc_score(y,oof)) 

# roc_auc_score(y_valid, y_pred_valid)



sub.columns = ['TransactionID','old']
test['isFraud'] = prediction
sub = sub.merge(test[['TransactionID','isFraud']],on=['TransactionID'])
del sub['old']

# add D2
sub.to_csv('../submission/cat_submit_549_1001_2seed.csv',index=False


# In[ ]:




