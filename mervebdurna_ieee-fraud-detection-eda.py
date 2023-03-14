#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import gc
import time
import datetime
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics, preprocessing

from sklearn.model_selection import KFold # StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA # KernelPCA

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.preprocessing import LabelEncoder


pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns',700)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

#plt.style.use('ggplot')
#color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
# pd.show_versions (as_json = False)


# In[2]:


def getCatFeatureDetail(df,cat_cols):
    cat_detail_dict = {} 
    for col in cat_cols:
        cat_detail_dict[col] = df[col].nunique()
    cat_detail_df = pd.DataFrame.from_dict(cat_detail_dict, orient='index', columns=['nunique'])
    print('There are ' + str(len(cat_cols)) + ' categorical columns.')
    print(cat_detail_df)
    

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
 
    

                        
def ploting_cnt_amt(df, col, lim=2000):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    total = len(df)
    
    plt.figure(figsize=(16,14))    
    plt.suptitle(f'{col} Distributions ', fontsize=24)
    
    plt.subplot(211)
    g = sns.countplot( x=col,  data=df, order=list(tmp[col].values))
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0,tmp['Fraud'].max()*1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    sizes = []
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=12) 
        
    g.set_ylim(0,max(sizes)*1.15)
    plt.show()


# In[3]:


# Load data



train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

# Fix column name 
fix_col_name = {testIdCol:trainIdCol for testIdCol, trainIdCol in zip(test_identity.columns, train_identity.columns)}
test_identity.rename(columns=fix_col_name, inplace=True)
    
# Reduce memory
train_transaction = reduce_mem_usage(train_transaction)
train_identity = reduce_mem_usage(train_identity)

test_transaction = reduce_mem_usage(test_transaction)
test_identity = reduce_mem_usage(test_identity)
    
# Merge (transaction - identity)
train = train_transaction.merge(train_identity, on='TransactionID', how='left')
test = test_transaction.merge(test_identity, on='TransactionID', how='left')

# Merge (X_train - X_test)
train_test = pd.concat([train, test], ignore_index=True)

print(f'train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')

del train_transaction, train_identity, test_transaction, test_identity; x = gc.collect()


# In[4]:


train_test = train_test.copy()
train = train.copy()
test = test.copy()


# In[5]:


train.head()


# In[6]:


train.isnull().sum().sort_values(ascending =False)


# In[7]:


train.dtypes


# In[8]:


train_fraud = train.loc[train['isFraud'] == 1]
train_non_fraud = train.loc[train['isFraud'] == 0]


# In[9]:


train['isFraud'].value_counts(normalize=True)


# In[10]:


print('  {:.2f}% of Transactions that are fraud in train_transaction '.format(train['isFraud'].mean() * 100))


# In[11]:


sns.countplot(x="isFraud", data=train).set_title('Distribution of Target')
plt.show()


# In[12]:


plt.figure(figsize=(15,5))
sns.distplot(train["TransactionDT"])
sns.distplot(test["TransactionDT"])
plt.title('train vs test TransactionDT distribution')
plt.show()


# In[13]:


plt.figure(figsize=(15,5))
sns.distplot(train_fraud["TransactionDT"], color='b', label='Fraud')
sns.distplot(train_non_fraud["TransactionDT"], color='r', label ='non-Fraud')
plt.title('Fraud vs non-Fraud TransactionDT distribution')
plt.legend()


# In[14]:


START_DATE = '2015-04-22'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

train_test['New_Date'] = train_test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
train_test['New_Date_YMD'] = train_test['New_Date'].dt.year.astype(str) + '-' + train_test['New_Date'].dt.month.astype(str) + '-' + train_test['New_Date'].dt.day.astype(str)
train_test['New_Date_YearMonth'] = train_test['New_Date'].dt.year.astype(str) + '-' + train_test['New_Date'].dt.month.astype(str)
train_test['New_Date_Weekday'] = train_test['New_Date'].dt.dayofweek
train_test['New_Date_Hour'] = train_test['New_Date'].dt.hour
train_test['New_Date_Day'] = train_test['New_Date'].dt.day


fig,ax = plt.subplots(4, 1, figsize=(16,15))

train_test.groupby('New_Date_Weekday')['isFraud'].mean().to_frame().plot.bar(ax=ax[0])
train_test.groupby('New_Date_Hour')['isFraud'].mean().to_frame().plot.bar(ax=ax[1])
train_test.groupby('New_Date_Day')['isFraud'].mean().to_frame().plot.bar(ax=ax[2])
train_test.groupby('New_Date_YearMonth')['isFraud'].mean().to_frame().plot.bar(ax=ax[3])


# In[15]:


print(pd.concat([train['TransactionAmt'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index(),
                 train_fraud['TransactionAmt'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index(), 
                 train_non_fraud['TransactionAmt'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index()],
                   axis=1, keys=['Total','Fraud', "No Fraud"]))


# In[16]:


print(' Fraud TransactionAmt mean      :  '+str(train_fraud['TransactionAmt'].mean()))
print(' Non - Fraud TransactionAmt mean:  '+str(train_non_fraud['TransactionAmt'].mean()))


# In[17]:


plt.figure(figsize=(15,5))
sns.distplot(train_test["TransactionAmt"].apply(np.log))
plt.title('Train - Test TransactionAmt distribution')
plt.show()


# In[18]:


plt.figure(figsize=(15,5))
sns.distplot(train_fraud["TransactionAmt"].apply(np.log), label = 'Fraud | isFraud = 1')
sns.distplot(train_non_fraud["TransactionAmt"].apply(np.log), label = 'non-Fraud | isFraud = 0')
plt.title('Fraud vs non-Fraud TransactionAmt distribution')
plt.legend()
plt.show()


# In[19]:


train['New_TransactionAmt_Bin'] = pd.qcut(train['TransactionAmt'],15)
train.groupby('New_TransactionAmt_Bin')[['isFraud']].mean()


# In[20]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

train['dist1'].plot(kind='hist',bins=5000,ax=ax1,title='dist1 distribution',logx=True)
train['dist2'].plot(kind='hist',bins=5000,ax=ax2,title='dist2 distribution',logx=True)

plt.show()


# In[21]:


cat_features = ['isFraud','ProductCD','addr1', 'addr2', 'P_emaildomain','R_emaildomain','DeviceType','DeviceInfo']
all_cat_features = cat_features+ [f'card{i}' for i in range(1,7)]+ [f'M{i}' for i in range(1,10)] + [f'id_{i}' for i in range(12,39)]

getCatFeatureDetail(train_test, cat_features)


# In[22]:


ploting_cnt_amt(train, 'ProductCD')


# In[23]:


train['addr1'].value_counts()


# In[24]:


train['addr2'].value_counts()


# In[25]:


train.loc[train['addr1'].isin(train['addr1'].value_counts()[train['addr1'].value_counts() <= 5000 ].index), 'addr1'] = "Others"
train.loc[train['addr2'].isin(train['addr2'].value_counts()[train['addr2'].value_counts() <= 50 ].index), 'addr2'] = "Others"

test.loc[test['addr1'].isin(test.addr1.value_counts()[test['addr1'].value_counts() <= 5000 ].index), 'addr1'] = "Others"
test.loc[test['addr2'].isin(test.addr2.value_counts()[test['addr2'].value_counts() <= 50 ].index), 'addr2'] = "Others"

train['addr1'].fillna("NoInf", inplace=True)
test['addr1'].fillna("NoInf", inplace=True)

train['addr2'].fillna("NoInf", inplace=True)
test['addr2'].fillna("NoInf", inplace=True)


# In[26]:


ploting_cnt_amt(train, 'addr1')


# In[27]:


ploting_cnt_amt(train, 'addr2')


# In[28]:


train['P_emaildomain'].value_counts()


# In[29]:


train.loc[train['P_emaildomain'].isin(['gmail.com', 'gmail']),'P_emaildomain'] = 'Google'

train.loc[train['P_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                         'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                         'yahoo.es']), 'P_emaildomain'] = 'Yahoo Mail'
train.loc[train['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                         'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                         'outlook.es', 'live.com', 'live.fr',
                                         'hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
train.loc[train['P_emaildomain'].isin(train['P_emaildomain']                                         .value_counts()[train.P_emaildomain.value_counts() <= 500 ]                                         .index), 'P_emaildomain'] = "Others"
train['P_emaildomain'].fillna("NoInf", inplace=True)


# In[30]:


ploting_cnt_amt(train, 'P_emaildomain')


# In[31]:


train['R_emaildomain'].value_counts()


# In[32]:


train.loc[train['R_emaildomain'].isin(['gmail.com', 'gmail']),'R_emaildomain'] = 'Google'

train.loc[train['R_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                             'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                             'yahoo.es']), 'R_emaildomain'] = 'Yahoo Mail'
train.loc[train['R_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                             'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                             'outlook.es', 'live.com', 'live.fr',
                                             'hotmail.fr']), 'R_emaildomain'] = 'Microsoft'
train.loc[train['R_emaildomain'].isin(train.R_emaildomain                                         .value_counts()[train['R_emaildomain'].value_counts() <= 300 ]                                         .index), 'R_emaildomain'] = "Others"
train['R_emaildomain'].fillna("NoInf", inplace=True)


# In[33]:


ploting_cnt_amt(train, 'R_emaildomain')


# In[34]:


train['DeviceType'].value_counts()


# In[35]:


ploting_cnt_amt(train, 'DeviceType')


# In[36]:


train['DeviceInfo'].value_counts()


# In[37]:


train['DeviceInfo'].value_counts()[train['DeviceInfo'].value_counts() > 1000 ], 'DeviceInfo'] = "Others"


# In[38]:


train['DeviceInfo'].value_counts().head(20).plot(kind='barh', figsize=(15, 5), title='Top 20 Devices in Train')
plt.show()


# In[39]:


train_test['DeviceInfo'] = train_test['DeviceInfo'].fillna('unknown_device').str.lower()
train_test['DeviceName'] = train_test['DeviceInfo'].str.split('/', expand=True)[0]

train_test.loc[train_test['DeviceName'].str.contains('SM', na=False), 'DeviceName'] = 'Samsung'
train_test.loc[train_test['DeviceName'].str.contains('SAMSUNG', na=False), 'DeviceName'] = 'Samsung'
train_test.loc[train_test['DeviceName'].str.contains('GT-', na=False), 'DeviceName'] = 'Samsung'
train_test.loc[train_test['DeviceName'].str.contains('Moto G', na=False), 'DeviceName'] = 'Motorola'
train_test.loc[train_test['DeviceName'].str.contains('Moto', na=False), 'DeviceName'] = 'Motorola'
train_test.loc[train_test['DeviceName'].str.contains('moto', na=False), 'DeviceName'] = 'Motorola'
train_test.loc[train_test['DeviceName'].str.contains('LG-', na=False), 'DeviceName'] = 'LG'
train_test.loc[train_test['DeviceName'].str.contains('rv:', na=False), 'DeviceName'] = 'RV'
train_test.loc[train_test['DeviceName'].str.contains('HUAWEI', na=False), 'DeviceName'] = 'Huawei'
train_test.loc[train_test['DeviceName'].str.contains('ALE-', na=False), 'DeviceName'] = 'Huawei'
train_test.loc[train_test['DeviceName'].str.contains('-L', na=False), 'DeviceName'] = 'Huawei'
train_test.loc[train_test['DeviceName'].str.contains('Blade', na=False), 'DeviceName'] = 'ZTE'
train_test.loc[train_test['DeviceName'].str.contains('BLADE', na=False), 'DeviceName'] = 'ZTE'
train_test.loc[train_test['DeviceName'].str.contains('Linux', na=False), 'DeviceName'] = 'Linux'
train_test.loc[train_test['DeviceName'].str.contains('XT', na=False), 'DeviceName'] = 'Sony'
train_test.loc[train_test['DeviceName'].str.contains('HTC', na=False), 'DeviceName'] = 'HTC'
train_test.loc[train_test['DeviceName'].str.contains('ASUS', na=False), 'DeviceName'] = 'Asus'

train_test.loc[train_test['DeviceName'].isin(train_test['DeviceName'].value_counts()[train_test['DeviceName'].value_counts() < 1000].index), 'DeviceName'] = "Others"
 


# In[40]:


ploting_cnt_amt(train_test, 'DeviceName')


# In[41]:


card_cols = [c for c in train.columns if 'card' in c]
train[card_cols].head()


# In[42]:


train_test[card_cols].isnull().sum()


# In[43]:


for col in card_cols:
    print(col+'  :' + str(train[col].nunique()))


# In[44]:


#f = lambda x: np.nan if x.isnull().all() else x.value_counts(dropna=False).index[0]
for col in ['card2','card3','card4','card5','card6']:
    train_test[col] = train_test.groupby(['card1'])[col].transform(lambda x: x.mode(dropna=False).iat[0])
    train_test[col].fillna(train_test[col].mode()[0], inplace=True)
    print(col+' has : '+str(train_test[col].isnull().sum())+' missing values')


# In[45]:


ploting_cnt_amt(train, 'card4')


# In[46]:


ploting_cnt_amt(train, 'card6')


# In[47]:


c_cols = [c for c in train if c[0] == 'C']
train[c_cols].head()


# In[48]:


train[c_cols].describe()


# In[49]:


train[c_cols].quantile([.01, .1, .25, .5, .75, .9, .99])


# In[50]:


#train[train['C6']>118.000]['isFraud'].mean()


# In[51]:


for col in c_cols:
    print('\n Fraud '+col+' mean    :  '+str(train_fraud[train_fraud[col]<=37.00][col].mean()))
    print(' Non - Fraud '+col+' mean:  '+str(train_non_fraud[train_non_fraud[col]<=37.00][col].mean()))


# In[52]:


d_cols = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14']
train[d_cols].head()


# In[53]:


train[d_cols].describe()


# In[54]:


for col in d_cols:
    plt.figure(figsize=(15,5))
    plt.scatter(train['TransactionDT'] ,train[col])
    plt.title(col + ' Vs TransactionDT')
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.show()


# In[55]:


msno.matrix(train[d_cols])


# In[56]:


m_cols = [c for c in train if c[0] == 'M']
for col in m_cols:
    ploting_cnt_amt(train, col, lim=2500)


# In[57]:


msno.matrix(train[m_cols]) 


# In[58]:


v_cols = [c for c in train if c[0] == 'V']
train[v_cols].head()


# In[59]:


train[v_cols].describe()


# In[60]:


v_cols = [c for c in train_test if c[0] == 'V']
v_nan_df = train_test[v_cols].isna()
nan_groups={}

for col in v_cols:
    cur_group = v_nan_df[col].sum()
    try:
        nan_groups[cur_group].append(col)
    except:
        nan_groups[cur_group]=[col]
del v_nan_df; x=gc.collect()

'''
for nan_cnt, v_group in nan_groups.items():
    train_test['New_v_group_'+str(nan_cnt)+'_nulls'] = nan_cnt
    sc = preprocessing.MinMaxScaler()
    pca = PCA(n_components=2)
    v_group_pca = pca.fit_transform(sc.fit_transform(train_test[v_group].fillna(-1)))
    train_test['New_v_group_'+str(nan_cnt)+'_pca0'] = v_group_pca[:,0]
    train_test['New_v_group_'+str(nan_cnt)+'_pca1'] = v_group_pca[:,1]
'''


# In[61]:


def plot_corr(v_cols):
    cols = v_cols + ['TransactionDT']
    plt.figure(figsize=(15,15))
    sns.heatmap(train[cols].corr(),cmap='RdBu_r', annot=True, center=0.0)
    plt.title(v_cols[0]+' - '+v_cols[-1],fontsize=14)
    plt.show()
    


# In[62]:


for k,v in nan_groups.items():
    plot_corr(v)


# In[63]:


id_cols = [c for c in train_test if c[:2] == 'id']

id_num_cols=id_cols[:11]
id_cat_cols=id_cols[11:]


# In[64]:


train[id_num_cols].describe()


# In[65]:


for col in id_num_cols:
    print('\n'+col)
    print(' Fraud mean    :  ' + str(train_fraud[col].mean()))
    print(' Non - Fraud mean:  ' + str(train_non_fraud[col].mean()))


# In[66]:


getCatFeatureDetail(train,id_cat_cols)


# In[67]:


for col in  ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29']:
    ploting_cnt_amt(train, col, lim=2500)

