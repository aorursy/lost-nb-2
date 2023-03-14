#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN


# In[ ]:


train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv',index_col='TransactionID')
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv',index_col='TransactionID')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv',index_col='TransactionID') 
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv',index_col='TransactionID')
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')


# In[ ]:


train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction,test_identity,on='TransactionID', how='left')


# In[ ]:


del (test_identity,test_transaction,train_identity,train_transaction)


# In[ ]:


train.memory_usage(deep=True).sum()


# In[ ]:


train.isFraud.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


train.describe(include='all')


# In[ ]:


train.columns


# In[ ]:


train.dtypes.value_counts()


# In[ ]:


features_object=train.select_dtypes(include=['object']).columns
print(features_object)


# In[ ]:


pd.crosstab(index=train['isFraud'], columns=train['ProductCD'], normalize = True, margins=True, margins_name='Total')

ProductCD has 5 types, among which type "C" and "W" has most fraud cases.
# In[ ]:


sns.countplot(x="ProductCD", hue = "isFraud", data=train,  palette="Set1")


# In[ ]:


pd.crosstab(index=train['isFraud'], columns=train['card4'], normalize = True, margins=True, margins_name="total").round(3)*100


# In[ ]:


sns.countplot(x="card4", hue = "isFraud", data=train,  palette="Set2")


# In[ ]:


pd.crosstab(index=train['isFraud'], columns=train['card6'], normalize = True, margins=True, margins_name="total").round(2)*100


# In[ ]:


sns.countplot(x="card6", hue = "isFraud", data=train,  palette="Set2")


# In[ ]:



pd.crosstab([train.ProductCD, train.card6,train.card4],[train.isFraud], 
    rownames=['card4','ProductCD', 'card6'], 
    colnames=[ 'isFraud'], normalize=True).round(4)*100


# In[ ]:


f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(pd.crosstab([train.ProductCD, train.card6],[train.card4,train.isFraud], 
    rownames=['ProductCD', 'card6'], 
    colnames=['card4', 'isFraud'], normalize=True).round(4)*100,cmap="YlGnBu",linewidth=0.5 ,square=False,center=0, annot=True, cbar_kws={"shrink": .5})


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(20,20))
sns.countplot(y="P_emaildomain",hue="isFraud", ax=ax[0], data=train)
ax[0].set_title('P_emaildomain')
sns.countplot(y="R_emaildomain",hue="isFraud", ax=ax[1], data=train)
ax[1].set_title('R_emaildomain')


# In[ ]:


id_features = train[['id_12','id_15','id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33',
       'id_34', 'id_35', 'id_36', 'id_37', 'id_38']]


# In[ ]:


id_features.describe()


# In[ ]:


id_features.isnull().sum()*100/len(id_features)


# In[ ]:


m_features = [c for c in train if c[0] == 'M']
train[m_features].describe()


# In[ ]:


def bar_plot(col, data, hue=None):
    f, ax = plt.subplots(figsize = (20, 5))
    sns.countplot(x=col, hue=hue, data=data, alpha=0.5)
    


# In[ ]:


m_features= train.loc[:,'M1':'M9']   
for col in m_features:
    bar_plot(col, train, hue='isFraud')
        


# In[ ]:


pd.crosstab(train.isFraud,train.DeviceType).plot(kind='bar')


# In[ ]:


train.DeviceInfo.value_counts()


# In[ ]:


float_dtypes = train.select_dtypes(include='bool')
features_float=train.select_dtypes(include=['float']).columns
print(features_float)


# In[ ]:


D_features = train.loc[:,'D1':'D14']
D_features.describe()


# In[ ]:


colormap = plt.cm.Spectral
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', size=15)
sns.heatmap(D_features.corr(),linewidths=0.1,vmax=1.0, 
           square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


[c for c in train if c[0] == 'C']


# In[ ]:


C_features = train.loc[:,'C1':'C14']


# In[ ]:


colormap = plt.cm.seismic
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', size=15)
sns.heatmap(C_features.corr(),linewidths=0.1,vmax=1.0, 
           square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


id_ = train.loc[:,'id_01':'id_32']


# In[ ]:


colormap = plt.cm.RdYlBu
plt.figure(figsize=(30,20))
plt.title('Pearson Correlation of Features', size=15)
sns.heatmap(id_.corr(),linewidths=0.1,vmax=1.0, 
           square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


v_feature = [v for v in train if v[0] == 'V']
print(train[v_feature].shape)


# In[ ]:


train[v_feature].describe()


# In[ ]:


def missing_val(train):
        mis_val = train.isnull().sum()
        mis_val_per =train.isnull().sum()*100 / len(train)
        missing_table = pd.concat([mis_val, mis_val_per], axis=1)
        
        mis_val_table = missing_table.rename(columns = {0 : 'Missing_val', 1 : '% of Missing_val'})
        
        mis_val_table = mis_val_table.sort_values('% of Missing_val', ascending=False).round(1)
        
        return mis_val_table
    
missing_val(train).head(10)


# In[ ]:


y_train = train['isFraud'].copy()
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()


# In[ ]:


for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))


# In[ ]:


X_train.fillna(-999, inplace = True)
X_test.fillna(-999, inplace = True)


# In[ ]:


from sklearn.model_selection import TimeSeriesSplit,KFold
n_fold = 5
folds = KFold(n_splits=n_fold,shuffle=True)

print(folds)


# In[ ]:


for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
    xgbclf = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=9,
        learning_rate=0.048,
        subsample=0.85,
        colsample_bytree=0.85,
        missing=-999,
        tree_method='gpu_hist',
        reg_alpha=0.15,
        reg_lamdba=0.85
    )


# In[ ]:


X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]


# In[ ]:


y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]


# In[ ]:


xgbclf.fit(X_train_,y_train_)
    


# In[ ]:


del X_train_,y_train_
    pred=xgbclf.predict_proba(X_test)[:,1]
    val=xgbclf.predict_proba(X_valid)[:,1]
    del xgbclf, X_valid
    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))
    del val,y_valid
    xgb_submission['isFraud'] = xgb_submission['isFraud']+pred/n_fold
    del pred
    gc.collect()


# In[ ]:




