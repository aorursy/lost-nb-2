#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

from tqdm import tqdm_notebook

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
# train_transaction = pd.read_csv('train_transaction.csv')
# train_identity = pd.read_csv('train_identity.csv')
# test_transaction = pd.read_csv('test_transaction.csv')
# test_identity = pd.read_csv('test_identity.csv')


# In[3]:


train_transaction.head()


# In[4]:


train_identity.head()


# In[5]:


train_df = train_transaction.merge(train_identity, how='left', left_on='TransactionID', right_on='TransactionID')


# In[6]:


train_df.shape


# In[7]:


del train_transaction
del train_identity


# In[8]:


test_df = test_transaction.merge(test_identity, how='left', left_on='TransactionID', right_on='TransactionID')


# In[9]:


del test_transaction
del test_identity


# In[10]:


one_value_cols = [col for col in train_df.columns if train_df[col].nunique() <= 1]
one_value_cols_test = [col for col in test_df.columns if test_df[col].nunique() <= 1]

many_null_cols = [col for col in train_df.columns if train_df[col].isnull().sum() / train_df.shape[0] > 0.9]
many_null_cols_test = [col for col in test_df.columns if test_df[col].isnull().sum() / test_df.shape[0] > 0.9]

big_top_value_cols = [col for col in train_df.columns if train_df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test_df.columns if test_df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))
cols_to_drop.remove('isFraud')
print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))

train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)


# In[11]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


# In[12]:


cat_features = ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']                 + [x for x in train_df.columns if 'card' in x]                 + [x for x in train_df.columns if 'M' in x]                 + [x for x in train_df.columns if ('id_' in x) and (int(x[-2:]) > 11)]


# In[13]:


cat_features


# In[14]:


for x in cat_features:
    s = {y for y in set(train_df[x].unique()).symmetric_difference(set(test_df[x].unique())) if y==y}
    l = train_df[x].nunique()
    d = train_df[x].dtypes
    if (len(s)/l > 0.1) and (l > 10) and (d in numerics):
        print('Feature %s; Type %s; Diff %d; Train len %d; Percent diff %.3f' % (x, d, len(s), l, 100*len(s)/l))


# In[15]:


cat_features= [x for x in cat_features if x not in ['addr1', 'addr2', 'card1', 'card3',                                                    'card5', 'id_13', 'id_14', 'id_17', 'id_19', 'id_20']]


# In[16]:


corrs = train_df.drop(cat_features+['isFraud', 'TransactionID'], axis=1).corr()


# In[17]:


tmp = corrs.abs().mask(np.eye(len(corrs), dtype=bool))
tmp = tmp[tmp > 0.8]
s = tmp.unstack().dropna()

corr_df = pd.DataFrame(s.index.tolist(), columns=['feature1', 'feature2'])


# In[18]:


corr_features = corr_df.feature1.unique().tolist()


# In[19]:


aggr_corr_df = corr_df.groupby('feature1').apply(lambda x:frozenset(list(x['feature1'])[:1]+list(x['feature2']))).reset_index()
aggr_corr_df.columns = ['feature1', 'features']
aggr_corr_df.drop_duplicates(subset='features', inplace=True)
aggr_corr_df['features'] = aggr_corr_df['features'].apply(list)
aggr_corr_df.shape


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[21]:


lr = LogisticRegression(solver='lbfgs')
feature_gini = {}

for col in tqdm_notebook(train_df[corr_features].columns):
    tmp = pd.concat([train_df[[col]], train_df.isFraud], axis=1)
    tmp.dropna(inplace=True)
    lr.fit(tmp.drop('isFraud', axis=1), tmp.isFraud)
    y_pred_col = lr.predict_proba(tmp.drop('isFraud', axis=1))[:,1]
    roc_auc_col = roc_auc_score(tmp.isFraud, y_pred_col)
    feature_gini[col] = np.abs((2*roc_auc_col - 1)*100)


# In[22]:


def max_in_list(list_):
    return list_.index(max(list_))


# In[23]:


aggr_corr_df['feature_gini'] = aggr_corr_df.features.apply(lambda x: x[max_in_list([feature_gini.get(i, 0) for i in x])])


# In[24]:


good_features = aggr_corr_df.feature_gini.unique().tolist() + [x for x in train_df.columns if x not in corr_features]


# In[25]:


for x in cat_features:
    if train_df[x].dtypes in numerics:
        train_df.loc[train_df[x].isnull(), x] = train_df[x].min() - 1000
        test_df.loc[test_df[x].isnull(), x] = train_df[x].min() - 1000
    else:
        train_df.loc[train_df[x].isnull(), x] = 'undefined'
        test_df.loc[test_df[x].isnull(), x] = 'undefined'


# In[26]:


for x in cat_features:
    le = LabelEncoder()
    le.fit(list(train_df[x].astype(str).values) + list(test_df[x].astype(str).values))
    train_df[x] = le.transform(list(train_df[x].astype(str).values))
    test_df[x] = le.transform(list(test_df[x].astype(str).values)) 


# In[27]:


# data is unbalanced, hence we should initialize class weights
class_weights = [1, train_df.isFraud.value_counts()[0]/train_df.isFraud.value_counts()[1]]


# In[28]:


cb = CatBoostClassifier(eval_metric='AUC',                        class_weights=class_weights,                        n_estimators=1500,                        random_seed=42,                        one_hot_max_size=10,                        silent=True)


# In[29]:


from sklearn.model_selection import StratifiedKFold


# In[30]:


skf = StratifiedKFold(10)


# In[31]:


roc_auc_scores = []
i = 0
for train_index, val_index in skf.split(train_df[good_features].drop('isFraud', axis=1), train_df.isFraud):
    X_train, X_val = train_df[good_features].drop('isFraud', axis=1).iloc[train_index], train_df[good_features].drop('isFraud', axis=1).iloc[val_index]
    y_train, y_val = train_df.isFraud.iloc[train_index], train_df.isFraud.iloc[val_index]
    cb.fit(X_train, y_train, cat_features=cat_features)
    y_pred = cb.predict_proba(X_val)[:,1]
    roc_auc = roc_auc_score(y_val, y_pred)
    roc_auc_scores.append(roc_auc)
    print('AUC in Fold #' + str(i) + ': ' + str(roc_auc))
    i+=1
print('Mean AUC: ' + str(np.mean(roc_auc_scores)))


# In[32]:


feature_dict = {'Features': cb.feature_names_, 'Importance': cb.feature_importances_}


# In[33]:


feature_imp = pd.DataFrame(feature_dict).sort_values(by=['Importance'], ascending=False)


# In[34]:


plt.figure(figsize=(10,7))
df_imp = feature_imp.head(20)
sns.barplot(y=df_imp['Features'], x=df_imp['Importance'], palette='coolwarm_r')


# In[35]:


cb.fit(train_df.drop('isFraud', axis=1), train_df.isFraud, cat_features=cat_features)


# In[36]:


y_pred = cb.predict_proba(test_df)


# In[37]:


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')


# In[38]:


submission['isFraud'] = y_pred


# In[39]:


submission.to_csv('submission.csv', index=False)

