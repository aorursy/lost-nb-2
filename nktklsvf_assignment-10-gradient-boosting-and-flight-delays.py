#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier


# In[2]:


train = pd.read_csv('../input/flight_delays_train.csv')
test = pd.read_csv('../input/flight_delays_test.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


places = pd.Series((test['Origin'].append(test['Dest']).append(train['Origin']).append(train['Dest'])).unique()).to_dict()
places_map = {v: k for k, v in places.items()}
places_map


# In[6]:


carriers = pd.Series((test['UniqueCarrier'].append(test['UniqueCarrier']).append(train['UniqueCarrier']).append(train['UniqueCarrier'])).unique()).to_dict()
carriers_map = {v: k for k, v in carriers.items()}
carriers


# In[7]:


def prepare_df(df):
    df_copy = df.copy()
    df_copy['Dest'] = df_copy['Dest'].map(places_map)
    df_copy['Origin'] = df_copy['Origin'].map(places_map)
    df_copy['UniqueCarrier'] = df_copy['UniqueCarrier'].map(carriers_map)
    df_copy['Month'] = df_copy['Month'].str.replace('c-', '').astype(int)
    df_copy['DayofMonth'] = df_copy['DayofMonth'].str.replace('c-', '').astype(int)
    df_copy['DayOfWeek'] = df_copy['DayOfWeek'].str.replace('c-', '').astype(int)
    return df_copy


# In[8]:


X_train, y_train = prepare_df(train), train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
X_train = X_train.drop(columns=['dep_delayed_15min'])
X_test = prepare_df(test)


# In[9]:


X_train_part, X_valid, y_train_part, y_valid =     train_test_split(X_train, y_train, 
                     test_size=0.3, random_state=17)
    
model = CatBoostClassifier(random_state=17, learning_rate=0.1, max_depth=5, verbose=False)

model.fit(X_train_part, y_train_part)
model_valid_pred = model.predict_proba(X_valid)[:, 1]

roc_auc_score(y_valid, model_valid_pred)


# In[10]:


model.fit(X_train, y_train)
model_test_pred = model.predict_proba(X_test)[:, 1]

pd.Series(model_test_pred, name='dep_delayed_15min').to_csv('xgb_2feat.csv', index_label='id', header=True)

