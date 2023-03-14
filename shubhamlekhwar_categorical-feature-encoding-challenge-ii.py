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


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import os
import gc
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
import tensorflow as tf


# In[3]:


train=pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
test=pd.read_csv('../input/cat-in-the-dat-ii/test.csv')


# In[4]:


print("training data size: ", train.shape)
print("training data size: ", test.shape)


# In[5]:


def create_model(data, features):
    inputs=[]
    output=[]
    
    for c in features:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))
        inp = layers.Input(shape=(1,))
        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
        out = layers.SpatialDropout1D(0.3)(out)
        out = layers.Reshape(target_shape=(embed_dim, ))(out)
        inputs.append(inp)
        output.append(out)
        
    x=layers.Concatenate()(output)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    y = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=y)
    return model


# In[6]:


def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)


# In[7]:


test["target"] = -1
data = pd.concat([train, test]).reset_index(drop=True)
features = [x for x in train.columns if x not in ["id", "target"]]
for feat in features:
    lbl_enc = LabelEncoder()
    data[feat] = lbl_enc.fit_transform(data[feat].fillna("-1").astype(str).values)
train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)
test_data = [test.loc[:, features].values[:, k] for k in range(test.loc[:, features].values.shape[1])]
oof_preds = np.zeros((len(train)))
test_preds = np.zeros((len(test)))

skf = StratifiedKFold(n_splits=50)
for train_index, test_index in skf.split(train, train.target.values):
    X_train, X_test = train.iloc[train_index, :], train.iloc[test_index, :]
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train, y_test = X_train.target.values, X_test.target.values
    model = create_model(data, features)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])
    X_train = [X_train.loc[:, features].values[:, k] for k in range(X_train.loc[:, features].values.shape[1])]
    X_test = [X_test.loc[:, features].values[:, k] for k in range(X_test.loc[:, features].values.shape[1])]
    
    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5,
                                 verbose=1, mode='max', baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                      patience=3, min_lr=1e-6, mode='max', verbose=1)
    
    model.fit(X_train,
              utils.to_categorical(y_train),
              validation_data=(X_test, utils.to_categorical(y_test)),
              verbose=1,
              batch_size=1024,
              callbacks=[es, rlr],
              epochs=100
             )
    valid_fold_preds = model.predict(X_test)[:, 1]
    test_fold_preds = model.predict(test_data)[:, 1]
    oof_preds[test_index] = valid_fold_preds.ravel()
    test_preds += test_fold_preds.ravel()
    print(metrics.roc_auc_score(y_test, valid_fold_preds))
    K.clear_session()


# In[8]:


tf.keras.models.save_model(model, '../output/kaggle/working/', overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)


# In[9]:


print("Overall AUC={}".format(metrics.roc_auc_score(train.target.values, oof_preds)))


# In[10]:


test_preds /= 50
test_ids = test.id.values
print("Saving submission file")
submission = pd.DataFrame.from_dict({
    'id': test_ids,
    'target': test_preds
})
submission.to_csv("submission.csv", index=False)


# In[11]:


train_df['target'].value_counts()


# In[12]:


target=train_df['target']
del train_df['target']


# In[13]:


train_df['nom_0'].value_counts()


# In[14]:


train_df['nom_1'].value_counts()


# In[15]:


train_df['nom_2'].value_counts()


# In[16]:


train_df['nom_3'].value_counts()


# In[17]:


for i in range(10):
    print(train_df['nom_{}'.format(i)].value_counts())
    print('\n')


# In[18]:


data=pd.concat([train_df, test_df])


# In[19]:


cols=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']


# In[20]:


mode = data.filter(cols).mode()
data[cols]=data[cols].fillna(mode.iloc[0])


# In[21]:


data['bin_3']=data['bin_3'].map({'F': 0, 'T':1})
data['bin_4']=data['bin_4'].map({'Y':1, 'N':0})


# In[22]:


data[['bin_0', 'bin_1', 'bin_2']]=data[['bin_0', 'bin_1', 'bin_2']].astype('int8')


# In[23]:


cols=[]
for i in range(10):
    cols.append("nom_{}".format(i))


# In[24]:


data=data.fillna(-1)


# In[25]:


data.reset_index(inplace=True)


# In[26]:


data=data.drop('id', axis=1)


# In[27]:


data=data.drop('index', axis=1)


# In[28]:


from sklearn.preprocessing import LabelEncoder

features=[i for i in data.columns]


# In[29]:


le=LabelEncoder()
for i in features:
    data[i]=le.fit_transform(data[i].astype(str))


# In[30]:


train_data=data[:600000]
test_data=data[600000:]


# In[31]:


X_train, X_test, y_train, y_test=train_test_split(train_data, target, test_size=.20)


# In[32]:


classifier=RandomForestClassifier(n_estimators=1200, max_depth=100, n_jobs=-1, verbose=2, max_features='sqrt')


# In[33]:


classifier.fit(X_train, y_train, )


# In[34]:


pred=classifier.predict(X_test)


# In[35]:


from sklearn.metrics import f1_score


# In[36]:


fs=f1_score(y_test, pred)


# In[37]:


fs


# In[38]:


import lightgbm as lgb


# In[39]:


param = {
    'num_leaves': 31,
    'objective': 'binary'
    'metric': ''}
param['metric'] = ['auc', 'binary_logloss']

