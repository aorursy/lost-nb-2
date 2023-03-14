#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import datetime

from tensorflow.keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import History

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score
# import codecs
# import re

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 25)


# In[2]:


t_start = datetime.datetime.now()

train_df = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
test_df = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip')

sample_submission = pd.read_csv('/kaggle/input/sf-crime/sampleSubmission.csv.zip')


# In[3]:


def preprocess(df):
    
    df['Dates'] = df['Dates'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['Year'] = df['Dates'].apply(lambda x: x.year)
    df['Month'] = df['Dates'].apply(lambda x: x.month)
    df['Day'] = df['Dates'].apply(lambda x: x.day)
    df['Hour'] = df['Dates'].apply(lambda x: x.hour)
        
    return df


# In[4]:


train_df = preprocess(train_df)
test_df = preprocess(test_df)


# In[5]:


train_df.drop_duplicates(inplace=True)


# In[6]:


drop_cols = ['Dates', 'Descript', 'Resolution', 'Id']

for col in drop_cols:
    if col in train_df.columns:
        train_df.drop(col, axis=1, inplace=True)
    if col in test_df.columns:
        test_df.drop(col, axis=1, inplace=True)
        
X = train_df.drop('Category', axis=1)
X_test = test_df


# In[7]:


y_cats = train_df['Category']
unique_cats = np.sort(y_cats.unique())

y = np.zeros((y_cats.shape[0], 39))
for idx, target in enumerate(list(y_cats)):
    y[idx, np.where(unique_cats == target)] = 1

y = pd.DataFrame(y, columns = unique_cats)


# In[8]:


X['train'] = 1
X_test['train'] = 0

combined = pd.concat([X, X_test])

for col in combined.columns:
    if combined.dtypes[col] == 'object':
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])
        
X = combined[combined['train'] == 1]
X.drop(['train'], axis=1, inplace=True)
X_test = combined[combined['train'] == 0]
X_test.drop(['train'], axis=1, inplace=True)  


# In[9]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# In[10]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1)


# In[11]:


The model developed below is designed to serve as a starting point for further development and tuning. The hyperparameters chosen are mostly arbitrary and I haven't attempted to do any tuning. Here are some hyperparameters we can consider modifying to further refine the model:

* Number of Hidden Layers (currently 3, chosen arbitrarily) 
* Number of Neurons per Layer (currently arbitrary powers of 2)
* Dropout Rates for each Hidden Layer (currently 0.5 for all 3)
* Optimizer (adam is usually best but sometimes it's not, it's worth trying some others)
* Batch Size (currently 256, an arbitrary power of 2)
* Learning Rate 
* Weight Decay
* Momentum 
* Regularization (L1 and/or L2, incorporate into loss function)

I also want to add cross validation. 


# In[12]:


def get_model(x_tr, y_tr, x_val, y_val):
    K.clear_session()
    inp = Input(shape = (x_tr.shape[1],))
    
    dl_1 = 1024  
    drop_1 = 0.5
    dl_2 = 512 
    drop_2 = 0.5 
    dl_3 = 256
    drop_3 = 0.5 
    
    x = Dense(dl_1, input_dim=X.shape[1], activation='relu')(inp) 
    x = Dropout(drop_1)(x)
    x = BatchNormalization()(x)
    x = Dense(dl_2, activation='relu')(x)
    x = Dropout(drop_2)(x)
    x = BatchNormalization()(x)
    x = Dense(dl_3, activation='relu')(x)
    x = Dropout(drop_3)(x)
    x = BatchNormalization()(x)
    
    out = Dense(39, activation='softmax')(x)
    
    model = Model(inp,out)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
        
    
    bsz = 256
    steps = x_tr.shape[0]/bsz
    
    es = EarlyStopping(monitor='loss', patience=10) 

    y_tr = np.asarray(y_tr)
    y_val = np.asarray(y_val)
    history = model.fit(x_tr, y_tr, callbacks=[es], epochs=50, batch_size=bsz, verbose=1)

    return model, history.history['loss'][-1]


# In[13]:


mod, loss = get_model(X_train, y_train, X_val, y_val)


# In[14]:


loss


# In[15]:


preds = mod.predict(X_test)


# In[16]:


sub_df = pd.DataFrame(preds, columns=unique_cats)


# In[17]:


sub_df.index = sub_df.index.set_names(['Id'])
sub_df.reset_index(drop=False, inplace=True)


# In[18]:


sub_df.to_csv('sub_file_area.csv', index=False)


# In[19]:


t_final = datetime.datetime.now()


# In[20]:


print('Total Execution Time:  {}'.format(t_final - t_start))

